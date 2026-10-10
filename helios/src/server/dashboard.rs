//! Terminal rendering samples counters without contacting the execution worker.

use std::{
    collections::VecDeque,
    io::{self, IsTerminal},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, RecvTimeoutError},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crossterm::{
    cursor::{Hide, Show},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols::Marker,
    text::{Line, Span},
    widgets::{Axis, Block, BorderType, Chart, Dataset, GraphType, Paragraph},
};

use super::metrics::{Metrics, Snapshot};

const REFRESH: Duration = Duration::from_millis(500);
const WINDOW: Duration = Duration::from_secs(2);
const HISTORY: f64 = 120.0;
const BACKGROUND: Color = Color::Rgb(13, 18, 28);
const MUTED: Color = Color::Rgb(126, 146, 166);
const TEXT: Color = Color::Rgb(221, 232, 242);
const CYAN: Color = Color::Rgb(84, 214, 223);
const GREEN: Color = Color::Rgb(145, 222, 139);
const PURPLE: Color = Color::Rgb(191, 157, 245);
const GOLD: Color = Color::Rgb(244, 196, 112);
static VISIBLE: AtomicBool = AtomicBool::new(false);

#[cfg(feature = "python")]
pub(super) fn restore_for_exit() {
    if VISIBLE.load(Ordering::Relaxed) {
        let leave = b"\x1b[?1049l";
        // SAFETY: Signal handlers may only use async-signal-safe terminal writes.
        unsafe {
            libc::write(libc::STDERR_FILENO, leave.as_ptr().cast(), leave.len());
        }
    }
}

pub(super) struct Dashboard {
    stop: mpsc::Sender<()>,
    thread: Option<JoinHandle<()>>,
}

impl Dashboard {
    pub fn start(
        metrics: Arc<Metrics>,
        title: String,
        learned_drafter: bool,
    ) -> io::Result<Option<Self>> {
        if !io::stderr().is_terminal()
            || std::env::var("TERM").is_ok_and(|term| term == "dumb")
            || VISIBLE
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                .is_err()
        {
            return Ok(None);
        }
        let (stop, receiver) = mpsc::channel();
        let thread = thread::Builder::new()
            .name("helios-dashboard".into())
            .spawn(move || {
                // Linux priorities are per-thread; rendering yields to serving under CPU pressure.
                #[cfg(target_os = "linux")]
                unsafe {
                    libc::setpriority(libc::PRIO_PROCESS, 0, 10);
                }
                let _visible = Visible;
                let result = (|| -> io::Result<()> {
                    let _screen = Screen;
                    execute!(io::stderr(), EnterAlternateScreen, Hide)?;
                    let mut terminal = Terminal::new(CrosstermBackend::new(io::stderr()))?;
                    terminal.clear()?;
                    let start = Instant::now();
                    let mut history = History::new(start, metrics.snapshot(learned_drafter));
                    loop {
                        history.sample(Instant::now(), metrics.snapshot(learned_drafter));
                        terminal.draw(|frame| render(frame, &title, &history))?;
                        match receiver.recv_timeout(REFRESH) {
                            Err(RecvTimeoutError::Timeout) => {}
                            Ok(()) | Err(RecvTimeoutError::Disconnected) => return Ok(()),
                        }
                    }
                })();
                if let Err(error) = result {
                    eprintln!("Helios dashboard stopped: {error}");
                }
            });
        match thread {
            Ok(thread) => Ok(Some(Self {
                stop,
                thread: Some(thread),
            })),
            Err(error) => {
                VISIBLE.store(false, Ordering::Release);
                Err(error)
            }
        }
    }
}

impl Drop for Dashboard {
    fn drop(&mut self) {
        let _ = self.stop.send(());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

struct Visible;
impl Drop for Visible {
    fn drop(&mut self) {
        VISIBLE.store(false, Ordering::Release);
    }
}

struct Screen;
impl Drop for Screen {
    fn drop(&mut self) {
        let _ = execute!(io::stderr(), Show, LeaveAlternateScreen);
    }
}

#[derive(Clone, Copy, Default)]
struct Rates {
    prefill: f64,
    generated: f64,
    draft: f64,
    verify: f64,
    acceptance: Option<f64>,
}

struct History {
    start: Instant,
    previous: VecDeque<(Instant, Snapshot)>,
    points: VecDeque<(f64, Rates)>,
    snapshot: Snapshot,
    rates: Rates,
}

impl History {
    fn new(start: Instant, snapshot: Snapshot) -> Self {
        Self {
            start,
            previous: VecDeque::from([(start, snapshot)]),
            points: VecDeque::new(),
            snapshot,
            rates: Rates::default(),
        }
    }

    fn sample(&mut self, now: Instant, snapshot: Snapshot) {
        while self.previous.len() > 1 && now.duration_since(self.previous[1].0) >= WINDOW {
            self.previous.pop_front();
        }
        let (then, before) = self.previous[0];
        let elapsed = now.duration_since(then).as_secs_f64();
        let per_second = |after: u64, before: u64| {
            if elapsed > 0.0 {
                after.saturating_sub(before) as f64 / elapsed
            } else {
                0.0
            }
        };
        let rounds = snapshot.draft_rounds.saturating_sub(before.draft_rounds);
        self.rates = Rates {
            prefill: per_second(snapshot.prefill_tokens, before.prefill_tokens),
            generated: per_second(snapshot.generated_tokens, before.generated_tokens),
            draft: per_second(snapshot.draft_steps, before.draft_steps),
            verify: per_second(snapshot.verify_steps, before.verify_steps),
            acceptance: (rounds > 0).then(|| {
                snapshot
                    .accepted_drafts
                    .saturating_sub(before.accepted_drafts) as f64
                    / rounds as f64
            }),
        };
        let time = now.duration_since(self.start).as_secs_f64();
        self.points.push_back((time, self.rates));
        while self
            .points
            .front()
            .is_some_and(|(at, _)| time - at > HISTORY)
        {
            self.points.pop_front();
        }
        self.previous.push_back((now, snapshot));
        self.snapshot = snapshot;
    }
}

fn rate(value: f64) -> String {
    if value >= 10_000.0 {
        format!("{:.1}k", value / 1000.0)
    } else {
        format!("{value:.1}")
    }
}

fn render(frame: &mut Frame, title: &str, history: &History) {
    let area = frame.area();
    frame.render_widget(
        Block::default().style(Style::default().bg(BACKGROUND).fg(TEXT)),
        area,
    );
    let s = history.snapshot;
    let r = history.rates;
    let acceptance = r
        .acceptance
        .map_or_else(|| "—".into(), |v| format!("{v:.2}"));
    if area.width < 64 || area.height < 22 {
        frame.render_widget(
            Paragraph::new(vec![
                Line::from(" FLARE / HELIOS")
                    .style(Style::default().fg(CYAN).add_modifier(Modifier::BOLD)),
                Line::from(format!(" {title}")),
                Line::from(format!(" Prefill      {} tok/s", rate(r.prefill))),
                Line::from(format!(" Generation   {} tok/s", rate(r.generated))),
                Line::from(format!(" Drafter      {} steps/s", rate(r.draft))),
                Line::from(format!(" Verify       {} steps/s", rate(r.verify))),
                Line::from(format!(" Acceptance   {acceptance} drafts/round")),
                Line::from(format!(" Active {}  Queued {}", s.active, s.pending)),
                Line::from(" Enlarge terminal for history plots.")
                    .style(Style::default().fg(MUTED)),
                Line::from(" Ctrl-C stops server").style(Style::default().fg(MUTED)),
            ]),
            area,
        );
        return;
    }
    let rows = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(8),
        Constraint::Min(8),
        Constraint::Length(2),
        Constraint::Length(1),
    ])
    .split(area);
    let age = history.points.back().map_or(0, |(t, _)| *t as u64);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(vec![
                Span::styled(
                    " FLARE ",
                    Style::default()
                        .fg(BACKGROUND)
                        .bg(CYAN)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" / HELIOS  ", Style::default().fg(CYAN)),
                Span::styled(
                    if s.active > 0 {
                        "● RUNNING"
                    } else {
                        "● READY"
                    },
                    Style::default().fg(GREEN),
                ),
                Span::styled(
                    format!("    {:02}:{:02}:{:02}", age / 3600, age / 60 % 60, age % 60),
                    Style::default().fg(MUTED),
                ),
            ]),
            Line::from(format!(" {title}")).style(Style::default().fg(MUTED)),
        ]),
        rows[0],
    );
    let top =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).split(rows[1]);
    let bottom =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).split(rows[2]);
    plot(
        frame,
        top[0],
        &format!(" PREFILL  {} tok/s ", rate(r.prefill)),
        history,
        &[("", CYAN, |r| r.prefill)],
    );
    plot(
        frame,
        top[1],
        &format!(" GENERATION  {} tok/s ", rate(r.generated)),
        history,
        &[("", GREEN, |r| r.generated)],
    );
    plot(
        frame,
        bottom[0],
        &format!(
            " DRAFT {} / VERIFY {} steps/s ",
            rate(r.draft),
            rate(r.verify)
        ),
        history,
        &[
            ("draft", PURPLE, |r| r.draft),
            ("verify", GOLD, |r| r.verify),
        ],
    );
    plot(
        frame,
        bottom[1],
        &format!(" ACCEPTANCE  {acceptance} drafts/round "),
        history,
        &[("", GOLD, |r| r.acceptance.unwrap_or(0.0))],
    );
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(format!(
                " ACTIVE {}   QUEUED {}   COMPLETE {}   FAILED {}",
                s.active, s.pending, s.completed, s.failed
            )),
            Line::from(format!(
                " PREFILL {}   GENERATED {}   REUSED {}   KV PAGES {} cached / {} free",
                s.prefill_tokens, s.generated_tokens, s.reused_tokens, s.cached_pages, s.free_pages
            ))
            .style(Style::default().fg(MUTED)),
        ]),
        rows[3],
    );
    frame.render_widget(
        Paragraph::new(
            " 2s rates · 120s history · acceptance excludes bonus token · Ctrl-C stops server",
        )
        .style(Style::default().fg(MUTED)),
        rows[4],
    );
}

type Series = (&'static str, Color, fn(Rates) -> f64);

fn plot(frame: &mut Frame, area: Rect, title: &str, history: &History, series: &[Series]) {
    let now = history.points.back().map_or(0.0, |(time, _)| *time);
    let points: Vec<Vec<_>> = series
        .iter()
        .map(|(_, _, value)| {
            history
                .points
                .iter()
                .map(|(time, rates)| (time - now, value(*rates)))
                .collect()
        })
        .collect();
    let peak = points
        .iter()
        .flatten()
        .map(|(_, value)| *value)
        .fold(1.0, f64::max)
        * 1.15;
    let datasets = series
        .iter()
        .zip(&points)
        .map(|((name, color, _), data)| {
            let dataset = Dataset::default()
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(*color))
                .data(data);
            if name.is_empty() {
                dataset
            } else {
                dataset.name(*name)
            }
        })
        .collect();
    frame.render_widget(
        Chart::new(datasets)
            .block(
                Block::bordered()
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(MUTED))
                    .title(title.to_owned())
                    .title_style(Style::default().fg(TEXT).add_modifier(Modifier::BOLD)),
            )
            .style(Style::default().bg(BACKGROUND))
            .x_axis(
                Axis::default()
                    .bounds([-HISTORY, 0.0])
                    .labels(["−120s", "now"])
                    .style(Style::default().fg(MUTED)),
            )
            .y_axis(
                Axis::default()
                    .bounds([0.0, peak])
                    .labels(["0".to_owned(), rate(peak)])
                    .style(Style::default().fg(MUTED)),
            ),
        area,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;

    #[test]
    fn rates_use_elapsed_time_and_weight_acceptance_by_request_rounds() {
        let start = Instant::now();
        let mut history = History::new(start, Snapshot::default());
        let first = Snapshot {
            prefill_tokens: 100,
            generated_tokens: 8,
            draft_steps: 7,
            verify_steps: 1,
            accepted_drafts: 6,
            draft_rounds: 2,
            ..Default::default()
        };
        history.sample(start + Duration::from_secs(1), first);
        let last = Snapshot {
            prefill_tokens: 300,
            generated_tokens: 10,
            draft_steps: 8,
            verify_steps: 2,
            accepted_drafts: 7,
            draft_rounds: 3,
            ..Default::default()
        };
        history.sample(start + Duration::from_secs(2), last);
        assert_eq!(history.rates.prefill, 150.0);
        assert_eq!(history.rates.generated, 5.0);
        assert_eq!(history.rates.draft, 4.0);
        assert_eq!(history.rates.verify, 1.0);
        assert_eq!(history.rates.acceptance, Some(7.0 / 3.0));
        for tick in 3..=130 {
            history.sample(start + Duration::from_secs(tick), last);
        }
        assert_eq!(history.rates.generated, 0.0);
        assert_eq!(history.rates.acceptance, None);
        assert!(history.points.len() <= 121);
        assert!(history.previous.len() <= 3);
    }

    #[test]
    fn terminal_renders_braille_and_handles_resize_without_nonfinite_rates() {
        let start = Instant::now();
        let mut history = History::new(start, Snapshot::default());
        history.sample(start, Snapshot::default());
        for tick in 1..=120 {
            history.sample(
                start + Duration::from_secs(tick),
                Snapshot {
                    generated_tokens: tick * tick,
                    prefill_tokens: tick * 100,
                    draft_steps: tick * 12,
                    verify_steps: tick * 3,
                    accepted_drafts: tick * 7,
                    draft_rounds: tick * 3,
                    ..Default::default()
                },
            );
        }
        for (width, height) in [(120, 35), (80, 24), (40, 12), (1, 1), (0, 0)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal
                .draw(|frame| render(frame, "Qwen · 127.0.0.1:8000", &history))
                .unwrap();
            let content: String = terminal
                .backend()
                .buffer()
                .content
                .iter()
                .map(|cell| cell.symbol())
                .collect();
            assert!(!content.contains("NaN"));
            assert!(!content.contains("inf"));
            if width >= 80 {
                assert!(content.contains("PREFILL"));
                assert!(content.contains("ACCEPTANCE"));
                assert!(
                    content
                        .chars()
                        .any(|c| ('\u{2801}'..='\u{28ff}').contains(&c))
                );
            }
        }
    }
}
