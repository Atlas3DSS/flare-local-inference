//! Terminal loading rows with measured progress and elapsed time.

use std::{
    io::{IsTerminal, Write},
    sync::{Arc, Mutex, mpsc},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

/// Displays one loading stage at a time on stderr; redirected output is plain text.
pub struct StartupProgress {
    enabled: bool,
    terminal: bool,
}

impl StartupProgress {
    pub fn stderr() -> Self {
        Self {
            enabled: true,
            terminal: std::io::stderr().is_terminal(),
        }
    }

    pub fn silent() -> Self {
        Self {
            enabled: false,
            terminal: false,
        }
    }

    pub fn phase(&self, label: &str, total: Option<u64>) -> Phase {
        let state = Arc::new(Mutex::new(State {
            label: label.to_owned(),
            detail: String::new(),
            started: Instant::now(),
            completed: 0,
            total,
        }));
        let mut phase = Phase {
            state: state.clone(),
            enabled: self.enabled,
            terminal: self.terminal,
            finished: false,
            animation: None,
        };
        if self.enabled {
            state.lock().unwrap().render(self.terminal, Status::Running);
        }
        if self.terminal {
            let (stop, receiver) = mpsc::channel();
            let worker = thread::spawn(move || {
                while matches!(
                    receiver.recv_timeout(Duration::from_millis(125)),
                    Err(mpsc::RecvTimeoutError::Timeout)
                ) {
                    state.lock().unwrap().render(true, Status::Running);
                }
            });
            phase.animation = Some((stop, worker));
        }
        phase
    }
}

/// A stage prints FAILED if dropped without an explicit successful finish.
pub struct Phase {
    state: Arc<Mutex<State>>,
    enabled: bool,
    terminal: bool,
    finished: bool,
    animation: Option<(mpsc::Sender<()>, JoinHandle<()>)>,
}

impl Phase {
    pub fn advance(&self, count: u64) {
        let mut state = self.state.lock().unwrap();
        state.completed = state.completed.saturating_add(count);
    }

    pub fn detail(&self, detail: impl Into<String>) {
        self.state.lock().unwrap().detail = detail.into();
    }

    pub fn finish(mut self, detail: impl Into<String>) {
        self.detail(detail);
        self.finished = true;
    }
}

impl Drop for Phase {
    fn drop(&mut self) {
        if let Some((stop, worker)) = self.animation.take() {
            let _ = stop.send(());
            let _ = worker.join();
        }
        if self.enabled {
            self.state.lock().unwrap().render(
                self.terminal,
                if self.finished {
                    Status::Ok
                } else {
                    Status::Failed
                },
            );
        }
    }
}

struct State {
    label: String,
    detail: String,
    started: Instant,
    completed: u64,
    total: Option<u64>,
}

#[derive(Clone, Copy)]
enum Status {
    Running,
    Ok,
    Failed,
}

impl State {
    fn render(&self, terminal: bool, status: Status) {
        let elapsed = self.started.elapsed().as_secs_f64();
        let status_text = match status {
            Status::Ok => "    OK    ".to_owned(),
            Status::Failed => "  FAILED  ".to_owned(),
            Status::Running if terminal => {
                if let Some(total) = self.total {
                    let filled = ((self.completed as f64 / total.max(1) as f64) * 10.0)
                        .clamp(0.0, 10.0) as usize;
                    "━".repeat(filled) + &"─".repeat(10 - filled)
                } else {
                    let tick = (elapsed * 8.0) as usize % 14;
                    let start = if tick <= 7 { tick } else { 14 - tick };
                    "─".repeat(start) + "━━━" + &"─".repeat(7 - start)
                }
            }
            Status::Running => "  LOADING ".to_owned(),
        };
        let label = format!("] {}.", self.label);
        let detail = if self.detail.is_empty() {
            String::new()
        } else {
            format!("  {}", self.detail)
        };
        let timing = format!("  {elapsed:.1}s");
        let width = if terminal {
            terminal_width()
        } else {
            usize::MAX
        };
        let available = width.saturating_sub(11 + timing.chars().count());
        let body = truncate(&(label.clone() + &detail), available);
        let label_end = body
            .char_indices()
            .nth(label.chars().count())
            .map_or(body.len(), |(i, _)| i);
        let (label, detail) = body.split_at(label_end);
        let padding = if terminal {
            available.saturating_sub(body.chars().count())
        } else {
            0
        };
        let mut out = std::io::stderr().lock();
        if terminal {
            let color = match status {
                Status::Running => "36",
                Status::Ok => "1;32",
                Status::Failed => "1;31",
            };
            let _ = write!(
                out,
                "\r\x1b[2K[\x1b[{color}m{status_text}\x1b[0m{label}\x1b[2m{detail}{:padding$}{timing}\x1b[0m",
                ""
            );
        } else {
            let _ = write!(out, "[{status_text}{label}{detail}{timing}");
        }
        if !terminal || !matches!(status, Status::Running) {
            let _ = writeln!(out);
        }
        let _ = out.flush();
    }
}

fn truncate(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        return text.to_owned();
    }
    if width == 0 {
        return String::new();
    }
    text.chars().take(width - 1).collect::<String>() + "…"
}

fn terminal_width() -> usize {
    let mut size: libc::winsize = unsafe { std::mem::zeroed() };
    // SAFETY: ioctl writes the terminal dimensions into a valid winsize.
    if unsafe { libc::ioctl(libc::STDERR_FILENO, libc::TIOCGWINSZ, &mut size) } == 0
        && size.ws_col > 0
    {
        usize::from(size.ws_col)
    } else {
        80
    }
}
