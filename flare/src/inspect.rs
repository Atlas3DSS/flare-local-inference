//! `flare inspect`: a package's signed metadata as a dossier on the terminal, or as JSON for
//! other programs. The index is verified against the publisher's key and read as it is; nothing
//! is authorized, decrypted, or loaded onto a GPU.
//!
//! The dossier is drawn the way a bulletin board drew its login screen: a block wordmark under
//! a shaded strip, double-ruled panels with the title set into the top rule, dotted leaders
//! between a key and its value, and shaded gauges for the values that have a natural range.
//! Colors are the sun palette of the serving banner on a cyan frame. A redirected stream, a
//! dumb terminal, or `--plain` gets the same characters without escape sequences.

use std::{fmt::Write as _, io::IsTerminal};

use flare_helios::runtime_package::{PackageClaim, VerifiedPackage};
use serde_json::Value;

use crate::cli::Inspect;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn run(args: Inspect) -> Result<()> {
    let root = args
        .package
        .canonicalize()
        .map_err(|e| format!("cannot access package {}: {e}", args.package.display()))?;
    let claim = PackageClaim::read(&root)?;
    let mut profiles = crate::trust::publishers(
        &args.publisher_keys,
        claim.development,
        crate::environment::home_directory,
    )?;
    if !profiles.iter().any(|p| p.fingerprint() == claim.signer) {
        if let Some(profile) =
            crate::trust::remembered_profile(&claim, args.trust_store.as_deref())?
        {
            profiles.push(profile);
        }
    }
    let package = VerifiedPackage::open(&root, &profiles, args.rank).map_err(
        |error| -> Box<dyn std::error::Error> {
            if error.to_string().contains("unknown package publisher") {
                return format!(
                    "cannot verify signing key {}: no embedded, remembered, or --publisher-keys \
                     profile names it (publisher {} at {})",
                    claim.signer, claim.namespace, claim.official_source
                )
                .into();
            }
            error.into()
        },
    )?;
    let mut value = package.describe()?;
    value["package"]["path"] = Value::String(root.display().to_string());
    if args.json {
        println!("{}", serde_json::to_string_pretty(&value)?);
        return Ok(());
    }
    let stdout = std::io::stdout();
    let color = !args.plain
        && stdout.is_terminal()
        && !std::env::var("TERM").is_ok_and(|t| t == "dumb");
    let width = if stdout.is_terminal() {
        terminal_width().clamp(MIN_WIDTH, MAX_WIDTH)
    } else {
        DEFAULT_WIDTH
    };
    print!("{}", render(&value, width, color));
    Ok(())
}

const MIN_WIDTH: usize = 64;
const MAX_WIDTH: usize = 100;
const DEFAULT_WIDTH: usize = 80;
/// Columns for a key and its dotted leader inside a panel.
const KEY_WIDTH: usize = 26;
const GAUGE_CELLS: usize = 20;

const BORDER: u32 = 0x2fc4e6;
const TAG: u32 = 0xff5fd7;
const KEY: u32 = 0xffd27b;
const VALUE: u32 = 0xfff8e7;
const NUMBER: u32 = 0xffb957;
const DIM: u32 = 0x6b7b90;
const EMPTY: u32 = 0x3a4656;
/// The sun palette, limb to edge, for the wordmark rows and the shaded strips.
const SUN: [u32; 6] = [0xffe8b0, 0xffd27b, 0xffb957, 0xff9c42, 0xff7033, 0xe54b32];
const GAUGE: [u32; 4] = [0xe54b32, 0xff9c42, 0xffd27b, 0xffe8b0];

/// The wordmark's letters in the serving banner's block face, joined with one column between.
const LETTERS: [[&str; 6]; 7] = [
    ["██╗", "██║", "██║", "██║", "██║", "╚═╝"],
    ["███╗   ██╗", "████╗  ██║", "██╔██╗ ██║", "██║╚██╗██║", "██║ ╚████║", "╚═╝  ╚═══╝"],
    ["███████╗", "██╔════╝", "███████╗", "╚════██║", "███████║", "╚══════╝"],
    ["██████╗ ", "██╔══██╗", "██████╔╝", "██╔═══╝ ", "██║     ", "╚═╝     "],
    ["███████╗", "██╔════╝", "█████╗  ", "██╔══╝  ", "███████╗", "╚══════╝"],
    [" ██████╗", "██╔════╝", "██║     ", "██║     ", "╚██████╗", " ╚═════╝"],
    ["████████╗", "╚══██╔══╝", "   ██║   ", "   ██║   ", "   ██║   ", "   ╚═╝   "],
];

fn wordmark() -> [String; 6] {
    std::array::from_fn(|row| {
        LETTERS
            .iter()
            .map(|letter| letter[row])
            .collect::<Vec<_>>()
            .join(" ")
    })
}

fn terminal_width() -> usize {
    #[cfg(unix)]
    {
        let mut size: libc::winsize = unsafe { std::mem::zeroed() };
        // SAFETY: TIOCGWINSZ fills a winsize for a terminal descriptor; failure leaves zeros.
        if unsafe { libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, &mut size) } == 0
            && size.ws_col > 0
        {
            return size.ws_col as usize;
        }
    }
    DEFAULT_WIDTH
}

fn visible(text: &str) -> usize {
    text.chars().count()
}

fn thousands(value: u64) -> String {
    let digits = value.to_string();
    let mut output = String::new();
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index) % 3 == 0 {
            output.push(',');
        }
        output.push(digit);
    }
    output
}

fn bytes(value: u64) -> String {
    const KIB: f64 = 1024.0;
    let v = value as f64;
    if v >= KIB * KIB * KIB {
        format!("{:.2} GiB", v / (KIB * KIB * KIB))
    } else if v >= KIB * KIB {
        format!("{:.1} MiB", v / (KIB * KIB))
    } else if v >= KIB {
        format!("{:.0} KiB", v / KIB)
    } else {
        format!("{value} B")
    }
}

fn duration(nanoseconds: u64) -> String {
    if nanoseconds == 0 {
        "–".into()
    } else if nanoseconds >= 1_000_000 {
        format!("{:.2} ms", nanoseconds as f64 / 1e6)
    } else {
        format!("{:.0} µs", nanoseconds as f64 / 1e3)
    }
}

fn decimal(value: f64) -> String {
    let text = format!("{value:.4}");
    let text = text.trim_end_matches('0').trim_end_matches('.');
    if text.is_empty() { "0".into() } else { text.into() }
}

/// A value as text, with control characters shown as escapes so a token such as a marker
/// ending in a newline stays on its row.
fn text(value: &Value) -> String {
    let raw = match value {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    };
    let mut output = String::new();
    for c in raw.chars() {
        match c {
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            c if c.is_control() => {
                let _ = write!(output, "\\u{{{:x}}}", c as u32);
            }
            c => output.push(c),
        }
    }
    output
}

fn u64_of(value: &Value) -> u64 {
    value.as_u64().unwrap_or(0)
}

fn f64_of(value: &Value) -> f64 {
    value.as_f64().unwrap_or(0.0)
}

fn list(value: &Value) -> String {
    value
        .as_array()
        .map(|items| items.iter().map(text).collect::<Vec<_>>().join(", "))
        .unwrap_or_default()
}

struct Canvas {
    out: String,
    width: usize,
    color: bool,
}

impl Canvas {
    fn paint(&mut self, color: u32, bold: bool, text: &str) {
        if self.color {
            let _ = write!(
                self.out,
                "\x1b[{};38;2;{};{};{}m{text}\x1b[0m",
                if bold { 1 } else { 22 },
                color >> 16,
                (color >> 8) & 255,
                color & 255
            );
        } else {
            self.out.push_str(text);
        }
    }

    fn end(&mut self) {
        self.out.push('\n');
    }

    fn inner(&self) -> usize {
        self.width - 4
    }

    /// A shaded strip across the page in the sun gradient, mirrored so it is bright at the
    /// center.
    fn strip(&mut self, glyph: char) {
        let width = self.width;
        for column in 0..width {
            let position = (column as f64 / (width - 1).max(1) as f64 - 0.5).abs() * 2.0;
            let index = ((position * (SUN.len() - 1) as f64).round() as usize).min(SUN.len() - 1);
            self.paint(SUN[index], false, &glyph.to_string());
        }
        self.end();
    }

    fn centered(&mut self, color: u32, bold: bool, text: &str) {
        let pad = self.width.saturating_sub(visible(text)) / 2;
        self.out.push_str(&" ".repeat(pad));
        self.paint(color, bold, text);
        self.end();
    }

    fn header(&mut self, model_id: &str, signer: &str) {
        self.strip('▄');
        self.end();
        let rows = wordmark();
        for (index, row) in rows.iter().enumerate() {
            let row: String = row.chars().take(self.width).collect();
            self.centered(SUN[index], true, &row);
        }
        self.end();
        self.strip('▀');
        self.end();
        let mut title = String::new();
        let room = self.width.saturating_sub(12);
        let model_id: String = model_id.chars().take(room).collect();
        let _ = write!(title, "░▒▓█  {model_id}  █▓▒░");
        let pad = self.width.saturating_sub(visible(&title)) / 2;
        self.out.push_str(&" ".repeat(pad));
        self.paint(SUN[5], false, "░");
        self.paint(SUN[3], false, "▒");
        self.paint(SUN[1], false, "▓");
        self.paint(SUN[0], true, "█");
        self.paint(VALUE, true, &format!("  {model_id}  "));
        self.paint(SUN[0], true, "█");
        self.paint(SUN[1], false, "▓");
        self.paint(SUN[3], false, "▒");
        self.paint(SUN[5], false, "░");
        self.end();
        for line in Self::wrap(signer, self.width.saturating_sub(2)) {
            self.centered(DIM, false, &line);
        }
        self.end();
    }

    fn open(&mut self, title: &str) {
        self.paint(BORDER, false, "╔═[ ");
        self.paint(SUN[5], false, "▓");
        self.paint(SUN[3], false, "▒");
        self.paint(SUN[1], false, "░");
        self.paint(TAG, true, &format!(" {title} "));
        self.paint(SUN[1], false, "░");
        self.paint(SUN[3], false, "▒");
        self.paint(SUN[5], false, "▓");
        self.paint(BORDER, false, " ]");
        let used = 4 + 3 + visible(title) + 2 + 3 + 2;
        let rule = "═".repeat(self.width.saturating_sub(used + 1));
        self.paint(BORDER, false, &rule);
        self.paint(BORDER, false, "╗");
        self.end();
    }

    fn close(&mut self) {
        let rule = "═".repeat(self.width - 2);
        self.paint(BORDER, false, &format!("╚{rule}╝"));
        self.end();
        self.end();
    }

    fn left(&mut self) {
        self.paint(BORDER, false, "║ ");
    }

    fn right(&mut self, used: usize) {
        let pad = self.inner().saturating_sub(used);
        self.out.push_str(&" ".repeat(pad));
        self.paint(BORDER, false, " ║");
        self.end();
    }

    /// Lines of at most `width` characters, broken at spaces where there are any.
    fn wrap(value: &str, width: usize) -> Vec<String> {
        let width = width.max(8);
        let mut lines = Vec::new();
        let mut current = String::new();
        for word in value.split(' ') {
            let mut word = word.to_string();
            while visible(&word) > width {
                if !current.is_empty() {
                    lines.push(std::mem::take(&mut current));
                }
                let head: String = word.chars().take(width).collect();
                word = word.chars().skip(width).collect();
                lines.push(head);
            }
            let joined = if current.is_empty() {
                word.clone()
            } else {
                format!("{current} {word}")
            };
            if visible(&joined) > width {
                lines.push(std::mem::take(&mut current));
                current = word;
            } else {
                current = joined;
            }
        }
        if !current.is_empty() || lines.is_empty() {
            lines.push(current);
        }
        lines
    }

    fn key(&mut self, key: &str) -> usize {
        let key: String = key.chars().take(KEY_WIDTH - 2).collect();
        self.paint(KEY, false, &key);
        let leaders = KEY_WIDTH.saturating_sub(visible(&key));
        self.paint(DIM, false, &format!(" {}", "·".repeat(leaders.saturating_sub(1))));
        KEY_WIDTH
    }

    fn row(&mut self, key: &str, value: &str) {
        self.row_colored(key, value, VALUE);
    }

    fn number(&mut self, key: &str, value: &str) {
        self.row_colored(key, value, NUMBER);
    }

    fn row_colored(&mut self, key: &str, value: &str, color: u32) {
        let room = self.inner().saturating_sub(KEY_WIDTH + 2);
        let lines = Self::wrap(value, room);
        for (index, line) in lines.iter().enumerate() {
            self.left();
            let used = if index == 0 {
                self.key(key)
            } else {
                self.out.push_str(&" ".repeat(KEY_WIDTH));
                KEY_WIDTH
            };
            self.out.push(' ');
            self.paint(color, false, line);
            self.right(used + 1 + visible(line));
        }
    }

    /// A shaded gauge of `fraction` of its range, labelled with the value.
    fn gauge(&mut self, key: &str, fraction: f64, label: &str) {
        self.left();
        let mut used = self.key(key);
        self.out.push(' ');
        self.paint(DIM, false, "[");
        let filled = ((fraction.clamp(0.0, 1.0) * GAUGE_CELLS as f64).round() as usize).min(GAUGE_CELLS);
        for cell in 0..GAUGE_CELLS {
            if cell < filled {
                let index = cell * GAUGE.len() / GAUGE_CELLS;
                self.paint(GAUGE[index], true, "█");
            } else {
                self.paint(EMPTY, false, "░");
            }
        }
        self.paint(DIM, false, "] ");
        used += 1 + 1 + GAUGE_CELLS + 2;
        let room = self.inner().saturating_sub(used);
        let label: String = label.chars().take(room).collect();
        self.paint(NUMBER, false, &label);
        self.right(used + visible(&label));
    }

    fn note(&mut self, text: &str) {
        let room = self.inner().saturating_sub(2);
        for line in Self::wrap(text, room) {
            self.left();
            self.out.push_str("  ");
            self.paint(DIM, false, &line);
            self.right(2 + visible(&line));
        }
    }

    /// Columns sized to their widest cell; the first column gives way when the panel is narrow.
    fn table(&mut self, head: &[&str], rows: &[Vec<String>]) {
        let columns = head.len();
        let mut widths: Vec<usize> = head.iter().map(|h| visible(h)).collect();
        for row in rows {
            for (index, cell) in row.iter().enumerate().take(columns) {
                widths[index] = widths[index].max(visible(cell));
            }
        }
        let room = self.inner().saturating_sub(2);
        let total = |widths: &[usize]| widths.iter().sum::<usize>() + 2 * (columns - 1);
        while total(&widths) > room {
            let Some((widest, _)) = widths.iter().enumerate().filter(|(_, w)| **w > 4).max_by_key(|(index, w)| (**w, usize::MAX - index)) else {
                break;
            };
            widths[widest] -= 1;
        }
        let draw = |canvas: &mut Self, cells: &[String], color: u32, bold: bool| {
            canvas.left();
            canvas.out.push_str("  ");
            let mut used = 2;
            for (index, width) in widths.iter().enumerate() {
                let cell = cells.get(index).map_or("", String::as_str);
                let cell: String = if visible(cell) > *width {
                    let mut c: String = cell.chars().take(width.saturating_sub(1)).collect();
                    c.push('…');
                    c
                } else {
                    cell.to_string()
                };
                let numeric = index > 0 && cell.chars().next().is_some_and(|c| c.is_ascii_digit());
                let pad = width - visible(&cell);
                if numeric {
                    canvas.out.push_str(&" ".repeat(pad));
                    canvas.paint(if bold { color } else { NUMBER }, bold, &cell);
                } else {
                    canvas.paint(color, bold, &cell);
                    canvas.out.push_str(&" ".repeat(pad));
                }
                used += width;
                if index + 1 < widths.len() {
                    canvas.out.push_str("  ");
                    used += 2;
                }
            }
            canvas.right(used);
        };
        let head: Vec<String> = head.iter().map(|h| h.to_string()).collect();
        draw(self, &head, TAG, true);
        self.left();
        self.out.push_str("  ");
        let rule = "─".repeat(total(&widths).min(room));
        self.paint(DIM, false, &rule);
        self.right(2 + visible(&rule));
        for row in rows {
            draw(self, row, VALUE, false);
        }
    }

    fn footer(&mut self, path: &str) {
        let text = format!(" flare {} · {path} ", env!("CARGO_PKG_VERSION"));
        let text: String = text.chars().take(self.width.saturating_sub(6)).collect();
        let rule = self.width.saturating_sub(visible(&text));
        let left = rule / 2;
        self.paint(DIM, false, &"─".repeat(left));
        self.paint(DIM, false, &text);
        self.paint(DIM, false, &"─".repeat(rule - left));
        self.end();
    }
}

/// The dossier for a `describe` document, `width` columns wide.
pub fn render(value: &Value, width: usize, color: bool) -> String {
    let mut canvas = Canvas {
        out: String::new(),
        width: width.max(MIN_WIDTH),
        color,
    };
    let package = &value["package"];
    let model = &value["model"];
    let cache = &value["cache"];
    let speculation = &value["speculation"];
    let sampling = &value["sampling_defaults"];
    let frontend = &value["frontend"];
    let vision = &value["vision"];
    let engram = &value["engram"];
    let ranks = value["ranks"].as_array().cloned().unwrap_or_default();
    let publisher = &package["publisher"];

    let key_kind = if publisher["development"].as_bool().unwrap_or(false) {
        "development key"
    } else {
        "published key"
    };
    let hosts = match package["cluster"].as_object() {
        Some(cluster) => format!("{} hosts", u64_of(&cluster["world"])),
        None => match ranks.len() {
            1 => "1 rank".to_string(),
            n => format!("{n} ranks"),
        },
    };
    canvas.header(
        &text(&model["model_id"]),
        &format!(
            "signed by {} · {} · {key_kind} · {hosts}",
            text(&publisher["name"]),
            text(&publisher["namespace"])
        ),
    );

    // PACKAGE
    canvas.open("PACKAGE");
    canvas.row("file", &text(&package["path"]));
    let companions = package["companions"].as_array().cloned().unwrap_or_default();
    let companion_bytes: u64 = companions.iter().map(|c| u64_of(&c["bytes"])).sum();
    canvas.number(
        "size",
        &format!(
            "{} primary{}",
            bytes(u64_of(&package["bytes"])),
            if companions.is_empty() {
                String::new()
            } else {
                format!(
                    " + {} in {} companion file{}",
                    bytes(companion_bytes),
                    companions.len(),
                    if companions.len() == 1 { "" } else { "s" }
                )
            }
        ),
    );
    canvas.number("format version", &u64_of(&package["format_version"]).to_string());
    match package["cluster"].as_object() {
        Some(cluster) => {
            canvas.row(
                "ranks",
                &format!(
                    "{} on separate hosts, {} verified in this copy",
                    u64_of(&cluster["world"]),
                    ranks.len()
                ),
            );
            canvas.row("export", &text(&cluster["export_id"]));
        }
        None => canvas.row(
            "ranks",
            &format!("{} on one host (tensor parallel)", ranks.len()),
        ),
    }
    canvas.row(
        "publisher",
        &format!("{} ({})", text(&publisher["name"]), text(&publisher["namespace"])),
    );
    canvas.row("signing key", &format!("{} · {key_kind}", text(&publisher["fingerprint"])));
    canvas.row("source", &text(&package["official_source"]));
    match package["license"].as_object() {
        Some(license) if !text(&license["link"]).is_empty() => canvas.row(
            "license",
            &format!("{} · {}", text(&license["name"]), text(&license["link"])),
        ),
        Some(license) => canvas.row("license", &text(&license["name"])),
        None => canvas.row("license", "not recorded"),
    }
    canvas.close();

    // MODEL
    canvas.open("MODEL");
    canvas.row("model id", &text(&model["model_id"]));
    canvas.row("identity", &text(&model["identity"]));
    let served = list(&model["served_names"]);
    canvas.row("served names", if served.is_empty() { "model id only" } else { &served });
    let vocabulary = u64_of(&model["vocabulary"]);
    let input_vocabulary = u64_of(&model["input_vocabulary"]);
    canvas.number(
        "vocabulary",
        &if input_vocabulary != 0 && input_vocabulary != vocabulary {
            format!("{} output · {} input", thousands(vocabulary), thousands(input_vocabulary))
        } else {
            thousands(vocabulary)
        },
    );
    if let Some(first) = ranks.first() {
        let arch = first["architecture"].as_array().cloned().unwrap_or_default();
        if arch.len() == 2 {
            canvas.row(
                "gpu architecture",
                &format!(
                    "sm_{}{} (compute capability {}.{})",
                    u64_of(&arch[0]),
                    u64_of(&arch[1]),
                    u64_of(&arch[0]),
                    u64_of(&arch[1])
                ),
            );
        }
    }
    let decode_weights = u64_of(&model["decode_weight_bytes"]);
    canvas.number(
        "weights read per step",
        &if decode_weights == 0 {
            "not declared".into()
        } else {
            format!("{} per decode step", bytes(decode_weights))
        },
    );
    canvas.close();

    // LIMITS
    canvas.open("LIMITS");
    let context = u64_of(&model["max_sequence_tokens"]);
    let max_active = u64_of(&model["max_active"]);
    canvas.number("context", &format!("{} tokens", thousands(context)));
    canvas.number("prefill chunk", &format!("{} tokens", thousands(u64_of(&model["prefill_chunk_size"]))));
    canvas.number("concurrency", &format!("{} active requests", thousands(max_active)));
    canvas.number(
        "default max tokens",
        &match model["max_new_tokens"].as_u64() {
            Some(n) => format!("{} output tokens", thousands(n)),
            None => "the remaining context".into(),
        },
    );
    if let Some(n) = model["max_body_bytes"].as_u64() {
        canvas.number("request body limit", &bytes(n));
    }
    if let Some(n) = model["queue_capacity"].as_u64() {
        canvas.number("admission queue", &format!("{n} requests"));
    }
    if let Some(n) = model["frontend_workers"].as_u64() {
        canvas.number("frontend workers", &n.to_string());
    }
    if let Some(n) = model["reserve_bytes"].as_u64() {
        canvas.number("gpu reserve", &format!("{} beyond allocations", bytes(n)));
    }
    for (key, name) in [("context_hint", "context hint"), ("capacity_hint", "capacity hint")] {
        let hint = text(&model[key]);
        if !hint.is_empty() {
            canvas.row(name, &hint);
        }
    }
    canvas.close();

    // CACHE
    canvas.open("CACHE");
    let page_tokens = u64_of(&cache["page_tokens"]);
    let exported_pages = u64_of(&cache["pages"]);
    let mut kv_page_bytes = 0u64;
    let mut state_row_bytes = 0u64;
    let mut max_pages: Option<u64> = None;
    for rank in &ranks {
        for allocation in rank["capacity_allocations"].as_array().into_iter().flatten() {
            match text(&allocation["capacity"]).as_str() {
                "KV_PAGES" => {
                    kv_page_bytes += u64_of(&allocation["unit_bytes"]);
                    let maximum = u64_of(&allocation["maximum"]);
                    max_pages = Some(max_pages.map_or(maximum, |m| m.min(maximum)));
                }
                "STATE_ROWS" => state_row_bytes += u64_of(&allocation["unit_bytes"]),
                _ => {}
            }
        }
    }
    let pages = max_pages.unwrap_or(exported_pages);
    let extent = pages.saturating_mul(page_tokens);
    canvas.number("page", &format!("{} tokens", thousands(page_tokens)));
    canvas.number(
        "page extent",
        &format!(
            "{} at most · {} backed at export · serving sizes pages from free memory",
            thousands(pages),
            thousands(exported_pages)
        ),
    );
    canvas.number(
        "kv extent",
        &format!(
            "{} tokens{}",
            thousands(extent),
            if context > 0 {
                format!(" · {:.2} contexts", extent as f64 / context as f64)
            } else {
                String::new()
            }
        ),
    );
    if context > 0 && max_active > 0 {
        let full = (context * max_active) as f64;
        canvas.gauge(
            "full contexts",
            extent as f64 / full,
            &format!("{:.2} of {} concurrent", extent as f64 / context as f64, max_active),
        );
    }
    if kv_page_bytes > 0 {
        canvas.number(
            "kv bytes",
            &format!(
                "{} per page · {} per token · all ranks",
                bytes(kv_page_bytes),
                bytes(kv_page_bytes / page_tokens.max(1))
            ),
        );
    }
    canvas.number("state rows", &thousands(u64_of(&cache["state_rows"])));
    if state_row_bytes > 0 {
        canvas.number("state row bytes", &format!("{} · all ranks", bytes(state_row_bytes)));
    }
    canvas.number("snapshots", &thousands(u64_of(&cache["snapshots"])));
    // Identical regions, one per layer, are counted rather than listed.
    let mut regions: Vec<(Vec<String>, usize)> = Vec::new();
    for r in cache["regions"].as_array().into_iter().flatten() {
        let row = vec![
            text(&r["kind"]).to_lowercase().replace('_', " "),
            thousands(u64_of(&r["tokens_per_record"])),
            match u64_of(&r["window_tokens"]) {
                0 => "–".into(),
                w => thousands(w),
            },
            bytes(u64_of(&r["stride_bytes"])),
        ];
        match regions.iter_mut().find(|(known, _)| *known == row) {
            Some((_, count)) => *count += 1,
            None => regions.push((row, 1)),
        }
    }
    if !regions.is_empty() {
        let rows: Vec<Vec<String>> = regions
            .into_iter()
            .map(|(mut row, count)| {
                row.push(count.to_string());
                row
            })
            .collect();
        canvas.table(&["history region", "tokens/record", "window", "record stride", "count"], &rows);
    }
    canvas.close();

    // SPECULATION
    canvas.open("SPECULATION");
    let widths = list(&speculation["widths"]);
    if widths.is_empty() && u64_of(&speculation["token_capacity"]) == 0 {
        canvas.note("no speculative graphs prepared; every step decodes one token per request");
    } else {
        canvas.row(
            "proposal source",
            match text(&speculation["proposal_source"]).as_str() {
                "mtp" => "multi-token prediction head",
                "copy" => "copy drafts from the prompt",
                other => other,
            },
        );
        canvas.number("verification widths", &widths);
        canvas.number(
            "fixed width",
            &match speculation["fixed_width"].as_u64() {
                Some(w) => format!("{w} (adaptive verification off)"),
                None => "none (adaptive verification)".into(),
            },
        );
        canvas.number(
            "token capacity",
            &format!("{} target rows per speculative step", thousands(u64_of(&speculation["token_capacity"]))),
        );
        canvas.row(
            "history drafts",
            if speculation["history_drafts"].as_bool().unwrap_or(false) { "yes" } else { "no" },
        );
    }
    canvas.close();

    // SAMPLING DEFAULTS
    canvas.open("SAMPLING DEFAULTS");
    canvas.number("seed", &u64_of(&sampling["seed"]).to_string());
    let temperature = f64_of(&sampling["temperature"]);
    canvas.gauge("temperature", temperature / 2.0, &format!("{} (0 – 2)", decimal(temperature)));
    let top_p = f64_of(&sampling["top_p"]);
    canvas.gauge("top p", top_p, &decimal(top_p));
    let min_p = f64_of(&sampling["min_p"]);
    canvas.gauge("min p", min_p, &format!("{}{}", decimal(min_p), if min_p == 0.0 { " (off)" } else { "" }));
    let top_k = sampling["top_k"].as_i64().unwrap_or(0);
    canvas.number("top k", &if top_k <= 0 { "off".into() } else { top_k.to_string() });
    let repetition = f64_of(&sampling["repetition_penalty"]);
    canvas.gauge(
        "repetition penalty",
        repetition / 2.0,
        &format!("{}{}", decimal(repetition), if repetition == 1.0 { " (neutral)" } else { "" }),
    );
    for (key, name) in [("presence_penalty", "presence penalty"), ("frequency_penalty", "frequency penalty")] {
        let value = f64_of(&sampling[key]);
        canvas.gauge(
            name,
            (value + 2.0) / 4.0,
            &format!("{}{}", decimal(value), if value == 0.0 { " (neutral)" } else { "" }),
        );
    }
    canvas.close();

    // FRONTEND
    canvas.open("FRONTEND");
    canvas.row(
        "prompt encoder",
        match text(&frontend["prompt_encoder"]).as_str() {
            "PROMPT_JINJA" => "Jinja chat template",
            "DEEPSEEK_V4" => "DeepSeek V4 native encoder",
            "DEEPSEEK_V41" => "DeepSeek V4.1 native encoder",
            other => other,
        },
    );
    canvas.row(
        "template encoding",
        match text(&frontend["template_encoding"]).as_str() {
            "TEMPLATE_TEXT" => "text",
            "TEMPLATE_TOKEN_SEGMENTS" => "token segments",
            other => other,
        },
    );
    let tokenizer = &frontend["tokenizer"];
    canvas.number(
        "eos tokens",
        &format!(
            "{}{}",
            list(&frontend["eos_token_ids"]),
            match text(&tokenizer["eos_token"]).as_str() {
                "" => String::new(),
                token => format!(" · {token}"),
            }
        ),
    );
    for (key, name) in [("bos_token", "bos token"), ("pad_token", "pad token")] {
        let token = text(&tokenizer[key]);
        if !token.is_empty() {
            canvas.row(name, &token);
        }
    }
    canvas.number("special tokens", &format!("{} additional", u64_of(&tokenizer["additional_special_tokens"])));
    match frontend["tools"].as_object() {
        Some(tools) => canvas.row(
            "tool calls",
            &format!(
                "{} parser{}",
                text(&tools["kind"]).to_lowercase().replace('_', " "),
                match u64_of(&tools["wrapping_newlines"]) {
                    0 => String::new(),
                    n => format!(" · {n} wrapping newlines"),
                }
            ),
        ),
        None => canvas.row("tool calls", "no parser"),
    }
    match frontend["reasoning"].as_object() {
        Some(reasoning) => canvas.row(
            "reasoning",
            &format!(
                "{} parser · {} … {}",
                text(&reasoning["kind"]).to_lowercase().replace('_', " "),
                text(&reasoning["open"]),
                text(&reasoning["close"])
            ),
        ),
        None => canvas.row("reasoning", "no parser"),
    }
    let marker = text(&frontend["assistant_marker"]);
    if !marker.is_empty() {
        canvas.row("assistant marker", &marker);
    }
    let ends = list(&frontend["message_end_markers"]);
    if !ends.is_empty() {
        canvas.row("message end markers", &ends);
    }
    if let Some(boundary) = frontend["prefix_boundary"].as_object() {
        let roles = list(&boundary["roles"]);
        if !roles.is_empty() {
            canvas.row(
                "prefix boundary",
                &format!(
                    "{roles}{}",
                    if boundary["continue_final_message"].as_bool().unwrap_or(false) {
                        " · continues the final message"
                    } else {
                        ""
                    }
                ),
            );
        }
    }
    let assets: Vec<Vec<String>> = value["assets"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|a| {
            vec![
                text(&a["name"]),
                bytes(u64_of(&a["bytes"])),
                text(&a["sha256"]).chars().take(16).collect(),
            ]
        })
        .collect();
    if !assets.is_empty() {
        canvas.table(&["asset", "size", "sha256"], &assets);
    }
    canvas.close();

    // VISION
    if let Some(vision) = vision.as_object() {
        canvas.open("VISION");
        let image = &vision["image"];
        canvas.row("image profile", &text(&image["profile"]));
        canvas.number(
            "image tokens per prompt",
            &format!("{} shared by every image", thousands(u64_of(&vision["max_image_tokens_per_prompt"]))),
        );
        canvas.number("images per prompt", &thousands(u64_of(&vision["max_images"])));
        canvas.number(
            "patches per token",
            &u64_of(&vision["patches_per_feature"]).to_string(),
        );
        let capacities = vision["patch_capacities"]
            .as_array()
            .map(|c| c.iter().map(|v| thousands(u64_of(v))).collect::<Vec<_>>().join(", "))
            .unwrap_or_default();
        canvas.number("vision graph patches", &capacities);
        canvas.number("feature slots", &u64_of(&vision["feature_slots"]).to_string());
        for (key, name) in [("min_pixels", "min pixels"), ("max_pixels", "max pixels"), ("max_tokens", "max tokens")] {
            if let Some(n) = image[key].as_u64() {
                canvas.number(name, &thousands(n));
            }
        }
        if let Some(token) = frontend["image_placeholder_token_id"].as_u64() {
            canvas.number("placeholder token", &token.to_string());
        }
        canvas.close();
    }

    // ENGRAM
    if let Some(engram) = engram.as_object() {
        canvas.open("ENGRAM");
        canvas.number("compressed vocabulary", &thousands(u64_of(&engram["compressed_vocabulary"])));
        canvas.number("token map", &format!("{} entries", thousands(u64_of(&engram["token_map_entries"]))));
        let tables: Vec<Vec<String>> = engram["tables"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|t| {
                vec![
                    u64_of(&t["layer"]).to_string(),
                    thousands(u64_of(&t["rows"])),
                    u64_of(&t["embedding_width"]).to_string(),
                    text(&t["encoding"]).to_lowercase(),
                    bytes(u64_of(&t["weight_bytes"])),
                    bytes(u64_of(&t["scale_bytes"])),
                    u64_of(&t["shards"]).to_string(),
                ]
            })
            .collect();
        canvas.table(&["layer", "rows", "width", "encoding", "weights", "scales", "shards"], &tables);
        canvas.close();
    }

    // RANKS
    canvas.open("RANKS");
    let rows: Vec<Vec<String>> = ranks
        .iter()
        .map(|r| {
            let arch = r["architecture"].as_array().cloned().unwrap_or_default();
            vec![
                u64_of(&r["rank"]).to_string(),
                if arch.len() == 2 {
                    format!("sm_{}{}", u64_of(&arch[0]), u64_of(&arch[1]))
                } else {
                    "?".into()
                },
                bytes(u64_of(&r["device_bytes"])),
                bytes(u64_of(&r["mapped_host_bytes"])),
                bytes(u64_of(&r["peer_bytes"])),
                thousands(u64_of(&r["graphs"])),
                thousands(u64_of(&r["modules"])),
                thousands(u64_of(&r["functions"])),
                match r["roce"].as_object() {
                    Some(roce) => format!("{} ranks", u64_of(&roce["world"])),
                    None => "–".into(),
                },
            ]
        })
        .collect();
    canvas.table(
        &["rank", "arch", "device", "host", "peer", "graphs", "modules", "kernels", "roce"],
        &rows,
    );
    canvas.close();

    // GRAPHS
    if let Some(first) = ranks.first() {
        let graphs = first["execution_graphs"].as_array().cloned().unwrap_or_default();
        if !graphs.is_empty() {
            canvas.open(&format!("GRAPHS · RANK {}", u64_of(&first["rank"])));
            let rows: Vec<Vec<String>> = graphs
                .iter()
                .map(|g| {
                    vec![
                        text(&g["name"]),
                        text(&g["kind"]),
                        thousands(u64_of(&g["batch_capacity"])),
                        thousands(u64_of(&g["token_capacity"])),
                        thousands(u64_of(&g["packed_tokens"])),
                        match g["draft_tokens"].as_u64() {
                            Some(n) => n.to_string(),
                            None => "–".into(),
                        },
                        match g["verify_tokens"].as_u64() {
                            Some(n) => n.to_string(),
                            None => "–".into(),
                        },
                        duration(u64_of(&g["estimated_duration_ns"])),
                    ]
                })
                .collect();
            canvas.table(
                &["graph", "kind", "batch", "tokens", "packed", "draft", "verify", "estimate"],
                &rows,
            );
            canvas.close();
        }
    }

    canvas.footer(&text(&package["path"]));
    canvas.out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn document() -> Value {
        json!({
            "package": {
                "path": "/packages/model.safetensors", "format_version": 5, "bytes": 700_000_000u64,
                "companions": [{"name": "model-data-01-of-02.safetensors", "bytes": 60_000_000_000u64},
                               {"name": "model-data-02-of-02.safetensors", "bytes": 50_000_000_000u64}],
                "ranks": 1, "cluster": null,
                "publisher": {"name": "Local development signer", "namespace": "local-development",
                              "fingerprint": "ab".repeat(32), "development": true},
                "official_source": "https://huggingface.co/org/model",
                "license": {"name": "MIT", "link": "https://example.com/LICENSE"},
            },
            "model": {
                "model_id": "org/Model-Flash", "identity": "cd".repeat(32), "served_names": ["Ava"],
                "vocabulary": 151936, "input_vocabulary": 151936, "max_sequence_tokens": 1048576,
                "prefill_chunk_size": 4096, "max_active": 32, "max_new_tokens": 32768,
                "decode_weight_bytes": 0, "reserve_bytes": 2147483648u64, "queue_capacity": 16,
                "frontend_workers": 2, "max_body_bytes": 268435456, "context_hint": "", "capacity_hint": "",
            },
            "cache": {"page_tokens": 64, "pages": 64, "state_rows": 64, "snapshots": 32,
                      "regions": [{"kind": "PAGED_LATENT", "tokens_per_record": 64, "window_tokens": 0,
                                   "stride_bytes": 36864, "nbytes": 1},
                                  {"kind": "PAGED_LATENT", "tokens_per_record": 64, "window_tokens": 0,
                                   "stride_bytes": 36864, "nbytes": 1},
                                  {"kind": "SLIDING_WINDOW", "tokens_per_record": 1, "window_tokens": 4096,
                                   "stride_bytes": 512, "nbytes": 1}]},
            "speculation": {"proposal_source": "mtp", "widths": [5, 3], "fixed_width": null,
                            "token_capacity": 192, "history_drafts": false},
            "sampling_defaults": {"seed": 0, "temperature": 1.0, "top_k": 20, "top_p": 0.95,
                                  "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 0.0,
                                  "frequency_penalty": 0.0},
            "frontend": {"prompt_encoder": "DEEPSEEK_V41", "template_encoding": "TEMPLATE_TEXT",
                         "eos_token_ids": [1], "image_placeholder_token_id": 128815,
                         "tokenizer": {"eos_token": "<｜end▁of▁sentence｜>", "bos_token": "<｜begin▁of▁sentence｜>",
                                       "pad_token": null, "unk_token": null, "additional_special_tokens": 12},
                         "tools": {"kind": "DSML", "wrapping_newlines": 1, "tag_separator": ""},
                         "reasoning": {"kind": "THINK_TAGS", "open": "<think>", "close": "</think>",
                                       "wrapping_newlines": 0, "terminal_markers": []},
                         "assistant_marker": "<|im_start|>assistant\n", "message_end_markers": [],
                         "rejected_tokens": 0, "position_dimensions": 1, "prefix_boundary": null},
            "vision": {"feature_slots": 4, "max_image_tokens_per_prompt": 4096, "max_images": 4096,
                       "patches_per_feature": 9, "patch_capacities": [3456], "image": {"profile": "deepseek4"}},
            "assets": [{"name": "tokenizer", "bytes": 11_000_000, "sha256": "ef".repeat(32)}],
            "engram": {"compressed_vocabulary": 120000, "pad_token_id": 2, "token_map_entries": 151936,
                       "tables": [{"layer": 3, "rows": 4194304, "embedding_width": 512, "encoding": "NVFP4_E4M3_GLOBAL",
                                   "weight_bytes": 1_073_741_824u64, "scale_bytes": 134_217_728u64, "shards": 0}]},
            "ranks": [{"rank": 0, "architecture": [10, 3], "device_bytes": 160_000_000_000u64,
                       "mapped_host_bytes": 1_000_000, "peer_bytes": 0, "allocations": 40,
                       "capacity_allocations": [{"name": "kv", "capacity": "KV_PAGES", "unit_bytes": 36864,
                                                 "maximum": 57344, "memory": "DEVICE"},
                                                {"name": "state", "capacity": "STATE_ROWS", "unit_bytes": 4096,
                                                 "maximum": 64, "memory": "DEVICE"}],
                       "modules": 12, "functions": 3000, "graphs": 24, "events": 2, "abi_version": 9, "max_active": 32,
                       "execution_graphs": [
                           {"name": "decode_b32", "kind": "decode", "sampling_mode": "sample", "batch_capacity": 32,
                            "token_capacity": 1, "packed_tokens": 32, "prefill_capacity": 0, "draft_tokens": null,
                            "verify_tokens": null, "draft_source": "NO_DRAFT", "speculation_enabled": false,
                            "estimated_duration_ns": 2_100_000},
                           {"name": "verification_w5_b32_with_a_very_long_name_that_will_not_fit_anywhere", "kind": "verification",
                            "sampling_mode": "sample", "batch_capacity": 32, "token_capacity": 6, "packed_tokens": 192,
                            "prefill_capacity": 0, "draft_tokens": 5, "verify_tokens": 6, "draft_source": "MTP",
                            "speculation_enabled": true, "estimated_duration_ns": 4_000}],
                       "roce": null}],
            "graph_contracts": [],
        })
    }

    fn strip_escapes(text: &str) -> String {
        let mut output = String::new();
        let mut chars = text.chars();
        while let Some(c) = chars.next() {
            if c == '\x1b' {
                for c in chars.by_ref() {
                    if c == 'm' {
                        break;
                    }
                }
            } else {
                output.push(c);
            }
        }
        output
    }

    #[test]
    fn wordmark_rows_share_one_width() {
        let rows = wordmark();
        let width = visible(&rows[0]);
        assert!(rows.iter().all(|row| visible(row) == width), "{rows:?}");
        assert!(width <= MIN_WIDTH);
    }

    #[test]
    fn every_line_fits_the_width_and_panels_are_closed() {
        for width in [MIN_WIDTH, 80, MAX_WIDTH] {
            let plain = render(&document(), width, false);
            for line in plain.lines() {
                assert!(visible(line) <= width, "width {width}: {line:?} is {} wide", visible(line));
            }
            let opens = plain.lines().filter(|l| l.starts_with("╔═[")).count();
            let closes = plain.lines().filter(|l| l.starts_with('╚')).count();
            assert_eq!(opens, closes);
            for line in plain.lines().filter(|l| l.starts_with('║')) {
                assert_eq!(visible(line), width, "{line:?}");
                assert!(line.ends_with('║'));
            }
            for title in ["PACKAGE", "MODEL", "LIMITS", "CACHE", "SPECULATION", "SAMPLING DEFAULTS",
                          "FRONTEND", "VISION", "ENGRAM", "RANKS", "GRAPHS · RANK 0"] {
                assert!(plain.contains(&format!(" {title} ")), "{title}");
            }
        }
    }

    #[test]
    fn color_output_is_the_plain_output_with_escapes() {
        let plain = render(&document(), 80, false);
        let color = render(&document(), 80, true);
        assert!(color.contains("\x1b[1;38;2;"));
        assert_eq!(strip_escapes(&color), plain);
    }

    #[test]
    fn values_are_in_readable_units() {
        let plain = render(&document(), MAX_WIDTH, false);
        assert!(plain.contains("1,048,576 tokens"));
        assert!(plain.contains("3,670,016 tokens · 3.50 contexts"));
        assert!(plain.contains("36 KiB per page · 576 B per token"));
        assert!(plain.contains("57,344 at most · 64 backed at export"));
        assert!(plain.contains("paged latent"), "{plain}");
        assert!(plain.lines().filter(|l| l.contains("paged latent")).count() == 1, "{plain}");
        assert!(plain.contains("sliding window"));
        assert!(plain.contains("667.6 MiB primary + 102.45 GiB in 2 companion files"), "{plain}");
        assert!(plain.contains("sm_103 (compute capability 10.3)"));
        assert!(plain.contains("[██████████░░░░░░░░░░] 1 (0 – 2)"));
        assert!(plain.contains("[███████████████████░] 0.95"));
        assert!(plain.contains("DeepSeek V4.1 native encoder"));
        assert!(plain.contains("dsml parser · 1 wrapping newlines"));
        assert!(plain.contains("2.10 ms"));
        assert!(plain.contains("4 µs"));
        assert!(plain.lines().any(|l| l.contains("verification_w5_b32_with") && l.contains("…")), "{plain}");
        assert!(plain.contains("multi-token prediction head"));
        assert!(plain.contains("<|im_start|>assistant\\n"), "{plain}");
        assert!(plain.contains("signed by Local development signer · local-development · development key · 1 rank"));
    }

    #[test]
    fn long_values_wrap_inside_the_panel() {
        let mut document = document();
        document["package"]["path"] = json!("/a/very/long/directory/name/that/keeps/going/and/going/for/a/while/model.safetensors");
        let plain = render(&document, MIN_WIDTH, false);
        let file_lines: Vec<_> = plain.lines().filter(|l| l.contains("/a/very/long") || l.contains("model.safetensors")).collect();
        assert!(file_lines.len() >= 2, "{plain}");
        for line in plain.lines() {
            assert!(visible(line) <= MIN_WIDTH, "{line:?}");
        }
    }

    #[test]
    fn formatting_helpers() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1000), "1,000");
        assert_eq!(thousands(1_048_576), "1,048,576");
        assert_eq!(bytes(512), "512 B");
        assert_eq!(bytes(2048), "2 KiB");
        assert_eq!(bytes(3 * 1024 * 1024), "3.0 MiB");
        assert_eq!(bytes(5 * 1024 * 1024 * 1024), "5.00 GiB");
        assert_eq!(decimal(1.0), "1");
        assert_eq!(decimal(0.95), "0.95");
        assert_eq!(decimal(0.0), "0");
        assert_eq!(duration(0), "–");
        assert_eq!(Canvas::wrap("abc def ghi", 7), vec!["abc def", "ghi"]);
        assert_eq!(Canvas::wrap("abcdefghijkl", 8), vec!["abcdefgh", "ijkl"]);
    }
}
