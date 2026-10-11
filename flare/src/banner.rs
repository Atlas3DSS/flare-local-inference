use std::io::{IsTerminal, Write};

pub fn print() {
    let stderr = std::io::stderr();
    if !stderr.is_terminal() {
        return;
    }
    let colors = [
        0xa95348, 0xe54b32, 0xff7033, 0xff9c42, 0xffb957, 0xffd27b, 0xffd27b, 0xffb957, 0xff9c42,
        0xffbf69, 0xffe8b0, 0xff9c42, 0xe54b32, 0x96352e,
    ];
    let spans: [&[(usize, usize)]; 4] = [
        &[(36, 43)],
        &[(28, 45)],
        &[(11, 17), (24, 25), (27, 41), (45, 48)],
        &[(6, 10), (18, 24), (36, 40), (43, 48), (51, 53)],
    ];
    let mut size: libc::winsize = unsafe { std::mem::zeroed() };
    // SAFETY: ioctl writes the terminal dimensions into a valid winsize.
    let width = if unsafe { libc::ioctl(libc::STDERR_FILENO, libc::TIOCGWINSZ, &mut size) } == 0
        && size.ws_col > 0
    {
        usize::from(size.ws_col)
    } else {
        80
    };
    let mut out = stderr.lock();
    for (row, line) in include_str!("banner.txt").lines().enumerate() {
        for (column, character) in line.chars().take(width).enumerate() {
            let mut color = colors[row];
            if row >= 10
                && !spans[row - 10]
                    .iter()
                    .any(|&(start, end)| (start..end).contains(&column))
            {
                let limb = 41.2 + 275.0
                    - (275.0_f64.powi(2) - (column as f64 * 2.0 + 0.5 - 71.5).powi(2)).sqrt();
                let depth = row as f64 * 4.0 + 1.5 - limb;
                color = if depth < 2.0 {
                    0xfff1a8
                } else if depth < 6.0 {
                    0xffd15c
                } else {
                    0xffad38
                };
            }
            let wordmark = (3..=8).contains(&row) && (5..44).contains(&column);
            let highlight = "⣿⣾⣷✦".contains(character);
            if wordmark {
                color = 0xfff0c7;
            } else if highlight {
                color = 0xfff8e7;
            }
            let weight = if wordmark || highlight { 1 } else { 22 };
            let _ = write!(
                out,
                "\x1b[{weight};38;2;{};{};{}m{character}",
                color >> 16,
                (color >> 8) & 255,
                color & 255
            );
        }
        let _ = writeln!(out, "\x1b[0m");
    }
    let _ = writeln!(out);
    let _ = out.flush();
}
