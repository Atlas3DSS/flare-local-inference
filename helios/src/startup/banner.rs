use std::{fmt::Write as _, time::Duration};

const ART: &str = include_str!("banner.txt");
pub(super) const DURATION: Duration = Duration::from_secs(3);
const COLORS: [u32; 14] = [
    0xa95348, 0xe54b32, 0xff7033, 0xff9c42, 0xffb957, 0xffd27b, 0xffd27b, 0xffb957, 0xff9c42,
    0xffbf69, 0xffe8b0, 0xff9c42, 0xe54b32, 0x96352e,
];
const SURFACE_SPANS: [&[(usize, usize)]; 4] = [
    &[(36, 43)],
    &[(28, 45)],
    &[(11, 17), (24, 25), (27, 41), (45, 48)],
    &[(6, 10), (18, 24), (36, 40), (43, 48), (51, 53)],
];

pub(super) fn frame(progress: f64, width: usize) -> String {
    let frames: Vec<_> = ART.split("\n---\n").collect();
    let index = (progress * (frames.len() - 1) as f64).round() as usize;
    let art = frames[index];
    let mut output = String::new();
    let cooling = ((progress - 0.15) / 0.8).clamp(0.0, 1.0);
    let cooling = cooling * cooling * (3.0 - 2.0 * cooling);
    for (row, line) in art.lines().enumerate() {
        output.push_str("\r\x1b[2K");
        for (column, character) in line
            .chars()
            .chain(std::iter::repeat(' '))
            .take(width.min(72))
            .enumerate()
        {
            let wordmark = (3..=8).contains(&row) && (5..44).contains(&column);
            let highlight = "⣿⣾⣷✦".contains(character);
            let color = if wordmark {
                0xfff0c7
            } else if highlight {
                0xfff8e7
            } else if row >= 10 {
                let limb = 41.2 + 275.0
                    - (275.0_f64.powi(2) - (column as f64 * 2.0 + 0.5 - 71.5).powi(2)).sqrt();
                let depth = row as f64 * 4.0 + 1.5 - limb;
                let surface = if depth < 2.0 {
                    0xfff1a8
                } else if depth < 6.0 {
                    0xffd15c
                } else {
                    0xffad38
                };
                if SURFACE_SPANS[row - 10]
                    .iter()
                    .any(|&(start, end)| (start..end).contains(&column))
                {
                    [0, 8, 16].into_iter().fold(0, |color, shift| {
                        let hot = ((surface >> shift) & 255) as f64;
                        let cool = ((COLORS[row] >> shift) & 255) as f64;
                        color | (((hot * (1.0 - cooling) + cool * cooling).round() as u32) << shift)
                    })
                } else {
                    surface
                }
            } else {
                COLORS[row]
            };
            let _ = write!(
                output,
                "\x1b[{};38;2;{};{};{}m{character}",
                if wordmark || highlight { 1 } else { 22 },
                color >> 16,
                (color >> 8) & 255,
                color & 255,
            );
        }
        output.push_str("\x1b[0m\n");
    }
    output.push('\n');
    output
}
