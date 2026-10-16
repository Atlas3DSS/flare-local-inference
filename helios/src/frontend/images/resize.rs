// Coefficient generation follows PyTorch's uint8 antialiased bicubic kernel.
// See helios/licenses/PyTorch.txt and helios/README.md for source attribution.

use super::RgbImage;

struct Axis {
    taps: Vec<(usize, Vec<i32>)>,
    precision: u32,
}

fn cubic(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        ((1.5 * x - 2.5) * x) * x + 1.0
    } else if x < 2.0 {
        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    } else {
        0.0
    }
}

fn axis(input: usize, output: usize, pillow: bool, linear: bool) -> Axis {
    let scale = input as f64 / output as f64;
    let support = (if linear { 1.0 } else { 2.0 }) * scale.max(1.0);
    let inverse_scale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let max_taps = support.ceil() as usize * 2 + 1;
    let mut maximum = 0.0f64;
    let floating: Vec<_> = (0..output)
        .map(|i| {
            let center = scale * (i as f64 + 0.5);
            let start = ((center - support + 0.5) as i64).max(0) as usize;
            let end = ((center + support + 0.5) as usize).min(input);
            let count = (end - start).min(max_taps);
            let mut weights: Vec<_> = (0..count)
                .map(|j| {
                    let x = (j as f64 + start as f64 - center + 0.5) * inverse_scale;
                    if linear {
                        (1.0 - x.abs()).max(0.0)
                    } else {
                        cubic(x)
                    }
                })
                .collect();
            let sum: f64 = weights.iter().sum();
            if sum != 0.0 {
                for weight in &mut weights {
                    *weight /= sum;
                    maximum = maximum.max(*weight);
                }
            }
            (start, weights)
        })
        .collect();
    let mut precision = 0;
    while precision < 22 {
        let next = (0.5 + maximum * f64::from(1 << (precision + 1))) as i32;
        if next >= 1 << 15 {
            break;
        }
        precision += 1;
    }
    if pillow {
        precision = 22;
    }
    let multiplier = f64::from(1 << precision);
    let taps = floating
        .into_iter()
        .map(|(start, weights)| {
            (
                start,
                weights
                    .into_iter()
                    .map(|weight| {
                        let scaled = weight * multiplier;
                        if scaled < 0.0 {
                            (scaled - 0.5) as i32
                        } else {
                            (scaled + 0.5) as i32
                        }
                    })
                    .collect(),
            )
        })
        .collect();
    Axis { taps, precision }
}

pub(super) fn resize(input: &RgbImage, width: usize, height: usize) -> Vec<u8> {
    resample(input, width, height, false, false)
}

pub(super) fn pillow(input: &RgbImage, width: usize, height: usize) -> Vec<u8> {
    resample(input, width, height, true, false)
}

pub(super) fn bilinear(input: &RgbImage, width: usize, height: usize) -> Vec<u8> {
    resample(input, width, height, false, true)
}

fn resample(input: &RgbImage, width: usize, height: usize, pillow: bool, linear: bool) -> Vec<u8> {
    let source = &input.pixels;
    let horizontal = if width == input.width {
        None
    } else {
        let axis = axis(input.width, width, pillow, linear);
        let mut pixels = vec![0; input.height * width * 3];
        for y in 0..input.height {
            for (x, (start, weights)) in axis.taps.iter().enumerate() {
                for channel in 0..3 {
                    let mut value = 1i64 << (axis.precision - 1);
                    for (tap, weight) in weights.iter().enumerate() {
                        value += i64::from(source[(y * input.width + start + tap) * 3 + channel])
                            * i64::from(*weight);
                    }
                    pixels[(y * width + x) * 3 + channel] =
                        (value >> axis.precision).clamp(0, 255) as u8;
                }
            }
        }
        Some(pixels)
    };
    let source = horizontal.as_ref().unwrap_or(source);
    if height == input.height {
        return source.clone();
    }
    let axis = axis(input.height, height, pillow, linear);
    let mut pixels = vec![0; height * width * 3];
    for (y, (start, weights)) in axis.taps.iter().enumerate() {
        for x in 0..width * 3 {
            let mut value = 1i64 << (axis.precision - 1);
            for (tap, weight) in weights.iter().enumerate() {
                value += i64::from(source[(start + tap) * width * 3 + x]) * i64::from(*weight);
            }
            pixels[y * width * 3 + x] = (value >> axis.precision).clamp(0, 255) as u8;
        }
    }
    pixels
}
