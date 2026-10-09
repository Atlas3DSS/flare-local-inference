use super::RgbImage;
use crate::Result;

pub(super) fn exif_orientation(bytes: &[u8]) -> Result<u16> {
    let little = match bytes.get(..2) {
        Some(b"II") => true,
        Some(b"MM") => false,
        _ => return Err("invalid EXIF byte order".into()),
    };
    let u16_at = |at: usize| -> Result<u16> {
        let data = bytes
            .get(at..at.saturating_add(2))
            .ok_or("truncated EXIF field")?;
        let array = [data[0], data[1]];
        Ok(if little {
            u16::from_le_bytes(array)
        } else {
            u16::from_be_bytes(array)
        })
    };
    let u32_at = |at: usize| -> Result<u32> {
        let data = bytes
            .get(at..at.saturating_add(4))
            .ok_or("truncated EXIF field")?;
        let array = [data[0], data[1], data[2], data[3]];
        Ok(if little {
            u32::from_le_bytes(array)
        } else {
            u32::from_be_bytes(array)
        })
    };
    if u16_at(2)? != 42 {
        return Err("invalid EXIF TIFF magic".into());
    }
    let offset = u32_at(4)? as usize;
    if offset < 8 {
        return Err("invalid EXIF directory offset".into());
    }
    let count = usize::from(u16_at(offset)?);
    let entries = offset.checked_add(2).ok_or("EXIF directory overflow")?;
    let end = entries
        .checked_add(count * 12)
        .and_then(|n| n.checked_add(4))
        .ok_or("EXIF directory overflow")?;
    if end > bytes.len() {
        return Err("truncated EXIF directory".into());
    }
    let mut orientation = None;
    for entry in (entries..end - 4).step_by(12) {
        if u16_at(entry)? != 0x0112 {
            continue;
        }
        if orientation.is_some() || u16_at(entry + 2)? != 3 || u32_at(entry + 4)? != 1 {
            return Err("EXIF orientation must be one SHORT value".into());
        }
        let value = u16_at(entry + 8)?;
        if !(1..=8).contains(&value) {
            return Err("EXIF orientation must be in 1..=8".into());
        }
        orientation = Some(value);
    }
    Ok(orientation.unwrap_or(1))
}

pub(super) fn jpeg_orientation(bytes: &[u8]) -> Result<u16> {
    if !bytes.starts_with(&[0xff, 0xd8]) {
        return Err("invalid JPEG signature".into());
    }
    let mut offset = 2;
    let mut orientation = None;
    while offset < bytes.len() {
        if bytes[offset] != 0xff {
            return Err("invalid JPEG marker".into());
        }
        while bytes.get(offset) == Some(&0xff) {
            offset += 1;
        }
        let marker = *bytes.get(offset).ok_or("truncated JPEG marker")?;
        offset += 1;
        if marker == 0xd9 {
            break;
        }
        if marker == 0x01 || (0xd0..=0xd7).contains(&marker) {
            continue;
        }
        if marker == 0 || marker == 0xd8 {
            return Err("invalid JPEG marker".into());
        }
        let size = bytes
            .get(offset..offset + 2)
            .ok_or("truncated JPEG segment")?;
        let length = usize::from(u16::from_be_bytes([size[0], size[1]]));
        if length < 2 {
            return Err("invalid JPEG segment length".into());
        }
        let payload = bytes
            .get(offset + 2..offset + length)
            .ok_or("truncated JPEG segment")?;
        if marker == 0xe1 && payload.starts_with(b"Exif\0\0") {
            if orientation.is_some() {
                return Err("multiple JPEG EXIF segments are not supported".into());
            }
            orientation = Some(exif_orientation(&payload[6..])?);
        }
        if marker == 0xda {
            break;
        }
        offset += length;
    }
    Ok(orientation.unwrap_or(1))
}

pub(super) fn orient(image: RgbImage, orientation: u16) -> RgbImage {
    if orientation == 1 {
        return image;
    }
    let (width, height) = if orientation >= 5 {
        (image.height, image.width)
    } else {
        (image.width, image.height)
    };
    let mut pixels = vec![0; image.pixels.len()];
    for y in 0..image.height {
        for x in 0..image.width {
            let (dx, dy) = match orientation {
                2 => (image.width - 1 - x, y),
                3 => (image.width - 1 - x, image.height - 1 - y),
                4 => (x, image.height - 1 - y),
                5 => (y, x),
                6 => (image.height - 1 - y, x),
                7 => (image.height - 1 - y, image.width - 1 - x),
                8 => (y, image.width - 1 - x),
                _ => unreachable!("orientation was validated when reading EXIF"),
            };
            let src = (y * image.width + x) * 3;
            let dst = (dy * width + dx) * 3;
            pixels[dst..dst + 3].copy_from_slice(&image.pixels[src..src + 3]);
        }
    }
    RgbImage {
        width,
        height,
        pixels,
    }
}
