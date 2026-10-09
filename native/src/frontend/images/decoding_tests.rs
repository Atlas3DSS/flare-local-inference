use super::{ImageLimits, RgbImage, orientation, tests::processor};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};

fn fixtures() -> (PathBuf, Value) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/frontend");
    let manifest =
        serde_json::from_slice(&fs::read(root.join("qwen38_jpeg_exif.json")).unwrap()).unwrap();
    (root, manifest)
}

#[test]
fn jpeg_rgb_and_all_exif_orientations_match_pillow_within_two() {
    let (root, manifest) = fixtures();
    let processor = processor(ImageLimits::default());
    for case in manifest["cases"].as_array().unwrap() {
        let encoded = fs::read(root.join(case["encoded"].as_str().unwrap())).unwrap();
        let expected = fs::read(root.join(case["rgb"].as_str().unwrap())).unwrap();
        let image = if case["format"] == "JPEG" {
            processor.decode_jpeg(&encoded)
        } else {
            processor.decode_png(&encoded)
        }
        .unwrap();
        assert_eq!(
            [image.width(), image.height()],
            [
                case["width"].as_u64().unwrap() as usize,
                case["height"].as_u64().unwrap() as usize
            ],
            "{}",
            case["name"]
        );
        assert_eq!(image.pixels().len(), expected.len());
        let max = image
            .pixels()
            .iter()
            .zip(&expected)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        assert!(
            u64::from(max) <= case["max_channel_error"].as_u64().unwrap(),
            "{}: maximum channel error {max}",
            case["name"]
        );
        println!("{}: maximum channel error {max}", case["name"]);
    }
}

#[test]
fn identical_decoded_pixels_preserve_exact_checkpoint_preprocessing() {
    let (root, manifest) = fixtures();
    let processor = processor(ImageLimits::default());
    let hash = |bytes: &[u8]| format!("{:x}", Sha256::digest(bytes));
    for case in manifest["cases"].as_array().unwrap() {
        let pixels = fs::read(root.join(case["rgb"].as_str().unwrap())).unwrap();
        let image = RgbImage::new(
            case["width"].as_u64().unwrap() as usize,
            case["height"].as_u64().unwrap() as usize,
            pixels,
        )
        .unwrap();
        let (geometry, resized) = processor.resized_rgb(&image).unwrap();
        assert_eq!(serde_json::json!(geometry.grid_thw), case["grid_thw"]);
        assert_eq!(
            hash(&resized),
            case["resized_rgb_sha256"],
            "{} resize",
            case["name"]
        );
        let processed = processor.process(&[image]).unwrap();
        let bytes: Vec<_> = processed
            .pixel_values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(
            hash(&bytes),
            case["pixel_values_sha256"],
            "{} normalization/patches",
            case["name"]
        );
    }
}

fn exif(value: u16, little: bool) -> Vec<u8> {
    let u16_bytes = |n: u16| {
        if little {
            n.to_le_bytes()
        } else {
            n.to_be_bytes()
        }
    };
    let u32_bytes = |n: u32| {
        if little {
            n.to_le_bytes()
        } else {
            n.to_be_bytes()
        }
    };
    let mut data = if little {
        b"II".to_vec()
    } else {
        b"MM".to_vec()
    };
    data.extend(u16_bytes(42));
    data.extend(u32_bytes(8));
    data.extend(u16_bytes(1));
    data.extend(u16_bytes(0x112));
    data.extend(u16_bytes(3));
    data.extend(u32_bytes(1));
    data.extend(u16_bytes(value));
    data.extend([0; 6]);
    data
}

#[test]
fn exif_reads_both_endian_orders_and_rejects_ambiguous_or_truncated_orientation() {
    for little in [false, true] {
        for value in 1..=8 {
            let data = exif(value, little);
            assert_eq!(orientation::exif_orientation(&data).unwrap(), value);
            for end in 0..data.len() {
                assert!(orientation::exif_orientation(&data[..end]).is_err());
            }
        }
        for value in [0, 9, u16::MAX] {
            assert!(orientation::exif_orientation(&exif(value, little)).is_err());
        }
    }
    for (offset, values) in [
        (0, vec![0, 0]),
        (2, vec![0, 0]),
        (4, vec![255; 4]),
        (8, vec![255; 2]),
        (12, vec![4, 0]),
        (14, vec![2, 0, 0, 0]),
    ] {
        let mut data = exif(1, true);
        data[offset..offset + values.len()].copy_from_slice(&values);
        assert!(orientation::exif_orientation(&data).is_err());
    }
    let mut duplicate = exif(1, true);
    duplicate[8] = 2;
    let entry = duplicate[10..22].to_vec();
    duplicate.splice(22..22, entry);
    assert!(orientation::exif_orientation(&duplicate).is_err());
}

#[test]
fn jpeg_limits_and_malformed_data_fail_before_pixel_decompression() {
    let (root, _) = fixtures();
    let bytes = fs::read(root.join("jpeg-exif-inputs/RGB-0-2.jpg")).unwrap();
    let small = processor(ImageLimits {
        max_encoded_bytes: bytes.len() - 1,
        ..ImageLimits::default()
    });
    assert!(
        small
            .decode_jpeg(&bytes)
            .err()
            .unwrap()
            .contains("byte limit")
    );
    let small = processor(ImageLimits {
        max_input_pixels: 37 * 29 - 1,
        ..ImageLimits::default()
    });
    assert!(
        small
            .decode_jpeg(&bytes)
            .err()
            .unwrap()
            .contains("input pixel limit")
    );
    let normal = processor(ImageLimits::default());
    for end in [0, 1, 2, 30, bytes.len() / 2, bytes.len() - 2] {
        assert!(
            normal.decode_jpeg(&bytes[..end]).is_err(),
            "truncated length {end}"
        );
    }
    let mut huge = bytes.clone();
    let sof = huge.windows(2).position(|v| v == [255, 192]).unwrap();
    huge[sof + 5..sof + 9].copy_from_slice(&[128, 0, 128, 0]);
    assert!(
        normal
            .decode_jpeg(&huge)
            .err()
            .unwrap()
            .contains("input pixel limit")
    );
    let mut bad = vec![255, 216, 255, 225, 0, 10];
    bad.extend(b"Exif\0\0xx");
    bad.extend(&bytes[2..]);
    assert!(normal.decode_jpeg(&bad).err().unwrap().contains("EXIF"));
    let mut duplicate = vec![255, 216];
    for _ in 0..2 {
        let payload = exif(1, true);
        duplicate.extend([255, 225, 0, (payload.len() + 8) as u8]);
        duplicate.extend(b"Exif\0\0");
        duplicate.extend(payload);
    }
    duplicate.extend(&bytes[2..]);
    assert!(
        normal
            .decode_jpeg(&duplicate)
            .err()
            .unwrap()
            .contains("multiple JPEG EXIF")
    );
}

#[test]
fn malformed_png_exif_is_rejected() {
    let mut info = png::Info::with_size(1, 1);
    info.color_type = png::ColorType::Rgb;
    info.bit_depth = png::BitDepth::Eight;
    info.exif_metadata = Some(std::borrow::Cow::Owned(exif(9, true)));
    let mut bytes = vec![];
    {
        let encoder = png::Encoder::with_info(&mut bytes, info).unwrap();
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&[1, 2, 3]).unwrap();
    }
    assert!(
        processor(ImageLimits::default())
            .decode_png(&bytes)
            .err()
            .unwrap()
            .contains("orientation")
    );
}
