//! Convert reference checkpoint configuration for native frontend tests.

use crate::frontend::{Asset, TokenizerSettings};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn read(asset: &Asset) -> Result<Value, String> {
    if asset.range.is_some() {
        return Err("reference fixture requires a source configuration file".into());
    }
    let bytes = std::fs::read(&asset.path).map_err(|e| e.to_string())?;
    if format!("{:x}", Sha256::digest(&bytes)) != asset.sha256 {
        return Err("reference fixture asset SHA256 mismatch".into());
    }
    serde_json::from_slice(&bytes).map_err(|e| e.to_string())
}

pub fn tokens(asset: &Asset) -> Result<TokenizerSettings, String> {
    let source = read(asset)?;
    let content = |value: &Value| value.get("content").unwrap_or(value).clone();
    let mut settings = serde_json::Map::new();
    for name in [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
    ] {
        if let Some(value) = source.get(name).filter(|v| !v.is_null()) {
            settings.insert(name.into(), content(value));
        }
    }
    if let Some(values) = source
        .get("additional_special_tokens")
        .and_then(Value::as_array)
    {
        settings.insert(
            "additional_special_tokens".into(),
            json!({"values": values.iter().map(content).collect::<Vec<_>>()}),
        );
    }
    serde_json::from_value(settings.into()).map_err(|e| e.to_string())
}

#[cfg(feature = "images")]
pub fn image(
    preprocessor: &Asset,
    model: &Asset,
) -> Result<crate::frontend::images::ImageSettings, String> {
    let processor = read(preprocessor)?;
    let model = read(model)?;
    let settings = if processor["image_processor_type"] == "DeepSeekV4ImageProcessor" {
        json!({"deepseek4": {}})
    } else if model["model_type"] == "mimo_v2" {
        json!({"mimo": {
            "min_pixels": processor["min_pixels"], "max_pixels": processor["max_pixels"],
            "image_mean": processor["image_mean"], "image_std": processor["image_std"],
        }})
    } else {
        json!({"qwen": {
            "min_pixels": processor["size"]["shortest_edge"], "max_pixels": processor["size"]["longest_edge"],
            "position_side": 48, "image_token_id": model["image_token_id"], "video_token_id": model["video_token_id"],
        }})
    };
    serde_json::from_value(settings).map_err(|e| e.to_string())
}
