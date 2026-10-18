#![cfg(feature = "server")]
use flare_helios::{frontend, model, scheduler};
#[path = "support/frontend.rs"]
mod support;
use flare_helios::frontend::{Asset, Frontend, FrontendAssets};
use serde_json::{Map, json};
use sha2::{Digest, Sha256};
#[test]
#[ignore = "requires FLARE_DEEPSEEK_TEST_SNAPSHOT pointing to the DeepSeek checkpoint"]
fn canonical_text_template_renders_in_the_native_frontend() {
    let root = std::path::PathBuf::from(std::env::var_os("FLARE_DEEPSEEK_TEST_SNAPSHOT").unwrap());
    let asset = |path: std::path::PathBuf| Asset {
        sha256: format!("{:x}", Sha256::digest(std::fs::read(&path).unwrap())),
        path,
        range: None,
    };
    let frontend = Frontend::load(
        FrontendAssets {
            tokenizer: asset(root.join("tokenizer.json")),
            tokens: support::tokens(&asset(root.join("tokenizer_config.json"))).unwrap(),
            chat_template: asset(
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../ignition/recipes/deepseek41/assets/chat_template.jinja"),
            ),
        },
        Map::new(),
    )
    .unwrap();
    let messages = [json!({"role":"user", "content":"Hello"})];
    let mut options = Map::new();
    options.insert("add_generation_prompt".into(), true.into());
    let prompt = frontend.render_chat(&messages, &options).unwrap();
    assert_eq!(
        prompt.text,
        "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<｜User｜>Hello<｜Assistant｜><think>"
    );
    assert!(!prompt.token_ids.is_empty());
    let output = flare_helios::model::prepare_output(
        &support::model_config(support::contract("deepseek41")),
        &frontend,
        &prompt,
        &messages,
        &options,
        false,
    )
    .unwrap();
    assert!(output.content.reasoning);
    options.insert("enable_thinking".into(), false.into());
    let prompt = frontend.render_chat(&messages, &options).unwrap();
    assert_eq!(
        prompt.text,
        "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>"
    );
    let output = flare_helios::model::prepare_output(
        &support::model_config(support::contract("deepseek41")),
        &frontend,
        &prompt,
        &messages,
        &options,
        false,
    )
    .unwrap();
    assert!(!output.content.reasoning);
}
