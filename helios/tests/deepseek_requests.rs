#![cfg(all(feature = "server", feature = "runtime-package"))]

use flare_helios::{frontend, model, scheduler};
#[path = "support/frontend.rs"]
mod support;

use flare_helios::{
    frontend::{
        Asset, Frontend, FrontendAssets, FrontendSettings, PromptEncoder, TokenizerSettings,
    },
    grammar::GrammarCompiler,
    output::tagged::OutputStart,
    server::{PreparedRequest, prepare_request},
};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU64, Ordering};
use tokenizers::{AddedToken, Tokenizer};

static FIXTURE_ID: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    frontend: Frontend,
    compiler: GrammarCompiler,
    config: model::ModelConfig,
}

fn tokenizer_asset() -> Asset {
    let mut alphabet: Vec<_> = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
        .into_iter()
        .collect();
    alphabet.sort_unstable();
    let model = tokenizers::models::bpe::BPE::builder()
        .vocab_and_merges(
            alphabet
                .into_iter()
                .enumerate()
                .map(|(id, c)| (c.to_string(), id as u32))
                .collect::<tokenizers::models::bpe::Vocab>(),
            vec![],
        )
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::byte_level::ByteLevel::new(false, true, false),
    ));
    tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    tokenizer
        .add_special_tokens(
            [
                "<｜begin▁of▁sentence｜>",
                "<｜end▁of▁sentence｜>",
                "<｜System｜>",
                "<｜User｜>",
                "<｜Assistant｜>",
                "<think>",
                "</think>",
                "<｜DSML｜ calls>",
                "</｜DSML｜ calls>",
                "<｜deepseek_image｜>",
            ]
            .map(|s| AddedToken::from(s, true)),
        )
        .unwrap();
    let bytes = tokenizer.to_string(false).unwrap();
    let path = std::env::temp_dir().join(format!(
        "flare-deepseek-request-{}-{}.json",
        std::process::id(),
        FIXTURE_ID.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, &bytes).unwrap();
    Asset {
        path,
        sha256: format!("{:x}", Sha256::digest(bytes.as_bytes())),
        range: None,
    }
}

impl Fixture {
    fn new(encoder: PromptEncoder) -> Self {
        let asset = tokenizer_asset();
        let settings: FrontendSettings = serde_json::from_value(json!({
            "prompt_encoder": i32::from(encoder), "tokenizer": {}, "eos_token_ids": [257],
            "image_placeholder_token_id":Tokenizer::from_file(&asset.path).unwrap().token_to_id("<｜deepseek_image｜>").unwrap()
        }))
        .unwrap();
        let frontend = Frontend::load_with_settings(
            FrontendAssets {
                tokenizer: Asset {
                    path: asset.path.clone(),
                    sha256: asset.sha256.clone(),
                    range: asset.range,
                },
                tokens: TokenizerSettings::default(),
                chat_template: None,
            },
            Map::new(),
            &settings,
        )
        .unwrap();
        std::fs::remove_file(asset.path).unwrap();
        let mut config =
            support::model_config(support::contract(if encoder == PromptEncoder::DeepseekV4 {
                "deepseek4"
            } else {
                "deepseek41"
            }));
        config.eos_token_ids = vec![
            frontend
                .tokenizer()
                .token_to_id("<｜end▁of▁sentence｜>")
                .unwrap(),
        ];
        let compiler = GrammarCompiler::new(
            frontend.tokenizer(),
            config.contract.vocabulary,
            &config.eos_token_ids,
        )
        .unwrap();
        Self {
            frontend,
            compiler,
            config,
        }
    }

    fn prepare(&self, extra: Value) -> flare_helios::Result<PreparedRequest> {
        let mut body = json!({"model": self.config.model_id,
            "messages": [{"role":"system","content":"Policy"},{"role":"user","content":"Hello"}]});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        prepare_request(&self.frontend, None, &self.compiler, &self.config, body)
    }

    fn accepts(&self, extra: Value, text: &str) -> bool {
        let mut request = self.prepare(extra).unwrap();
        let mut grammar = request.prompt.grammar.take().unwrap();
        self.frontend
            .encode(text, false)
            .unwrap()
            .into_iter()
            .chain(self.config.eos_token_ids.iter().copied())
            .all(|token| grammar.advance(token).is_ok())
    }
}

#[test]
fn typed_native_settings_do_not_require_jinja_and_legacy_settings_do() {
    for encoder in [PromptEncoder::DeepseekV4, PromptEncoder::DeepseekV41] {
        let fixture = Fixture::new(encoder);
        assert_eq!(fixture.frontend.prompt_encoder(), encoder);
        let request = fixture.prepare(json!({})).unwrap();
        assert_eq!(request.output.content.start, OutputStart::Reasoning);
        assert!(request.rendered_prompt.ends_with("<｜Assistant｜><think>"));
        let boundary = request.prompt.system_boundary.unwrap();
        assert!(boundary > 0 && boundary < request.prompt.tokens.len());
    }
    let asset = tokenizer_asset();
    let settings: FrontendSettings =
        serde_json::from_value(json!({"tokenizer":{},"eos_token_ids":[257]})).unwrap();
    assert_eq!(settings.prompt_encoder, PromptEncoder::Jinja);
    let assets = FrontendAssets {
        tokenizer: Asset {
            path: asset.path.clone(),
            sha256: asset.sha256.clone(),
            range: asset.range,
        },
        tokens: TokenizerSettings::default(),
        chat_template: None,
    };
    assert!(Frontend::load_with_settings(assets, Map::new(), &settings).is_err());
    std::fs::remove_file(asset.path).unwrap();
    assert!(
        serde_json::from_value::<FrontendSettings>(
            json!({"prompt_encoder":3,"tokenizer":{},"eos_token_ids":[257]})
        )
        .is_err()
    );
}

#[test]
fn json_object_uses_the_official_prompt_and_schema_instructions_remain_explicit() {
    for encoder in [PromptEncoder::DeepseekV4, PromptEncoder::DeepseekV41] {
        let f = Fixture::new(encoder);
        let object = json!({"enable_thinking":false,"response_format":{"type":"json_object"}});
        let actual = f.prepare(object.clone()).unwrap();
        let expected = f
            .frontend
            .render_chat(
                &[
                    json!({"role":"system","content":"Policy"}),
                    json!({"role":"user","content":"Hello"}),
                ],
                json!({"enable_thinking":false,"response_format":"json_object"})
                    .as_object()
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(actual.rendered_prompt, expected.text);
        assert_eq!(actual.prompt.tokens, expected.token_ids);
        assert_eq!(actual.output.content.start, OutputStart::Content);
        assert!(actual.prompt.grammar.is_some());
        assert!(f.accepts(object.clone(), "{\"ok\":true}"));
        assert!(!f.accepts(object, "[1]"));
        let schema = json!({"enable_thinking":false,"response_format":{"type":"json_schema","json_schema":{
            "name":"answer","strict":true,"schema":{"type":"object","properties":{"ok":{"const":true}},"required":["ok"],"additionalProperties":false}
        }}});
        let request = f.prepare(schema.clone()).unwrap();
        assert!(request.rendered_prompt.contains("JSON schema:"));
        assert!(f.accepts(schema.clone(), "{\"ok\":true}"));
        assert!(!f.accepts(schema, "{\"ok\":false}"));
    }
}

#[test]
fn required_tools_in_chat_mode_start_inside_the_dsml_envelope() {
    for encoder in [PromptEncoder::DeepseekV4, PromptEncoder::DeepseekV41] {
        let f = Fixture::new(encoder);
        let options = json!({"enable_thinking":false,"tool_choice":"required",
            "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],
            "response_format":{"type":"json_object"}});
        let request = f.prepare(options.clone()).unwrap();
        assert_eq!(
            request.output.content.start,
            OutputStart::ToolBody,
            "{}",
            request.rendered_prompt
        );
        let (open, invoke, close) = if encoder == PromptEncoder::DeepseekV4 {
            (
                "<｜DSML｜tool_calls>",
                "<｜DSML｜invoke name=\"lookup\"></｜DSML｜invoke>",
                "</｜DSML｜tool_calls>",
            )
        } else {
            (
                "<｜DSML｜ calls>",
                "<｜DSML｜ invoke name=\"lookup\"></｜DSML｜ invoke>",
                "</｜DSML｜ calls>",
            )
        };
        assert!(request.rendered_prompt.trim_end().ends_with(open));
        assert!(request.prompt.grammar.is_some());
        assert!(!f.accepts(options.clone(), "{\"ok\":true}"));
        let completion = format!("{invoke}\n{close}{{\"ok\":true}}");
        assert!(f.accepts(options, &completion));
        let mut output = flare_helios::output::OutputPipeline::new(request.output);
        let mut events = output
            .push(flare_helios::output::Input::Text(&completion))
            .unwrap();
        events.extend(
            output
                .finish(flare_helios::output::Termination::Eos)
                .unwrap(),
        );
        assert!(events.iter().any(|event| matches!(event,
            flare_helios::output::Event::ToolBegin { name, .. } if name == "lookup")));
        assert!(
            events.contains(&flare_helios::output::Event::ToolArguments {
                index: 0,
                text: "{}".into()
            })
        );
        assert!(events.contains(&flare_helios::output::Event::Content(
            "{\"ok\":true}".into()
        )));
    }
}

#[test]
fn native_reasoning_options_are_validated_through_request_preparation() {
    for encoder in [PromptEncoder::DeepseekV4, PromptEncoder::DeepseekV41] {
        let f = Fixture::new(encoder);
        for effort in ["low", "high", "xhigh", "max"] {
            assert!(f.prepare(json!({"reasoning_effort":effort})).is_ok());
        }
        for extra in [
            json!({"chat_template_kwargs":{"reasoning_effort":75}}),
            json!({"chat_template_kwargs":{"drop_thinking":true}}),
            json!({"chat_template_kwargs":{"continue_final_message":true}}),
            json!({"chat_template_kwargs":{"add_generation_prompt":false}}),
            json!({"tool_choice":"required","tools":[{"type":"function","function":{"name":"lookup"}}]}),
        ] {
            assert!(f.prepare(extra.clone()).is_err(), "{extra}");
        }
    }
}

#[test]
fn base64_images_follow_the_rendered_placeholder_order() {
    use base64::Engine;
    use flare_helios::frontend::images::{ImageLimits, ImageProcessor, ImageSettings};

    fn encoded_image(rgb: [u8; 3]) -> (Vec<u8>, String) {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, 1, 1);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&rgb).unwrap();
        }
        let url = format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(&bytes)
        );
        (bytes, url)
    }
    let mut f = Fixture::new(PromptEncoder::DeepseekV4);
    f.config.contract = support::contract("deepseek4");
    let processor =
        ImageProcessor::load(ImageSettings::Deepseek4 {}, ImageLimits::default()).unwrap();
    let (red, red_url) = encoded_image([255, 0, 0]);
    let (green, green_url) = encoded_image([0, 255, 0]);
    let messages = json!([{"role":"user","content":[
        {"type":"image_url","image_url":{"url":green_url}},
        {"type":"text","text":"First"},
        {"type":"image_url","image_url":{"url":red_url}},
        {"type":"text","text":"Second"}
    ]}]);
    let rendered = f
        .frontend
        .render_chat(messages.as_array().unwrap(), &Map::new())
        .unwrap();
    assert_eq!(rendered.image_sources, vec![green_url, red_url]);
    let request = prepare_request(
        &f.frontend,
        Some(&processor),
        &f.compiler,
        &f.config,
        json!({"model":f.config.model_id,"messages":messages}),
    )
    .unwrap();
    let expected = processor
        .process(&[
            processor.decode_png(&green).unwrap(),
            processor.decode_png(&red).unwrap(),
        ])
        .unwrap();
    let actual = request.prompt.images.unwrap();
    assert_eq!(actual.pixel_values, expected.pixel_values);
    assert_eq!(actual.geometry.len(), 2);
    assert_eq!(request.rendered_prompt, rendered.text);
}
