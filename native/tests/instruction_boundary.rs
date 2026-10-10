#![cfg(feature = "server")]

use flare_native::{
    frontend::{
        Asset, Frontend, FrontendAssets,
        images::{ImageLimits, ImageProcessor},
    },
    grammar::GrammarCompiler,
    model::{ModelConfig, ModelContract, SamplingParameters, qwen4exp::Qwen4Exp},
    scheduler::SchedulerLimits,
    server::{PreparedRequest, prepare_request},
};
use serde_json::{Map, Value, json};
use std::{fs, path::PathBuf};

struct Fixture {
    frontend: Frontend,
    processor: ImageProcessor,
    compiler: GrammarCompiler,
    config: ModelConfig,
}

impl Fixture {
    fn load(defaults: Map<String, Value>) -> Self {
        let snapshot = PathBuf::from(
            std::env::var_os("FLARE_TEST_SNAPSHOT").expect("FLARE_TEST_SNAPSHOT is required"),
        );
        let fixture: Value = serde_json::from_slice(
            &fs::read(
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../tests/frontend/qwen38_jpeg_exif.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let asset = |name: &str| Asset {
            path: snapshot.join(name),
            sha256: fixture["assets"][name].as_str().unwrap().into(),
        };
        let frontend = Frontend::load(
            FrontendAssets {
                tokenizer: asset("tokenizer.json"),
                tokenizer_config: asset("tokenizer_config.json"),
                chat_template: asset("chat_template.jinja"),
            },
            defaults,
        )
        .unwrap();
        let processor = ImageProcessor::load(
            asset("preprocessor_config.json"),
            asset("config.json"),
            ImageLimits::default(),
        )
        .unwrap();
        let config = ModelConfig {
            contract: Qwen4Exp::metadata(),
            model_id: "local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD".into(),
            identity: [0; 32],
            limits: SchedulerLimits {
                max_active: 4,
                max_sequence_tokens: 262144,
                prefill_chunk_size: 4096,
                pages: 4096,
                snapshots: 37,
            },
            eos_token_ids: vec![248046, 248044],
            sampling_defaults: SamplingParameters::default(),
            max_new_tokens: 64,
            speculative_widths: vec![],
            speculative_fixed_width: None,
            speculative_copy: false,
            speculative_proposal_source: Default::default(),
        };
        let compiler =
            GrammarCompiler::new(frontend.tokenizer(), 248320, &config.eos_token_ids).unwrap();
        Self {
            frontend,
            processor,
            compiler,
            config,
        }
    }

    fn prepare(&self, messages: Value, options: Value) -> PreparedRequest<Qwen4Exp> {
        let expected = self
            .frontend
            .render_chat(messages.as_array().unwrap(), options.as_object().unwrap())
            .unwrap();
        let request = json!({"model":self.config.model_id,"messages":messages,"chat_template_kwargs":options});
        let prepared = prepare_request::<Qwen4Exp>(
            &self.frontend,
            Some(&self.processor),
            &self.compiler,
            &self.config,
            request,
        )
        .unwrap();
        assert_eq!(prepared.rendered_prompt, expected.text);
        assert_eq!(prepared.prompt.tokens, expected.token_ids);
        Qwen4Exp::validate_prompt(&self.config, &prepared.prompt).unwrap();
        prepared
    }

    fn prefix(&self, prepared: &PreparedRequest<Qwen4Exp>) -> Vec<u32> {
        let boundary = prepared
            .prompt
            .system_boundary
            .expect("instruction boundary");
        assert!(boundary > 0 && boundary < prepared.prompt.tokens.len());
        prepared.prompt.tokens[..boundary].to_vec()
    }
}

fn tool(name: &str) -> Value {
    json!({"type":"function","function":{"name":name,"description":"Look up weather.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}})
}

#[test]
#[ignore = "requires FLARE_TEST_SNAPSHOT pointing to the pinned Qwen checkpoint"]
fn exact_short_instruction_boundary_survives_changed_user_content() {
    let fixture = Fixture::load(Map::new());
    let options = json!({"enable_thinking":false,"add_generation_prompt":true});
    let messages =
        |text| json!([{"role":"system","content":"Be concise."},{"role":"user","content":text}]);
    let a = fixture.prepare(messages("What is two plus two?"), options.clone());
    let b = fixture.prepare(messages("Name a city in France."), options.clone());
    let prefix = fixture.prefix(&a);
    assert_eq!(prefix, fixture.prefix(&b));
    assert!(prefix.len() < 64);
    assert_ne!(prefix.len() % 64, 0);
    let decoded = fixture.frontend.decode(&prefix, false).unwrap();
    assert!(decoded.starts_with("<|im_start|>system\nBe concise.<|im_end|>\n"));
    assert!(!decoded.contains("two plus two"));
    assert!(decoded.contains("<|im_start|>user"));
    let expected = fixture.frontend.render_chat(
        json!([{"role":"system","content":"Be concise."},{"role":"user","content":""}]).as_array().unwrap(),
        json!({"enable_thinking":false,"add_generation_prompt":false,"continue_final_message":true}).as_object().unwrap(),
    ).unwrap();
    assert_eq!(prefix, expected.token_ids);
    let mut malformed = a;
    for boundary in [0, malformed.prompt.tokens.len(), usize::MAX] {
        malformed.prompt.system_boundary = Some(boundary);
        assert!(Qwen4Exp::validate_prompt(&fixture.config, &malformed.prompt).is_err());
    }
    println!("exact short boundary: {} tokens", prefix.len());
}

#[test]
#[ignore = "requires FLARE_TEST_SNAPSHOT pointing to the pinned Qwen checkpoint"]
fn tool_instructions_are_rendered_without_explicit_system_and_disable_with_none() {
    let fixture = Fixture::load(Map::new());
    let messages = json!([{"role":"user","content":"Weather in Oslo?"}]);
    let options =
        |name| json!({"enable_thinking":false,"tools":[tool(name)],"add_generation_prompt":true});
    let a = fixture.prepare(messages.clone(), options("weather"));
    let b = fixture.prepare(
        json!([{"role":"user","content":"Weather in Paris?"}]),
        options("weather"),
    );
    let c = fixture.prepare(messages.clone(), options("other_weather"));
    assert_eq!(fixture.prefix(&a), fixture.prefix(&b));
    assert_ne!(fixture.prefix(&a), fixture.prefix(&c));
    assert!(
        fixture
            .frontend
            .decode(&fixture.prefix(&a), false)
            .unwrap()
            .contains("\"name\": \"weather\"")
    );
    let request = json!({"model":fixture.config.model_id,"messages":messages,"tools":[tool("weather")],"tool_choice":"none","enable_thinking":false});
    let disabled = prepare_request::<Qwen4Exp>(
        &fixture.frontend,
        Some(&fixture.processor),
        &fixture.compiler,
        &fixture.config,
        request,
    )
    .unwrap();
    assert_eq!(disabled.prompt.system_boundary, None);
    assert!(!disabled.rendered_prompt.contains("# Tools"));
}

#[test]
#[ignore = "requires FLARE_TEST_SNAPSHOT pointing to the pinned Qwen checkpoint"]
fn reasoning_defaults_and_explicit_overrides_follow_actual_system_rendering() {
    let messages = json!([{"role":"user","content":"Compute two plus two."}]);
    let default = Fixture::load(Map::new());
    let a = default.prepare(messages.clone(), json!({"add_generation_prompt":true}));
    let prefix = default.frontend.decode(&default.prefix(&a), false).unwrap();
    assert!(prefix.contains("Reasoning effort is set to xhigh."));
    let disabled = Fixture::load(
        json!({"enable_thinking":false})
            .as_object()
            .unwrap()
            .clone(),
    );
    assert_eq!(
        disabled
            .prepare(messages.clone(), json!({}))
            .prompt
            .system_boundary,
        None
    );
    let overridden = disabled.prepare(
        messages.clone(),
        json!({"enable_thinking":true,"reasoning_effort":"low"}),
    );
    assert!(
        disabled
            .frontend
            .decode(&disabled.prefix(&overridden), false)
            .unwrap()
            .contains("Reasoning effort is set to low.")
    );
    assert_eq!(
        default
            .prepare(
                messages,
                json!({"enable_thinking":true,"reasoning_effort":"medium"})
            )
            .prompt
            .system_boundary,
        None
    );
}

#[test]
#[ignore = "requires FLARE_TEST_SNAPSHOT pointing to the pinned Qwen checkpoint"]
fn user_header_and_empty_system_do_not_create_checkpoints() {
    let fixture = Fixture::load(Map::new());
    for messages in [
        json!([{"role":"user","content":"Hello"}]),
        json!([{"role":"system","content":" "},{"role":"user","content":"Hello"}]),
    ] {
        assert_eq!(
            fixture
                .prepare(messages, json!({"enable_thinking":false}))
                .prompt
                .system_boundary,
            None
        );
    }
}

#[test]
#[ignore = "requires FLARE_TEST_SNAPSHOT pointing to the pinned Qwen checkpoint"]
fn token_prefix_mismatch_or_complete_prompt_match_suppresses_boundary() {
    let fixture = Fixture::load(Map::new());
    let messages = json!([{"role":"system","content":"Be concise."},{"role":"user","content":"Explain the sky."}]);
    let options = json!({"enable_thinking":false});
    let a = fixture.prepare(messages.clone(), options.clone());
    let boundary = a.prompt.system_boundary.unwrap();
    let mut mismatch = a.prompt.tokens.clone();
    mismatch[boundary - 1] += 1;
    for tokens in [&mismatch[..], &a.prompt.tokens[..boundary], &[]] {
        assert_eq!(
            Qwen4Exp::system_boundary(
                &fixture.frontend,
                tokens,
                messages.as_array().unwrap(),
                options.as_object().unwrap()
            ),
            None
        );
    }
    let malformed =
        json!([{"role":"system","content":{"not":"valid"}},{"role":"user","content":"Hello"}]);
    assert_eq!(
        Qwen4Exp::system_boundary(
            &fixture.frontend,
            &a.prompt.tokens,
            malformed.as_array().unwrap(),
            options.as_object().unwrap()
        ),
        None
    );
}

#[test]
#[ignore = "requires FLARE_TEST_SNAPSHOT pointing to the pinned Qwen checkpoint"]
fn tool_and_system_prefix_survives_assistant_continuation() {
    let fixture = Fixture::load(Map::new());
    let a = fixture.prepare(
        json!([{"role":"system","content":"Be concise."},{"role":"user","content":"Weather?"}]),
        json!({"tools":[tool("weather")],"enable_thinking":false}),
    );
    let b = fixture.prepare(json!([{"role":"system","content":"Be concise."},{"role":"user","content":"Weather?"},{"role":"assistant","content":"It is"}]), json!({"tools":[tool("weather")],"enable_thinking":false,"continue_final_message":true}));
    assert_eq!(fixture.prefix(&a), fixture.prefix(&b));
    assert!(b.rendered_prompt.ends_with("It is"));
}
