#![cfg(feature = "server")]

use flare_helios::{
    frontend::{Asset, Frontend, FrontendAssets},
    grammar::GrammarCompiler,
    model::{ModelConfig, ModelContract, SamplingParameters, glm5next::Glm5Next},
    output::{Event, Input, ModelOutput, Termination},
    scheduler::SchedulerLimits,
    server::prepare_request,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};

#[test]
#[ignore = "requires FLARE_GLM_TEST_SNAPSHOT and FLARE_GLM_FRONTEND_REFERENCE from generate_glm_fixtures.py"]
fn checkpoint_template_tokenization_requests_and_instruction_prefix_match_transformers() {
    let snapshot = PathBuf::from(std::env::var_os("FLARE_GLM_TEST_SNAPSHOT").unwrap());
    let asset = |name: &str| {
        let path = snapshot.join(name);
        Asset {
            sha256: format!("{:x}", Sha256::digest(fs::read(&path).unwrap())),
            path,
        }
    };
    let frontend = Frontend::load(
        FrontendAssets {
            tokenizer: asset("tokenizer.json"),
            tokenizer_config: asset("tokenizer_config.json"),
            chat_template: asset("chat_template.jinja"),
        },
        json!({"add_generation_prompt":true})
            .as_object()
            .unwrap()
            .clone(),
    )
    .unwrap();
    let config = ModelConfig {
        contract: Glm5Next::metadata(),
        identity: [9; 32],
        model_id: "glm53".into(),
        limits: SchedulerLimits {
            max_active: 4,
            max_sequence_tokens: 1048576,
            prefill_chunk_size: 4096,
            pages: 4096,
            snapshots: 16,
        },
        eos_token_ids: vec![154820, 154827, 154829],
        sampling_defaults: SamplingParameters::default(),
        max_new_tokens: 128,
        speculative_widths: vec![],
        speculative_fixed_width: None,
        speculative_copy: false,
        speculative_proposal_source: Default::default(),
    };
    let compiler =
        GrammarCompiler::new(frontend.tokenizer(), 154880, &config.eos_token_ids).unwrap();
    let reference: Value = serde_json::from_slice(
        &fs::read(std::env::var_os("FLARE_GLM_FRONTEND_REFERENCE").unwrap()).unwrap(),
    )
    .unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let prepared =
            prepare_request::<Glm5Next>(&frontend, None, &compiler, &config, case["body"].clone())
                .unwrap_or_else(|e| panic!("{name}: {e}"));
        flare_helios::model::admission::validate_prompt::<Glm5Next>(&config, &prepared.prompt)
            .unwrap();
        assert_eq!(
            prepared.rendered_prompt,
            case["rendered"].as_str().unwrap(),
            "{name}"
        );
        assert_eq!(
            serde_json::to_value(&prepared.prompt.tokens).unwrap(),
            case["token_ids"],
            "{name}"
        );
        let boundary = prepared
            .prompt
            .system_boundary
            .unwrap_or_else(|| panic!("{name}: missing instruction checkpoint"));
        assert_eq!(
            serde_json::to_value(&prepared.prompt.tokens[..boundary]).unwrap(),
            case["prefix"],
            "{name}"
        );
        assert_eq!(
            prepared.output.reasoning,
            case["reasoning"].as_bool().unwrap(),
            "{name}"
        );
        assert!(!prepared.output.verbatim, "{name}");
        assert!(prepared.prompt.grammar.is_none(), "{name}");
        let reasoning = prepared.output.reasoning;
        let mut output = <Glm5Next as ModelContract>::Output::new(prepared.output);
        let text = if reasoning {
            "analysis</think>result"
        } else {
            "result"
        };
        let mut events = output.push(Input::Text(text)).unwrap();
        events.extend(output.finish(Termination::Eos).unwrap());
        assert_eq!(
            events.last(),
            Some(&Event::Content("result".into())),
            "{name}"
        );
        assert_eq!(
            events.contains(&Event::Reasoning("analysis".into())),
            reasoning,
            "{name}"
        );
        if name == "parallel_tool_results" {
            let text = prepared
                .rendered_prompt
                .split_once("<|assistant|>")
                .unwrap()
                .1
                .split_once("<|observation|>")
                .unwrap()
                .0;
            let tools = case["body"]["chat_template_kwargs"]["tools"]
                .as_array()
                .unwrap()
                .clone();
            let mut output = <Glm5Next as ModelContract>::Output::new(
                flare_helios::model::glm5next::output::OutputConfig {
                    verbatim: false,
                    reasoning: false,
                    tools,
                },
            );
            let mut events = Vec::new();
            for character in text.chars() {
                events.extend(output.push(Input::Text(&character.to_string())).unwrap());
            }
            events.extend(output.finish(Termination::Eos).unwrap());
            let arguments: Vec<Value> = events
                .iter()
                .filter_map(|event| match event {
                    Event::ToolArguments { text, .. } => Some(serde_json::from_str(text).unwrap()),
                    _ => None,
                })
                .collect();
            assert_eq!(
                arguments,
                [
                    json!({"city":"Paris α", "count":2}),
                    json!({"city":"London", "count":1})
                ]
            );
        }
    }
    for request in [
        json!({"model":"glm53", "messages":[{"role":"user", "content":[{"type":"image_url", "image_url":{"url":"data:image/png;base64,AA=="}}]}]}),
        json!({"model":"glm53", "messages":[{"role":"user", "content":"<|begin_of_image|><|image|><|end_of_image|>"}]}),
        json!({"model":"glm53", "messages":[{"role":"user", "content":"JSON please"}], "response_format":{"type":"json_object"}}),
    ] {
        assert!(prepare_request::<Glm5Next>(&frontend, None, &compiler, &config, request).is_err());
    }
}
