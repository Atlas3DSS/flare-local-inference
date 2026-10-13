//! Verify frozen generation-corpus prompts through native HTTP request preparation.
use flare_helios::model::qwen4exp::Qwen4Exp;

use flare_helios::{
    frontend::{
        Asset, Frontend, FrontendAssets,
        images::{ImageLimits, ImageProcessor},
    },
    grammar::GrammarCompiler,
    model::{ModelConfig, SamplingParameters, qwen4exp::VOCABULARY},
    scheduler::SchedulerLimits,
    server::prepare_request,
};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};

fn main() -> Result<(), String> {
    run().map_err(|error| error.to_string())
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args_os().collect();
    if arguments.len() != 3 {
        return Err("usage: test_generation_corpus ABSOLUTE_SNAPSHOT CORPUS_JSON".into());
    }
    let snapshot = PathBuf::from(&arguments[1]);
    let bytes = fs::read(&arguments[2]).map_err(|error| error.to_string())?;
    let corpus: Value = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    if corpus["schema_version"] != 1
        || corpus["max_new_tokens"] != 64
        || corpus["ignore_eos"] != true
    {
        return Err("expected the frozen schema-v1 greedy64 corpus".into());
    }
    let asset = |name: &str| -> Result<Asset, String> {
        Ok(Asset {
            range: None,
            path: snapshot.join(name),
            sha256: corpus["tokenizer_assets"][name]["sha256"]
                .as_str()
                .ok_or_else(|| format!("missing corpus asset fingerprint {name}"))?
                .into(),
        })
    };
    let mut defaults = Map::new();
    defaults.insert("add_generation_prompt".into(), Value::Bool(true));
    let frontend = Frontend::load(
        FrontendAssets {
            tokenizer: asset("tokenizer.json")?,
            tokenizer_config: asset("tokenizer_config.json")?,
            chat_template: asset("chat_template.jinja")?,
        },
        defaults,
    )?;
    let images = ImageProcessor::load(
        asset("preprocessor_config.json")?,
        asset("config.json")?,
        ImageLimits::default(),
    )?;
    let model = ModelConfig {
        contract: Qwen4Exp::metadata(),
        model_id: corpus["model_id"]
            .as_str()
            .ok_or("missing corpus model_id")?
            .into(),
        identity: [0; 32],
        limits: SchedulerLimits {
            request_slots: 4,
            max_active: 4,
            max_sequence_tokens: 262144,
            prefill_chunk_size: 4096,
            pages: 4096,
            snapshots: 37,
        },
        eos_token_ids: vec![248046, 248044],
        sampling_defaults: SamplingParameters::default(),
        max_new_tokens: 256,
        speculative_widths: Vec::new(),
        speculative_fixed_width: None,
        history_drafts: true,
        speculative_proposal_source: Default::default(),
        state_rows_per_request: None,
    };
    let compiler = GrammarCompiler::new(frontend.tokenizer(), VOCABULARY, &model.eos_token_ids)?;
    let mut results = Vec::new();
    for case in corpus["prompts"]
        .as_array()
        .ok_or("missing corpus prompts")?
    {
        let id = case["id"].as_str().ok_or("missing corpus case ID")?;
        let mut request = case["request"].clone();
        request
            .as_object_mut()
            .ok_or("invalid corpus request")?
            .insert("model".into(), Value::String(model.model_id.clone()));
        let prepared =
            prepare_request::<Qwen4Exp>(&frontend, Some(&images), &compiler, &model, request)
                .map_err(|error| format!("{id}: {error}"))?;
        let expected: Vec<u32> = serde_json::from_value(case["prompt_tokens"].clone())
            .map_err(|error| format!("{id}: {error}"))?;
        if prepared.prompt.tokens != expected
            || Some(prepared.rendered_prompt.as_str()) != case["rendered_prompt"].as_str()
        {
            return Err((format!("{id}: actual native prompt bytes or token IDs differ")).into());
        }
        if prepared.prompt.max_new_tokens != 64
            || prepared.prompt.sampling.temperature != 0.
            || prepared.prompt.sampling.repetition_penalty != 1.
            || prepared.prompt.sampling.presence_penalty != 0.
            || prepared.prompt.sampling.frequency_penalty != 0.
            || prepared.stream
            || prepared.prompt.grammar.is_some()
            || prepared.prompt.images.is_some()
            || !prepared.prompt.stop_sequences.is_empty()
        {
            return Err(
                (format!("{id}: native request violates unconstrained greedy64 contract")).into(),
            );
        }
        let token_bytes: Vec<_> = prepared
            .prompt
            .tokens
            .iter()
            .flat_map(|token| token.to_le_bytes())
            .collect();
        results.push(json!({"id":id,"prompt_tokens":expected.len(),
            "prompt_tokens_sha256":format!("{:x}",Sha256::digest(token_bytes)),
            "rendered_prompt_sha256":format!("{:x}",Sha256::digest(prepared.rendered_prompt.as_bytes())),
            "passed":true}));
    }
    println!(
        "{}",
        json!({"passed":true,"cases":results.len(),"results":results,
        "corpus_sha256":format!("{:x}",Sha256::digest(bytes)),"tokenizer_assets":corpus["tokenizer_assets"]})
    );
    Ok(())
}
