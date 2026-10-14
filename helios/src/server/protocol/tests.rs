use super::*;
use crate::{
    output::{Event, Termination},
    server::inference::{Input, Part, RawPrompt, template_messages},
};
use serde_json::{Value, json};

fn request(protocol: Protocol, body: Value) -> inference::Request {
    decode(protocol, body, false).unwrap().request
}
fn tools() -> Vec<inference::Tool> {
    vec![inference::Tool {
        name: "lookup".into(),
        namespace: None,
        description: "Find a value".into(),
        parameters: json!({"type":"object"}),
        custom: false,
        grammar: None,
    }]
}
fn encoder(protocol: Protocol) -> Encoder {
    Encoder::new(
        protocol,
        "test".into(),
        "model".into(),
        1,
        1,
        17,
        tools(),
        true,
        json!({}),
    )
}
fn events() -> Vec<Event> {
    vec![
        Event::Reasoning("考え".into()),
        Event::Content("Answer ✓".into()),
        Event::ToolBegin {
            index: 0,
            name: "lookup".into(),
            id: Some("call_1".into()),
        },
        Event::ToolArguments {
            index: 0,
            text: "{\"query\":\"".into(),
        },
        Event::ToolArguments {
            index: 0,
            text: "café\"}".into(),
        },
        Event::ToolEnd { index: 0 },
    ]
}

#[test]
fn protocols_decode_to_equivalent_conversations() {
    let chat = request(
        Protocol::Chat,
        json!({"messages":[{"role":"system","content":"Instructions"},{"role":"user","content":"Hello"}],"max_tokens":123,"temperature":0.7}),
    );
    let messages = request(
        Protocol::Messages,
        json!({"system":"Instructions","messages":[{"role":"user","content":[{"type":"text","text":"Hello"}]}],"max_tokens":123,"temperature":0.7}),
    );
    let responses = request(
        Protocol::Responses,
        json!({"instructions":"Instructions","input":"Hello","max_output_tokens":123,"temperature":0.7,"store":false}),
    );
    assert_eq!(chat, messages);
    assert_eq!(chat, responses);
}

#[test]
fn developer_and_system_instructions_lower_to_one_template_preamble() {
    let req = request(
        Protocol::Responses,
        json!({"instructions":"System policy", "input":[
            {"role":"developer","content":"Repository rules"},
            {"role":"user","content":"Do the task"},
            {"role":"developer","content":"Additional rules"}
        ]}),
    );
    let Input::Conversation(messages) = req.input else {
        panic!()
    };
    let native = template_messages(&messages).unwrap();
    assert_eq!(native.len(), 2);
    assert_eq!(
        native[0],
        json!({"role":"system","content":"System instructions:\nSystem policy\n\nDeveloper instructions:\nRepository rules\n\nDeveloper instructions:\nAdditional rules"})
    );
    assert_eq!(native[1]["role"], "user");
}
#[test]
fn response_history_retains_reasoning_calls_results_and_namespaces() {
    let req = request(
        Protocol::Responses,
        json!({"input":[{"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"Think"}]},{"type":"function_call","call_id":"c","name":"run","namespace":"shell","arguments":"{\"x\":2}"},{"type":"function_call_output","call_id":"c","output":[{"type":"input_text","text":"ok"}]},{"role":"user","content":"Continue"}],"tools":[{"type":"namespace","name":"shell","tools":[{"type":"function","name":"run","parameters":{"type":"object"}}]}]}),
    );
    let Input::Conversation(messages) = req.input else {
        panic!()
    };
    assert_eq!(messages.len(), 3);
    assert!(
        matches!(&messages[0].parts[..],[Part::Reasoning(_),Part::ToolCall{name,..}] if name=="shell__run")
    );
    let native = template_messages(&messages).unwrap();
    assert_eq!(native[0]["reasoning_content"], "Think");
    assert_eq!(native[0]["tool_calls"][0]["id"], "c");
    assert_eq!(native[1]["tool_call_id"], "c");
}
#[test]
fn messages_preserve_mixed_tool_results_and_images() {
    let req = request(
        Protocol::Messages,
        json!({"max_tokens":4,"messages":[{"role":"user","content":[{"type":"text","text":"before"},{"type":"tool_result","tool_use_id":"c","is_error":true,"content":[{"type":"text","text":"failed"}]},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA=="}},{"type":"text","text":"after"}]}]}),
    );
    let Input::Conversation(messages) = req.input else {
        panic!()
    };
    let native = template_messages(&messages).unwrap();
    assert_eq!(native.len(), 3);
    assert_eq!(native[0]["content"], "before");
    assert_eq!(native[1]["is_error"], true);
    assert_eq!(
        native[2]["content"][0]["image_url"]["url"],
        "data:image/png;base64,AA=="
    );
}
#[test]
fn all_completion_prompt_forms_and_choice_counts() {
    for (value, prompts) in [
        (json!("x"), 1),
        (json!(["x", "y"]), 2),
        (json!([1, 2]), 1),
        (json!([[1, 2], [3]]), 2),
    ] {
        let req = request(
            Protocol::Completions,
            json!({"prompt":value,"n":3,"max_tokens":0,"echo":true}),
        );
        assert_eq!(req.generation_count().unwrap(), prompts * 3);
        assert_eq!(req.options.output_limit, Some(0));
    }
    let req = request(Protocol::Completions, json!({"prompt":[1,3,2]}));
    assert_eq!(
        req.input,
        Input::Raw(vec![RawPrompt::Tokens(vec![1, 3, 2])])
    );
    for value in [
        json!([]),
        json!([[]]),
        json!([-1]),
        json!([1, "x"]),
        json!(["x", [1]]),
    ] {
        assert!(decode(Protocol::Completions, json!({"prompt":value}), false).is_err());
    }
}
#[test]
fn unsupported_controls_are_rejected_before_generation() {
    for field in [
        json!({"store":true}),
        json!({"previous_response_id":"r"}),
        json!({"conversation":"c"}),
        json!({"background":true}),
        json!({"parallel_tool_calls":false}),
        json!({"tool_choice":"required"}),
        json!({"truncation":"auto"}),
        json!({"tools":[{"type":"function","name":"f","strict":true}]}),
        json!({"tools":[{"type":"web_search"}]}),
        json!({"audio":{"voice":"alloy"}}),
    ] {
        let mut body = json!({"input":"x"});
        body.as_object_mut()
            .unwrap()
            .extend(field.as_object().unwrap().clone());
        assert!(decode(Protocol::Responses, body, false).is_err());
    }
    for field in [
        json!({"suffix":"tail"}),
        json!({"logprobs":0}),
        json!({"best_of":2}),
    ] {
        let mut body = json!({"prompt":"x"});
        body.as_object_mut()
            .unwrap()
            .extend(field.as_object().unwrap().clone());
        assert!(decode(Protocol::Completions, body, false).is_err());
    }
}
#[test]
fn reasoning_mapping_and_explicit_template_overrides() {
    for (budget, effort) in [
        (5000, inference::Effort::Low),
        (5001, inference::Effort::High),
        (20001, inference::Effort::Max),
    ] {
        let req = request(
            Protocol::Messages,
            json!({"max_tokens":30000,"messages":[{"role":"user","content":"x"}],"thinking":{"type":"enabled","budget_tokens":budget}}),
        );
        assert_eq!(req.options.effort, Some(effort));
    }
    let req = request(
        Protocol::Responses,
        json!({"input":"x","reasoning":{"effort":"none"},"chat_template_kwargs":{"enable_thinking":true}}),
    );
    assert_eq!(req.options.template["enable_thinking"], true);
}

#[test]
fn messages_preserve_context_and_custom_function_schemas() {
    let body = json!({"max_tokens":32,"messages":[{"role":"user","content":"x"}],
        "context_management":{"edits":[{"type":"clear_thinking_20251015","keep":"all"}]},
        "tools":[{"type":"custom","name":"lookup","input_schema":{"type":"object","properties":{"query":{"type":"string"}}}}]});
    let req = request(Protocol::Messages, body.clone());
    assert!(!req.tools[0].custom);
    assert_eq!(
        req.tools[0].parameters["properties"]["query"]["type"],
        "string"
    );
    let mut edits = body;
    edits["context_management"]["edits"][0]["keep"] = json!({"type":"thinking_turns","value":1});
    assert!(decode(Protocol::Messages, edits, false).is_err());
}

#[test]
fn custom_tool_input_requires_a_string_before_completion() {
    for args in [r#"{"input":"hello"}"#, r#"{"input":3}"#, "{}"] {
        let mut tools = tools();
        tools[0].custom = true;
        let mut e = Encoder::new(
            Protocol::Responses,
            "test".into(),
            "model".into(),
            1,
            1,
            1,
            tools,
            false,
            json!({}),
        );
        let result = e.apply(
            0,
            vec![
                Event::ToolBegin {
                    index: 0,
                    name: "lookup".into(),
                    id: None,
                },
                Event::ToolArguments {
                    index: 0,
                    text: args.into(),
                },
                Event::ToolEnd { index: 0 },
            ],
        );
        if args.contains("hello") {
            assert!(result.is_ok());
            e.finish(0, Termination::Eos, 5, 0, None).unwrap();
            assert_eq!(e.response()["output"][0]["input"], "hello");
        } else {
            assert!(result.is_err());
            assert_eq!(
                e.failure("invalid tool input").data["type"],
                "response.failed"
            );
        }
    }
}
#[test]
fn tool_name_collisions_and_unknown_custom_formats_are_explicit_errors() {
    for tools in [
        json!([{"type":"function","name":"a__b"},{"type":"namespace","name":"a","tools":[{"type":"function","name":"b"}]}]),
        json!([{"type":"custom","name":"patch","format":{"type":"grammar","syntax":"unknown","definition":"start: /./"}}]),
    ] {
        assert!(
            decode(
                Protocol::Responses,
                json!({"input":"x","tools":tools}),
                false
            )
            .is_err()
        );
    }
}
#[test]
fn messages_stream_reconstructs_ordered_buffered_blocks() {
    let mut encoder = encoder(Protocol::Messages);
    let mut frames = encoder.start();
    frames.extend(encoder.apply(0, events()).unwrap());
    frames.extend(encoder.finish(0, Termination::Eos, 12, 3, None).unwrap());
    let expected = encoder.response();
    assert_eq!(expected["stop_reason"], "tool_use");
    assert_eq!(expected["usage"]["input_tokens"], 17);
    assert_eq!(expected["usage"]["output_tokens"], 12);
    let mut blocks = vec![];
    for frame in &frames {
        let i = frame.data["index"].as_u64().unwrap_or(0) as usize;
        match frame.data["type"].as_str() {
            Some("content_block_start") => blocks.push(frame.data["content_block"].clone()),
            Some("content_block_delta") => {
                let delta = &frame.data["delta"];
                match delta["type"].as_str().unwrap() {
                    "text_delta" => {
                        let s = blocks[i]["text"].as_str().unwrap().to_owned()
                            + delta["text"].as_str().unwrap();
                        blocks[i]["text"] = json!(s)
                    }
                    "thinking_delta" => {
                        let s = blocks[i]["thinking"].as_str().unwrap().to_owned()
                            + delta["thinking"].as_str().unwrap();
                        blocks[i]["thinking"] = json!(s)
                    }
                    "signature_delta" => blocks[i]["signature"] = delta["signature"].clone(),
                    "input_json_delta" => {
                        let s = blocks[i]
                            .get("partial")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_owned()
                            + delta["partial_json"].as_str().unwrap();
                        blocks[i]["partial"] = json!(s)
                    }
                    _ => panic!(),
                }
            }
            Some("content_block_stop") if blocks[i]["type"] == "tool_use" => {
                let partial = blocks[i]
                    .as_object_mut()
                    .unwrap()
                    .remove("partial")
                    .unwrap();
                blocks[i]["input"] = serde_json::from_str(partial.as_str().unwrap()).unwrap();
            }
            _ => {}
        }
    }
    assert_eq!(json!(blocks), expected["content"]);
    assert_eq!(frames.last().unwrap().data["type"], "message_stop");
}
#[test]
fn responses_stream_has_stable_ids_sequence_and_terminal_output() {
    let mut encoder = encoder(Protocol::Responses);
    let mut frames = encoder.start();
    frames.extend(encoder.apply(0, events()).unwrap());
    frames.extend(encoder.finish(0, Termination::Eos, 12, 3, None).unwrap());
    for (i, frame) in frames.iter().enumerate() {
        assert_eq!(frame.data["sequence_number"], i as u64);
    }
    let done: Vec<_> = frames
        .iter()
        .filter(|f| f.data["type"] == "response.output_item.done")
        .map(|f| f.data["item"].clone())
        .collect();
    assert_eq!(json!(done), encoder.response()["output"]);
    assert_eq!(frames.last().unwrap().data["response"], encoder.response());
    assert_eq!(
        encoder.response()["usage"]["input_tokens_details"]["cached_tokens"],
        3
    );
    let mut history = encoder.response()["output"].as_array().unwrap().clone();
    history.push(json!({"type":"function_call_output","call_id":"call_1","output":"found"}));
    assert!(
        decode(
            Protocol::Responses,
            json!({"input":history,"tools":[{"type":"function","name":"lookup"}]}),
            false
        )
        .is_ok()
    );
}
#[test]
fn thinking_markers_round_trip_and_foreign_opaque_history_is_rejected() {
    let mut encoder = encoder(Protocol::Messages);
    encoder
        .apply(0, vec![Event::Reasoning("think".into())])
        .unwrap();
    encoder
        .finish(0, Termination::OutputLimit, 1, 0, None)
        .unwrap();
    let output = encoder.response();
    assert!(decode(Protocol::Messages,json!({"messages":[{"role":"assistant","content":output["content"]},{"role":"user","content":"next"}],"max_tokens":2}),false).is_ok());
    assert!(
        decode(
            Protocol::Responses,
            json!({"input":[{"type":"reasoning","summary":[],"encrypted_content":"opaque"}]}),
            false
        )
        .is_err()
    );
}
#[test]
fn batch_stream_indices_echo_and_totals() {
    let mut encoder = Encoder::new(
        Protocol::Completions,
        "c".into(),
        "m".into(),
        1,
        4,
        7,
        vec![],
        true,
        json!({}),
    );
    for i in [3, 0, 2, 1] {
        let frames = encoder
            .apply(i, vec![Event::Content(format!("choice{i}"))])
            .unwrap();
        assert_eq!(frames[0].data["choices"][0]["index"], i);
        let frames = encoder
            .finish(i, Termination::OutputLimit, 2, 0, None)
            .unwrap();
        if i == 1 {
            assert!(frames.last().unwrap().done);
        } else {
            assert!(!frames.last().unwrap().done);
        }
    }
    let response = encoder.response();
    assert_eq!(
        response["usage"],
        json!({"prompt_tokens":7,"completion_tokens":8,"total_tokens":15})
    );
    for i in 0..4 {
        assert_eq!(response["choices"][i]["text"], format!("choice{i}"));
    }
}
#[test]
fn failures_have_no_success_terminal_and_stops_remain_exact() {
    for protocol in [Protocol::Chat, Protocol::Messages, Protocol::Responses] {
        let mut enc = encoder(protocol);
        enc.apply(0, vec![Event::Content("partial".into())])
            .unwrap();
        let frame = enc.failure("broken");
        assert!(!frame.done);
        if protocol == Protocol::Responses {
            assert_eq!(frame.data["type"], "response.failed");
            assert_eq!(frame.data["response"]["status"], "failed");
        }
    }
    let mut enc = encoder(Protocol::Messages);
    enc.finish(0, Termination::UserStop, 3, 0, Some("世界".into()))
        .unwrap();
    assert_eq!(enc.response()["stop_sequence"], "世界");
    assert_eq!(enc.response()["stop_reason"], "stop_sequence");
}

#[test]
fn output_limit_marks_only_the_open_response_item_incomplete() {
    let mut e = encoder(Protocol::Responses);
    let mut frames = e
        .apply(
            0,
            vec![
                Event::Reasoning("think".into()),
                Event::Content("partial".into()),
            ],
        )
        .unwrap();
    frames.extend(e.finish(0, Termination::OutputLimit, 4, 0, None).unwrap());
    let response = e.response();
    assert_eq!(response["status"], "incomplete");
    assert_eq!(response["output"][0]["status"], "completed");
    assert_eq!(response["output"][1]["status"], "incomplete");
    let done: Vec<_> = frames
        .iter()
        .filter(|f| f.data["type"] == "response.output_item.done")
        .map(|f| f.data["item"].clone())
        .collect();
    assert_eq!(json!(done), response["output"]);
}

#[test]
fn chat_history_without_call_ids_retains_template_semantics() {
    let req = request(
        Protocol::Chat,
        json!({"messages":[
            {"role":"user","content":"Find it"},
            {"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"lookup","arguments":"{}"}}]},
            {"role":"tool","content":"found"}
        ]}),
    );
    let Input::Conversation(messages) = req.input else {
        panic!()
    };
    let native = template_messages(&messages).unwrap();
    assert_eq!(native.len(), 3);
    assert!(native[1]["tool_calls"][0].get("id").is_none());
    assert!(native[2].get("tool_call_id").is_none());
}

#[test]
fn empty_text_messages_preserve_conversation_roles() {
    let req = request(
        Protocol::Chat,
        json!({"messages":[
            {"role":"system","content":""}, {"role":"user","content":""},
            {"role":"assistant","content":null}
        ]}),
    );
    let Input::Conversation(messages) = req.input else {
        panic!()
    };
    assert_eq!(
        template_messages(&messages).unwrap(),
        vec![
            json!({"role":"system","content":""}),
            json!({"role":"user","content":""}),
            json!({"role":"assistant","content":""})
        ]
    );
}
