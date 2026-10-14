//! Model discovery uses the same effective configuration as request admission.

use crate::model::ModelConfig;
use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};
use serde_json::{Value, json};
use std::{
    collections::HashSet,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

struct ModelCatalog {
    card: Value,
    names: HashSet<String>,
}

impl ModelCatalog {
    fn new(model: &ModelConfig, served_names: &[String]) -> Self {
        // The timestamp describes this serving instance, not the source checkpoint.
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let id = served_names
            .iter()
            .find(|name| !name.is_empty())
            .unwrap_or(&model.model_id);
        let names = std::iter::once(&model.model_id)
            .chain(served_names)
            .filter(|name| !name.is_empty())
            .cloned()
            .collect();
        // One card represents the loaded model; its other names only resolve lookups.
        // Match vLLM's ModelCard / ModelPermission wire fields while
        // advertising only features supported by this server.
        let card = json!({
            "id": id,
            "object": "model",
            "created": created,
            "owned_by": "local",
            "root": model.model_id,
            "parent": null,
            "max_model_len": model.limits.max_sequence_tokens,
            "permission": [{
                "id": format!("modelperm-{created}-0"),
                "object": "model_permission",
                "created": created,
                "allow_create_engine": false,
                "allow_sampling": true,
                "allow_logprobs": false,
                "allow_search_indices": false,
                "allow_view": true,
                "allow_fine_tuning": false,
                "organization": "*",
                "group": null,
                "is_blocking": false,
            }],
            "flare": {
                "vocab_size": model.contract.vocabulary,
                "max_concurrency": model.limits.max_active,
                "prefill_chunk_size": model.limits.prefill_chunk_size,
                "default_max_tokens": model.max_new_tokens,
                "sampling_defaults": model.sampling_defaults,
                "max_images_per_prompt": model.contract.vision.max_images,
                "input_modalities": if model.contract.vision.enabled() {
                    vec!["text", "image"]
                } else {
                    vec!["text"]
                },
                "output_modalities": ["text"],
            },
        });
        Self { card, names }
    }
}

pub(super) fn router<S>(model: &ModelConfig, served_names: &[String]) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new()
        .route("/v1/models", get(list))
        // A wildcard also accepts repository IDs such as organization/model.
        .route("/v1/models/{*model}", get(retrieve))
        .with_state(Arc::new(ModelCatalog::new(model, served_names)))
}

async fn list(State(catalog): State<Arc<ModelCatalog>>) -> Json<Value> {
    Json(json!({"object": "list", "data": [&catalog.card]}))
}

async fn retrieve(State(catalog): State<Arc<ModelCatalog>>, Path(model): Path<String>) -> Response {
    if catalog.names.contains(&model) {
        Json(catalog.card.clone()).into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(json!({"error": {
                "message": format!("The model `{model}` does not exist."),
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found",
            }})),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model::{ContractMetadata, SamplingParameters, VisionCapabilities},
        scheduler::SchedulerLimits,
    };
    use std::{
        io::{Read, Write},
        net::{SocketAddr, TcpStream},
        time::Duration,
    };

    fn config() -> ModelConfig {
        ModelConfig {
            prepared_sequence_tokens: 16384,
            contract: ContractMetadata {
                schema: "test".into(),
                vocabulary: 1024,
                vision: VisionCapabilities::disabled(),
                page_tokens: None,
            },
            identity: [0; 32],
            model_id: "organization/model".into(),
            limits: SchedulerLimits {
                request_slots: 4,
                max_active: 3,
                max_sequence_tokens: 8192,
                prefill_chunk_size: 512,
                pages: 64,
                snapshots: 8,
            },
            eos_token_ids: vec![2],
            sampling_defaults: SamplingParameters {
                temperature: 0.75,
                top_p: 1.,
                top_k: -1,
                ..Default::default()
            },
            max_new_tokens: 2048,
            speculative_widths: vec![],
            speculative_fixed_width: None,
            history_drafts: false,
            speculative_proposal_source: Default::default(),
            state_rows_per_request: None,
        }
    }

    #[test]
    fn metadata_reports_effective_limits_and_prepared_modalities() {
        let mut model = config();
        let text = ModelCatalog::new(&model, &[]).card;
        assert_eq!(text["max_model_len"], 8192);
        assert_eq!(text["flare"]["max_concurrency"], 3);
        assert_eq!(text["flare"]["prefill_chunk_size"], 512);
        assert_eq!(text["flare"]["vocab_size"], 1024);
        assert_eq!(text["flare"]["default_max_tokens"], 2048);
        assert_eq!(
            text["flare"]["sampling_defaults"],
            json!(model.sampling_defaults)
        );
        assert_eq!(text["flare"]["input_modalities"], json!(["text"]));
        assert_eq!(text["flare"]["output_modalities"], json!(["text"]));
        assert_eq!(text["flare"]["max_images_per_prompt"], 0);
        assert_eq!(text["permission"][0]["allow_logprobs"], false);
        assert_eq!(text["permission"][0]["allow_sampling"], true);
        assert_eq!(text["permission"][0]["allow_fine_tuning"], false);
        assert!(text["created"].as_u64().unwrap() > 0);
        assert_eq!(text["permission"][0]["created"], text["created"]);

        model.limits.max_sequence_tokens = 4096;
        model.contract.vision = VisionCapabilities {
            max_images: 4,
            features_per_request: 256,
            patches_per_feature: 4,
            patch_capacities: vec![1024],
        };
        let vision = ModelCatalog::new(&model, &[]).card;
        assert_eq!(vision["max_model_len"], 4096);
        assert_eq!(
            vision["flare"]["input_modalities"],
            json!(["text", "image"])
        );
        assert_eq!(vision["flare"]["max_images_per_prompt"], 4);
    }

    #[test]
    fn discovery_uses_the_first_nonempty_alias_or_checkpoint_id() {
        for (aliases, expected) in [
            (vec![], "organization/model"),
            (vec!["", ""], "organization/model"),
            (vec!["", "alias", "alias", "other"], "alias"),
            (vec!["organization/model", "alias"], "organization/model"),
        ] {
            let aliases = aliases.into_iter().map(String::from).collect::<Vec<_>>();
            let catalog = ModelCatalog::new(&config(), &aliases);
            assert_eq!(catalog.card["id"], expected);
            assert_eq!(catalog.card["root"], "organization/model");
            assert!(catalog.names.contains("organization/model"));
            assert!(!catalog.names.contains(""));
            assert!(
                aliases
                    .iter()
                    .filter(|name| !name.is_empty())
                    .all(|name| catalog.names.contains(name))
            );
        }
    }

    fn get(address: SocketAddr, path: &str) -> (u16, Value) {
        let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(5)).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        write!(
            stream,
            "GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
        )
        .unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        let (headers, body) = response.split_once("\r\n\r\n").unwrap();
        assert!(headers.contains("application/json"));
        let status = headers.split_whitespace().nth(1).unwrap().parse().unwrap();
        (status, serde_json::from_str(body).unwrap())
    }

    #[tokio::test]
    async fn http_discovery_and_lookup_share_cards_and_handle_repository_ids() {
        let aliases = [
            "alias",
            "organization/model",
            "alias",
            "",
            "alias with space",
        ]
        .map(String::from);
        let router = router::<()>(&config(), &aliases);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let serving = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        let result = tokio::task::spawn_blocking(move || {
            let (status, list) = get(address, "/v1/models");
            assert_eq!(status, 200);
            assert_eq!(list["object"], "list");
            let cards = list["data"].as_array().unwrap();
            assert_eq!(cards.len(), 1);
            let card = &cards[0];
            assert_eq!(card["id"], "alias");
            assert_eq!(card["object"], "model");
            assert_eq!(card["owned_by"], "local");
            assert_eq!(card["root"], "organization/model");
            assert_eq!(card.get("parent"), Some(&Value::Null));
            assert_eq!(card["max_model_len"], 8192);
            assert_eq!(card["permission"][0]["object"], "model_permission");
            for path in [
                "/v1/models/organization/model",
                "/v1/models/organization%2Fmodel",
                "/v1/models/alias",
                "/v1/models/alias%20with%20space",
            ] {
                let (status, retrieved) = get(address, path);
                assert_eq!(status, 200, "{path}");
                assert_eq!(&retrieved, card);
            }
            assert_eq!(get(address, "/v1/models").1, list);
            for path in ["/v1/models/unknown", "/v1/models/organization/model/extra"] {
                let (status, error) = get(address, path);
                assert_eq!(status, 404);
                assert_eq!(error["error"]["type"], "invalid_request_error");
                assert_eq!(error["error"]["param"], "model");
                assert_eq!(error["error"]["code"], "model_not_found");
            }
        })
        .await;
        serving.abort();
        result.unwrap();
    }
}
