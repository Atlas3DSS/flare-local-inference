# Native steering on a V100

This repository contains the reusable SM70 runtime. The linked [Character Creation](https://github.com/Atlas3DSS/Character-Creation) repository contains the persona explorer, prompt banks, calibration, experiment planning and analysis. You can use the runtime independently through its HTTP API; Atlas’s private host configuration and gpu-workboard are not required by the native server.

The current implementation targets the existing `qwen38_dense` SM70 recipe for `nvidia/Qwen3.8-27B-NVFP4`, revision `482ca0f3832238542f8f5295dde86b5f22711d80`. Follow the existing [V100 build instructions](docs/sm70/BUILD.md), including CUDA 12.9, the pinned Python dependencies and liburing development support. An SSD/NVMe checkpoint and package cache is strongly recommended: the loaders use direct I/O, so a warm operating-system page cache does not eliminate repeated HDD reads.

## Status

The implementation builds and its native CPU tests pass. The compiled/captured steering component has passed GPU checks on six V100s, including signed vectors, zero/off bit equality, phase boundaries and physical request-slot isolation. Character Creation completed 132 whole-model discovery captures on all six cards; its [calibration report](https://github.com/Atlas3DSS/Character-Creation/blob/main/reports/persona_calibration.json) records signed tests at one interior layer and zero checks across the catalog. A retained-graph worker passed prefill and ordinary-generation smoke checks. Production export gates, native serving/cache ownership, speculative execution and behavioral persona validation require separate results. Do not treat component or calibration checks as proof of a useful persona, or assume the original throughput benchmarks include steering overhead.

## Three distinct sites

| Site | Width | Operation |
|---|---:|---|
| `layers.L.mlp_input` | 17,408 | Add to the rounded SwiGLU product, round to BF16 values, then run the quantized `down_proj` |
| `layers.L.mlp_out` | 5,120 | Add after the down projection, before its residual addition |
| `layers.L.residual` | 5,120 | Add after the block residual addition |

The first site exposes the actual intermediate neuron input. Preparing steering keeps that product explicit instead of fusing it into the projection producer. Default preparation with steering and observations disabled retains the existing fused path. A post-down output-column addition is a useful comparison but is not numerically interchangeable with a pre-down neuron edit under quantization.

Following **[Lyra’s neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer)**, a model-calibrated scalar addition for neuron `n` is:

```
dose × median(norm(residual leaving layer L), tokens[1:]) / norm(W_down[:, n])
```

The HTTP server accepts the resulting vector, not a guessed neuron dose. Character Creation’s adapter extracts decoded columns, measures fresh residual medians and constructs the vector. Calibration and coordinates must belong to the exact checkpoint and quantization being served. Lyra’s supplied Qwen3.5-4B neuron recipes are reference experiments, not discovered coordinates for this 27B model.

## Prepare and discover support

Rebuild both the native CLI and Heliopause after changing the runtime protocol. A stale Python extension can import successfully while rejecting the new package schema. The existing [preparation harness](tools/prepare_replay.py) accepts `--steering-max-sites 64`; add it to the normal fully qualified preparation/export command. It retains the existing strict, row-invariance, recurrent-commit, consumer, component, accuracy and coding/token gates. Keep signing keys outside the source tree.

In Python, the equivalent recipe option is `PreparationConfig(steering_max_sites=64, ...)`. Zero remains the default and creates no steering catalog. Previously prepared packages still load, but reject nonempty steering requests. All site widths and the request limit are advertised at `GET /v1/models` under `data[0].flare.steering`, alongside `flare.prepared_identity`. Clients must verify that identity against the package they intended to load.

The preparation script also supports an explicit skipped-reference-gates mode described in its own help. That does not turn missing accuracy measurements into a pass. The Character Creation release wrapper requires all reference gates and compares neutral logits with its pinned unsteered baseline before export.

## Request example

Send the ordinary chat request with this extension. This Python example targets one neuron with a previously calibrated scalar `alpha`:

```python
import requests

server = "http://127.0.0.1:8000"
descriptor = requests.get(server + "/v1/models").json()["data"][0]
capability = descriptor.get("flare", {})
steering = capability.get("steering", {})
# expected_prepared_identity comes from the package you qualified locally.
assert capability.get("prepared_identity") == expected_prepared_identity
assert steering.get("protocol") == "additive-bf16-sites-v1"
assert steering.get("max_sites", 0) >= 1
assert {s["name"]: s["width"] for s in steering.get("sites", [])}.get("layers.14.mlp_input") == 17408
vector = [0.0] * 17408
vector[8949] = alpha  # measured on this checkpoint; not copied from another model
response = requests.post(server + "/v1/chat/completions", json={
    "model": descriptor["id"],
    "messages": [{"role": "user", "content": "I got promoted today. What do you think?"}],
    "seed": 7035,
    "max_tokens": 2048,
    "chat_template_kwargs": {"enable_thinking": False},
    "steering": {
        "phase": "all",
        "sites": [{"site": "layers.14.mlp_input", "vector": vector}]
    }
})
response.raise_for_status()
print(response.json())
```

The numeric coordinates above illustrate the API only; no effect on 27B is asserted. Each site must be unique and contain exactly its advertised number of finite FP32 values. Composite edits at one site must be summed by the client. The server does not silently clip or rescale vectors.

Phases are `all`, `prefill` and `decode`, determined by absolute position relative to the original prompt length. This boundary is independent of chunk scheduling and cache hits. The first generated token is sampled from the final prompt position, so its intervention belongs to `prefill`. `decode` does not mean “answer only” when a model emits a thinking section.

Programs remain immutable for a request. Their canonical hash participates in prefix-cache conditioning. Ragged per-slot buffers upload at initialization/restoration, not every decode step. Packed tokens resolve to their physical request slot and generation; padded and stale rows cannot borrow another request’s push. Zero coordinates preserve the original bits. MTP targets receive the intervention; the draft remains unchanged.

## Offline observations

`PreparationConfig(capture_activations=True, retain_graphs=True, ...)` exposes shared diagnostic buffers on ordinary prefill owners at `owner.decoder.activation_capture`:

- `residual_last`: final valid token’s output from each layer.
- `pre_mlp_last`: final valid token’s normalized MLP input.
- `mlp_input_last`: actual rounded SwiGLU product after any neuron edit and before down projection.
- `residual_l2`: per-layer norms for the current prefill chunk.
- `positions`: valid absolute token positions, with padding marked `-1`.

Read/copy these buffers after each replay and before the next one overwrites them. A generation with steering enabled observes the steered activations; collect calibration with all steering disabled. Diagnostic capture is not the serving package and cannot be combined with the harness’s export/measurement mode. The native serving loop contains no Python hooks or callbacks.

## Attribution

- **NVIDIA and the Flare community:** original runtime, retained Apache-2.0 [LICENSE](LICENSE) and [NOTICE](NOTICE), and the preparation/serving architecture this port extends.
- **Atlas3DSS:** V100 port, runtime integration and Character Creation’s broad trait/layer experiments.
- **Lyra (@_lyraaaa_):** primary methodological inspiration for fine-grained neuron interventions, signed relative-dose normalization, read/write distinctions and comparative rollouts. Reviewed [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer) revision `f79b2196e29aacbad89684938baac8d97b172fd1`. This is attribution, not a claim of collaboration or endorsement; Lyra’s code is not vendored.
- **RhizoNymph/vLLM:** [request steering design](https://github.com/RhizoNymph/vllm/blob/feat/integration/docs/design/steering_runtime.md), a reference for request ownership, graph-safe staging and cache identity.
- **heterodoxin / AbliteratedOutTheBox:** [Strata branch](https://github.com/heterodoxin/AbliteratedOutTheBox/tree/strata), a reference for native residual interventions; its experimental results are not substituted for validation here.
