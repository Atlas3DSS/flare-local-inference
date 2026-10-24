#!/usr/bin/env python3
"""Independent FP32 Hugging Face reference for the pinned Qwen checkpoint.

Run only inside an authorized compute lease. Decoder parameters are decoded
from raw safetensors on CPU, transferred one layer at a time, then discarded.
The canonical HF forward and DynamicCache own attention and recurrent state.
Equal-length prompts share each streamed layer in a batch, with independent
cache rows and greedy trajectories.
No Flare, b12x, FlashInfer, fused FLA, or quantization-loader code is imported.

Outputs: reference.safetensors contains all prefill logits, the next 15 decode
logits, prompt IDs and 16 greedy IDs per prompt. reference.json records exact
token contexts and provenance. Compare a candidate on these SAME token contexts
(teacher-force the reference continuation); independently generated continuations
cannot be compared position by position after their first differing token.
"""

import argparse
from contextlib import ExitStack
import hashlib
import inspect
import json
import os
from pathlib import Path
import time


MODEL_ID = "nvidia/Qwen3.8-27B-NVFP4"
REVISION = "482ca0f3832238542f8f5295dde86b5f22711d80"
TEXT_PREFIX = "model.language_model"
PROMPTS = (
    "A careful scientist measures the temperature of a sealed container every minute. "
    "The first readings are twenty, twenty two, twenty four, and twenty six degrees. "
    "Explain the pattern and predict the next reading using the available observations.",
    "Write a clear explanation of why the sky often appears blue during the day and "
    "why a sunset can look orange or red. Begin with the way sunlight interacts "
    "with the small particles and molecules in Earth's atmosphere.",
    "A Python function receives a list of integers and returns the sum of the even "
    "numbers without changing the original list. Describe a simple implementation "
    "and explain how it behaves when the input list is empty or contains negative numbers.",
)


def progress(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RawCheckpoint:
    """Memory-map original rows and unswizzled scales, never packed Flare views."""

    def __init__(self, path):
        from safetensors import safe_open

        self.path = path
        self.config = json.loads((path / "config.json").read_text())
        self.index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        self.quantization = self.config["quantization_config"]["quantized_layers"]
        self.stack = ExitStack()
        self.readers = {}
        files = sorted(set(self.index.values()) | {"config.json", "model.safetensors.index.json",
                                                   "tokenizer.json", "tokenizer_config.json"})
        # snapshot_download(local_dir=...) supplies these records. A mismatching
        # or missing pin is an error, rather than silently accepting other weights.
        self.provenance = {}
        for filename in files:
            metadata = path / ".cache" / "huggingface" / "download" / (filename + ".metadata")
            lines = metadata.read_text().splitlines()
            if len(lines) < 2 or lines[0] != REVISION:
                raise ValueError(f"{filename}: checkpoint revision differs from {REVISION}")
            self.provenance[filename] = {"revision": lines[0], "etag": lines[1]}
        for filename in sorted(set(self.index.values())):
            self.readers[filename] = self.stack.enter_context(safe_open(path / filename, framework="pt", device="cpu"))

    def close(self):
        self.stack.close()

    def raw(self, name, rows=None):
        reader = self.readers[self.index[name]]
        return reader.get_tensor(name) if rows is None else reader.get_slice(name)[rows[0]:rows[1]]

    def scalar(self, name):
        import torch

        value = self.raw(name)
        if value.dtype != torch.float32 or value.numel() != 1:
            raise ValueError(f"{name}: expected one FP32 calibration scale")
        result = value.item()
        if not 0 < result < float("inf"):
            raise ValueError(f"{name}: scale must be positive and finite")
        return result

    def materialize(self, name, shape, rows=None):
        """Dequantize ModelOpt checkpoint formats directly into CPU FP32."""
        import torch

        raw = self.raw(name, rows)
        prefix = name.removesuffix(".weight")
        specification = self.quantization.get(prefix) if name.endswith(".weight") else None
        if specification is None:
            result = raw.float()
        elif specification["quant_algo"] == "FP8":
            if raw.dtype != torch.float8_e4m3fn:
                raise ValueError(f"{name}: expected raw E4M3 FP8")
            result = raw.float() * self.scalar(prefix + ".weight_scale")
        elif specification["quant_algo"] == "NVFP4":
            if raw.dtype != torch.uint8 or raw.ndim != 2 or specification.get("group_size") != 16:
                raise ValueError(f"{name}: expected packed E2M1 with group size 16")
            n, half_width = raw.shape
            width = 2 * half_width
            scales = self.raw(prefix + ".weight_scale", rows)
            if scales.dtype != torch.float8_e4m3fn or tuple(scales.shape) != (n, width // 16):
                raise ValueError(f"{name}: expected unswizzled row-major E4M3 block scales")
            # Bit 3 is the sign; bits 2:0 index the positive E2M1 magnitudes.
            # The first column occupies the low nibble of each checkpoint byte.
            magnitudes = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float32)
            table = torch.cat((magnitudes, -magnitudes))
            result = torch.empty((n, width), dtype=torch.float32)
            result[:, 0::2] = table[(raw & 15).long()]
            result[:, 1::2] = table[(raw >> 4).long()]
            result.view(n, width // 16, 16).mul_(scales.float().unsqueeze(-1))
            result.mul_(self.scalar(prefix + ".weight_scale_2"))
        else:
            raise ValueError(f"{prefix}: unsupported reference quantization {specification}")
        if tuple(result.shape) != tuple(shape):
            raise ValueError(f"{name}: decoded shape {tuple(result.shape)} != canonical shape {tuple(shape)}")
        return result


def fp8_input_hook(scale):
    """Preserve checkpoint-calibrated activation quantization, with FP32 GEMM."""
    def hook(module, inputs):
        import torch

        x, *rest = inputs
        quantized = (x.float() / scale).clamp(-448., 448.).to(torch.float8_e4m3fn).float()
        return (quantized * scale, *rest)
    return hook


def build_streaming_model(checkpoint, device):
    import torch
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

    # Select the original HF torch functions even if optional FLA/conv packages
    # happen to be installed. Their source stays canonical and unmodified.
    for name in ("causal_conv1d_fn", "causal_conv1d_update", "torch_chunk_gated_delta_rule",
                 "torch_recurrent_gated_delta_rule"):
        setattr(hf, name, inspect.unwrap(getattr(hf, name)))
    config = Qwen3_5TextConfig.from_dict(checkpoint.config["text_config"])
    config.dtype = torch.float32
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = hf.Qwen3_5TextModel(config)
    model.eval().requires_grad_(False)

    class Embedding(torch.nn.Module):
        def forward(self, ids):
            weight = checkpoint.raw(TEXT_PREFIX + ".embed_tokens.weight")
            selected = torch.nn.functional.embedding(ids.cpu(), weight)
            return selected.to(device=device, dtype=torch.float32)

    model.embed_tokens = Embedding()
    model.rotary_emb = hf.Qwen3_5TextRotaryEmbedding(config=config).to(device)
    model.norm.weight = torch.nn.Parameter(checkpoint.materialize(TEXT_PREFIX + ".norm.weight",
        (config.hidden_size,)).to(device), requires_grad=False)

    def pre_hook(index):
        prefix = f"{TEXT_PREFIX}.layers.{index}"

        def load(layer, inputs):
            for name, parameter in list(layer.named_parameters()):
                tensor = checkpoint.materialize(prefix + "." + name, parameter.shape).to(device)
                parent, _, field = name.rpartition(".")
                setattr(layer.get_submodule(parent), field, torch.nn.Parameter(tensor, requires_grad=False))
            if index in (0, len(model.layers) - 1):
                progress(f"reference layer {index + 1}/{len(model.layers)}")
        return load

    def unload(layer, inputs, output):
        # Preserve the layer's canonical metadata and hooks; release only its
        # parameters. DynamicCache tensors are owned by HF outside this module.
        layer.to_empty(device="meta")

    for index, layer in enumerate(model.layers):
        prefix = f"{TEXT_PREFIX}.layers.{index}"
        for name, module in layer.named_modules():
            specification = checkpoint.quantization.get(prefix + "." + name)
            if specification and specification["quant_algo"] == "FP8":
                module.register_forward_pre_hook(fp8_input_hook(checkpoint.scalar(prefix + "." + name + ".input_scale")))
        layer.register_forward_pre_hook(pre_hook(index))
        layer.register_forward_hook(unload, always_call=True)
    return model, hf


def project_vocabulary(checkpoint, hidden, tile_rows):
    """Keep the full vocabulary matrix off GPU; return unnormalized CPU logits."""
    import torch

    vocab = checkpoint.config["text_config"]["vocab_size"]
    width = hidden.shape[-1]
    flat = hidden.reshape(-1, width)
    logits = torch.empty((flat.shape[0], vocab), dtype=torch.float32)
    for start in range(0, vocab, tile_rows):
        end = min(start + tile_rows, vocab)
        weight = checkpoint.materialize("lm_head.weight", (end - start, width), rows=(start, end)).to(hidden.device)
        logits[:, start:end] = torch.nn.functional.linear(flat, weight).cpu()
        del weight
    return logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="existing output directory")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0", help="Use cuda:0 inside a lease exposing one GPU, or cpu")
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--prompt-batch-size", type=int, choices=range(1, len(PROMPTS) + 1), default=len(PROMPTS))
    parser.add_argument("--head-tile", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=max(1, len(os.sched_getaffinity(0)) - 1))
    args = parser.parse_args()
    if min(args.prompt_tokens, args.new_tokens, args.head_tile, args.threads) <= 0:
        parser.error("token counts, head tile and thread count must be positive")

    # This executable is a reproducible offline reference, including the kernels.
    os.environ["USE_HUB_KERNELS"] = "NO"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    import transformers
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda":
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise RuntimeError("assign CUDA_VISIBLE_DEVICES through a GPU lease before launching")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    checkpoint = RawCheckpoint(args.checkpoint)
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint.path, local_files_only=True)
        model, hf = build_streaming_model(checkpoint, device)
        tensors, prompts = {}, []
        prompt_ids = []
        for index, text in enumerate(PROMPTS):
            prompt = tokenizer.encode(text, add_special_tokens=False)
            if len(prompt) < args.prompt_tokens:
                raise ValueError(f"fixed prompt {index} has only {len(prompt)} tokens")
            prompt_ids.append(prompt[:args.prompt_tokens])
        with torch.inference_mode():
            for start in range(0, len(PROMPTS), args.prompt_batch_size):
                selected = prompt_ids[start:start + args.prompt_batch_size]
                batch = len(selected)
                inputs = torch.tensor(selected, dtype=torch.int64, device=device)
                generated, decode_logits, cache = [[] for _ in selected], [[] for _ in selected], None
                for step in range(args.new_tokens):
                    progress(f"prompts {start + 1}-{start + batch}/{len(PROMPTS)}, "
                             f"greedy step {step + 1}/{args.new_tokens}")
                    result = model(input_ids=inputs, past_key_values=cache, use_cache=True, return_dict=True)
                    cache = result.past_key_values
                    if result.last_hidden_state.dtype != torch.float32:
                        raise RuntimeError("canonical reference computation must stay FP32")
                    logits = project_vocabulary(checkpoint, result.last_hidden_state, args.head_tile).view(
                        batch, inputs.shape[1], model.config.vocab_size)
                    if not bool(torch.isfinite(logits).all()):
                        raise RuntimeError(f"reference produced nonfinite logits at prompt batch {start}, step {step}")
                    tokens = logits[:, -1, :].argmax(dim=-1)
                    for row, token in enumerate(tokens.tolist()):
                        # Each saved tensor owns separate contiguous CPU storage.
                        # This also preserves the original per-prompt file schema.
                        if step == 0:
                            tensors[f"prompt_{start + row}.prefill_logits"] = logits[row].clone()
                        else:
                            decode_logits[row].append(logits[row].clone())
                        generated[row].append(token)
                    inputs = tokens[:, None].to(device)
                    del result
                for row, prompt in enumerate(selected):
                    index = start + row
                    tensors[f"prompt_{index}.decode_logits"] = torch.cat(decode_logits[row]) if decode_logits[row] else torch.empty(
                        (0, model.config.vocab_size), dtype=torch.float32)
                    tensors[f"prompt_{index}.prompt_ids"] = torch.tensor(prompt, dtype=torch.int64)
                    tensors[f"prompt_{index}.generated_ids"] = torch.tensor(generated[row], dtype=torch.int64)
                    prompts.append({"index": index, "source_text": PROMPTS[index], "prompt_text": tokenizer.decode(prompt),
                        "prompt_ids": prompt, "generated_ids": generated[row], "generated_text": tokenizer.decode(generated[row]),
                        "eos_positions": [i for i, token in enumerate(generated[row]) if token == tokenizer.eos_token_id]})
                del cache
        artifact = args.root / "reference.safetensors"
        save_file(tensors, artifact, metadata={"model_id": MODEL_ID, "revision": REVISION,
                  "computation": "HF canonical eager FP32; raw NVFP4 weight-only; calibrated FP8 activations"})
        report = {"model_id": MODEL_ID, "revision": REVISION, "status": "complete", "artifact": str(artifact),
            "torch": torch.__version__, "transformers": transformers.__version__, "device": str(device),
            "config_sha256": sha256(checkpoint.path / "config.json"),
            "index_sha256": sha256(checkpoint.path / "model.safetensors.index.json"),
            "hf_model_source_sha256": sha256(Path(inspect.getsourcefile(hf))), "checkpoint_files": checkpoint.provenance,
            "options": vars(args) | {"root": str(args.root)}, "prompts": prompts, "stop_on_eos": False,
            "elapsed_seconds": time.monotonic() - started,
            "comparison": "Teacher-force prompt_ids followed by generated_ids[:-1] in the candidate. "
                "Compare all prefill_logits and decode_logits at identical positions. Report KLD(reference||candidate), "
                "top-1 agreement, and separate independent greedy continuations; do not silently loosen bounds."}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            report.update(device_name=torch.cuda.get_device_name(device),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        (args.root / "reference.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        progress(f"reference complete: {artifact}")
    finally:
        checkpoint.close()


if __name__ == "__main__":
    main()
