#!/usr/bin/env python3
"""GPU-resident FP32 references, reusing reference_model.py arithmetic unchanged.

Only weight *placement* differs from reference_model.py: the raw checkpoint rows (packed
E2M1 bytes, E4M3 block scales, E4M3 FP8 weights, BF16 tensors) are copied to the GPU once, and
RawCheckpoint.materialize runs the identical decode ops there (same table, same multiply order,
FP32 throughout).  Every op in that decode is exact or a single IEEE FP32 multiply, so the decoded
weights are bit-identical to the CPU decode; --self-check asserts this on sample tensors.  The
canonical HF forward, DynamicCache, FP8 activation hooks and vocabulary projection are reference_model.py's
own functions (imported, not copied).

Modes
  repro  run reference_model.py's main() unmodified (its prompts, batch, greedy loop, output schema) with the
         GPU-resident checkpoint; writes <root>/reference.{safetensors,json}.
  long   six longest non-thinking code prompts of mtp-accept/prompts.jsonl rendered with the
         checkpoint chat template (enable_thinking=False, add_generation_prompt=True, as the
         llama.cpp --jinja acceptance test did), 32-token greedy continuation (cached decode),
         then ONE teacher-forced forward of prompt+continuation without cache; saves full-vocab
         logits at the last 48 positions.  Also writes main-model final (post-norm) hidden states
         and per-position argmax for MTP calibration/evaluation (Job B), for these prompts and the
         remaining non-thinking prompts (prompt-only forwards).
"""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

import reference_model as ref  # noqa: E402

CPU_RAW_CHECKPOINT = ref.RawCheckpoint  # reference_model.py's original class, kept for the self-check



def progress(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


class GpuRawCheckpoint(CPU_RAW_CHECKPOINT):
    """reference_model.py's RawCheckpoint with decoder/head rows resident on the GPU."""

    def __init__(self, path, device=None):
        import torch

        super().__init__(path)
        self.device = torch.device(device or "cuda:0")
        self.gpu, self.cpu_cache = {}, {}
        keep = (ref.TEXT_PREFIX + ".layers.", ref.TEXT_PREFIX + ".norm.", "lm_head.")
        started, total = time.monotonic(), 0
        for name, filename in sorted(self.index.items()):
            if not name.startswith(keep):
                continue
            tensor = self.readers[filename].get_tensor(name)
            if tensor.numel() == 1 and tensor.dtype == torch.float32:
                continue  # calibration scalars stay on CPU for scalar()/.item()
            self.gpu[name] = tensor.to(self.device)
            total += tensor.numel() * tensor.element_size()
        progress(f"resident raw tensors: {len(self.gpu)}, {total / 2**30:.2f} GiB in {time.monotonic() - started:.0f}s")

    def raw(self, name, rows=None):
        tensor = self.gpu.get(name)
        if tensor is not None:
            return tensor if rows is None else tensor[rows[0]:rows[1]]
        if rows is None:
            # Embedding table: read once, keep the same CPU tensor (values identical to get_tensor).
            if name not in self.cpu_cache:
                self.cpu_cache[name] = super().raw(name)
            return self.cpu_cache[name]
        return super().raw(name, rows)

    def materialize(self, name, shape, rows=None):
        """reference_model.py's materialize() verbatim, with temporaries created on the raw tensor's device."""
        import torch

        raw = self.raw(name, rows)
        device = raw.device
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
            magnitudes = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float32, device=device)
            table = torch.cat((magnitudes, -magnitudes))
            result = torch.empty((n, width), dtype=torch.float32, device=device)
            result[:, 0::2] = table[(raw & 15).long()]
            result[:, 1::2] = table[(raw >> 4).long()]
            result.view(n, width // 16, 16).mul_(scales.float().unsqueeze(-1))
            result.mul_(self.scalar(prefix + ".weight_scale_2"))
        else:
            raise ValueError(f"{prefix}: unsupported reference quantization {specification}")
        if tuple(result.shape) != tuple(shape):
            raise ValueError(f"{name}: decoded shape {tuple(result.shape)} != canonical shape {tuple(shape)}")
        return result


def self_check(checkpoint):
    """GPU decode must equal reference_model.py's CPU decode bit for bit."""
    import torch

    cpu = CPU_RAW_CHECKPOINT(checkpoint.path)
    try:
        P = ref.TEXT_PREFIX
        cases = [(f"{P}.layers.0.mlp.down_proj.weight", (5120, 17408), None),
                 (f"{P}.layers.0.linear_attn.in_proj_qkv.weight", (10240, 5120), None),
                 (f"{P}.layers.3.self_attn.q_norm.weight", (256,), None),
                 (f"{P}.layers.63.mlp.gate_proj.weight", (17408, 5120), None),
                 ("lm_head.weight", (2048, 5120), (0, 2048)),
                 ("lm_head.weight", (2048, 5120), (246272, 248320))]
        for name, shape, rows in cases:
            a = cpu.materialize(name, shape, rows)
            b = checkpoint.materialize(name, shape, rows).cpu()
            if not torch.equal(a.view(torch.int32), b.view(torch.int32)):
                raise RuntimeError(f"GPU decode differs from CPU decode for {name} rows={rows}")
        progress(f"self-check: GPU decode bit-identical to CPU decode on {len(cases)} tensors")
        return [c[0] + ("" if c[2] is None else f"[{c[2][0]}:{c[2][1]}]") for c in cases]
    finally:
        cpu.close()


def head_argmax(checkpoint, hidden2d, tile):
    """Per-row argmax and max of hidden @ lm_head^T without materializing all logits."""
    import torch

    vocab = checkpoint.config["text_config"]["vocab_size"]
    width = hidden2d.shape[-1]
    best = torch.full((hidden2d.shape[0],), -float("inf"), device=hidden2d.device)
    index = torch.zeros((hidden2d.shape[0],), dtype=torch.int64, device=hidden2d.device)
    for start in range(0, vocab, tile):
        end = min(start + tile, vocab)
        weight = checkpoint.materialize("lm_head.weight", (end - start, width), rows=(start, end))
        value, position = torch.nn.functional.linear(hidden2d, weight).max(dim=-1)
        better = value > best
        best = torch.where(better, value, best)
        index = torch.where(better, position + start, index)
        del weight
    return index.cpu(), best.cpu()


def render(tokenizer, row):
    kwargs = dict(add_generation_prompt=True, enable_thinking=row["thinking"], tokenize=True, return_dict=False)
    if row.get("tools"):
        kwargs["tools"] = row["tools"]
    ids = tokenizer.apply_chat_template(row["messages"], **kwargs)
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    return [int(i) for i in ids]


def kl_rows(p_logits, q_logits):
    """KL(p||q) per row from logits, FP64."""
    import torch

    lp = torch.log_softmax(p_logits.double(), -1)
    lq = torch.log_softmax(q_logits.double(), -1)
    return (lp.exp() * (lp - lq)).sum(-1)


def run_long(args):
    import torch
    import transformers
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    calib_dir = args.calib_out
    calib_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = GpuRawCheckpoint(args.checkpoint, device)
    checked = self_check(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.path, local_files_only=True)
    model, hf = ref.build_streaming_model(checkpoint, device)
    rows = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]
    rendered = {row["id"]: render(tokenizer, row) for row in rows}
    code = [row for row in rows if row["id"].startswith("code") and not row["thinking"]]
    selected = sorted(code, key=lambda r: (-len(rendered[r["id"]]), r["id"]))[:args.count]
    selected_ids = {r["id"] for r in selected}
    calibration = [row for row in rows if not row["thinking"] and row["id"] not in selected_ids]
    stop_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    tensors, records = {}, []
    with torch.inference_mode():
        for order, row in enumerate(selected):
            prompt = rendered[row["id"]]
            prompt_len = len(prompt)
            # 1) greedy continuation with reference_model.py's cached-decode loop shape (batch 1)
            inputs = torch.tensor([prompt], dtype=torch.int64, device=device)
            cache, generated, decode_logits = None, [], []
            t0 = time.monotonic()
            for step in range(args.new_tokens):
                result = model(input_ids=inputs, past_key_values=cache, use_cache=True, return_dict=True)
                cache = result.past_key_values
                logits = ref.project_vocabulary(checkpoint, result.last_hidden_state[:, -1:, :], args.head_tile)
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError(f"nonfinite logits, {row['id']} step {step}")
                token = int(logits[0].argmax())
                generated.append(token)
                decode_logits.append(logits[0].clone())
                inputs = torch.tensor([[token]], dtype=torch.int64, device=device)
                del result
            del cache
            generation_s = time.monotonic() - t0
            # 2) one teacher-forced forward of prompt + continuation, no cache
            t0 = time.monotonic()
            full = prompt + generated
            result = model(input_ids=torch.tensor([full], dtype=torch.int64, device=device), use_cache=False,
                           return_dict=True)
            hidden = result.last_hidden_state
            if hidden.dtype != torch.float32:
                raise RuntimeError("canonical reference computation must stay FP32")
            del result
            last = args.positions
            logits = ref.project_vocabulary(checkpoint, hidden[:, -last:, :], args.head_tile).view(last, -1)
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError(f"nonfinite teacher-forced logits, {row['id']}")
            argmax_all, max_all = head_argmax(checkpoint, hidden[0], args.head_tile)
            forward_s = time.monotonic() - t0
            positions = list(range(len(full) - last, len(full)))
            # Diagnostics: teacher-forced vs cached-decode logits at the 32 shared positions
            # (position prompt_len-1+j predicts generated[j]).
            offset = last - args.new_tokens - 1  # index of position prompt_len-1 within the saved 48
            tf = logits[offset:offset + args.new_tokens]
            dec = torch.stack(decode_logits)
            kl = kl_rows(dec, tf)
            tf_argmax = tf.argmax(-1).tolist()
            key = f"prompt_{order}"
            tensors[key + ".logits"] = logits.contiguous()
            tensors[key + ".input_ids"] = torch.tensor(full, dtype=torch.int64)
            tensors[key + ".positions"] = torch.tensor(positions, dtype=torch.int64)
            records.append({
                "index": order, "id": row["id"], "thinking": False, "messages_sha256":
                    hashlib.sha256(json.dumps(row["messages"], sort_keys=True).encode()).hexdigest(),
                "prompt_tokens": prompt_len, "total_tokens": len(full), "prompt_ids": prompt,
                "generated_ids": generated, "generated_text": tokenizer.decode(generated),
                "saved_positions": positions,
                "saved_positions_note": f"0-based positions in input_ids; logits row i predicts input_ids[positions[i]+1] "
                                        f"(the last row predicts the token after the 32-token continuation)",
                "eos_in_continuation": [i for i, t in enumerate(generated) if t in stop_ids],
                "teacher_forced_argmax_matches_greedy": sum(int(a == b) for a, b in zip(tf_argmax, generated)),
                "teacher_forced_argmax_mismatch_steps": [j for j, (a, b) in enumerate(zip(tf_argmax, generated)) if a != b],
                "decode_vs_teacher_forced": {"kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
                                             "max_abs_logit_diff": float((dec - tf).abs().max())},
                "generation_seconds": generation_s, "teacher_forced_seconds": forward_s})
            progress(f"{row['id']}: P={prompt_len} gen {generation_s:.0f}s tf {forward_s:.0f}s "
                     f"tf/greedy argmax {records[-1]['teacher_forced_argmax_matches_greedy']}/32 "
                     f"KL(dec||tf) max {float(kl.max()):.3e}")
            save_file({"hidden": hidden[0].cpu().contiguous(), "input_ids": torch.tensor(full, dtype=torch.int64),
                       "main_argmax": argmax_all, "main_max_logit": max_all},
                      calib_dir / f"{row['id']}.safetensors",
                      metadata={"id": row["id"], "role": "reference-eval", "prompt_tokens": str(prompt_len),
                                "hidden": "final post-norm hidden of the FP32 reference, teacher-forced"})
            del hidden, logits
        for row in calibration:
            prompt = rendered[row["id"]]
            t0 = time.monotonic()
            result = model(input_ids=torch.tensor([prompt], dtype=torch.int64, device=device), use_cache=False,
                           return_dict=True)
            hidden = result.last_hidden_state
            del result
            argmax_all, max_all = head_argmax(checkpoint, hidden[0], args.head_tile)
            save_file({"hidden": hidden[0].cpu().contiguous(), "input_ids": torch.tensor(prompt, dtype=torch.int64),
                       "main_argmax": argmax_all, "main_max_logit": max_all},
                      calib_dir / f"{row['id']}.safetensors",
                      metadata={"id": row["id"], "role": "calibration", "prompt_tokens": str(len(prompt)),
                                "hidden": "final post-norm hidden of the FP32 reference, prompt only"})
            progress(f"calibration forward {row['id']}: {len(prompt)} tokens, {time.monotonic() - t0:.0f}s")
            del hidden
    artifact = out_dir / "long-code-reference.safetensors"
    save_file(tensors, artifact, metadata={"model_id": ref.MODEL_ID, "revision": ref.REVISION,
              "computation": "HF canonical eager FP32; raw NVFP4 weight-only; calibrated FP8 activations; "
                             "teacher-forced single forward, no cache"})
    torch.cuda.synchronize(device)
    report = {
        "model_id": ref.MODEL_ID, "revision": ref.REVISION, "status": "complete", "artifact": str(artifact),
        "artifact_sha256": sha256_file(artifact),
        "schema": {"prompt_<i>.logits": "float32 [48, 248320], unnormalized logits at saved_positions",
                   "prompt_<i>.input_ids": "int64 [prompt_tokens+32], prompt ids followed by the greedy continuation",
                   "prompt_<i>.positions": "int64 [48], 0-based positions of the logits rows"},
        "selection": f"the {args.count} longest (by rendered token count) non-thinking 'code-*' prompts in "
                     f"{args.prompts}; ties broken by id",
        "rendering": "tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, "
                     "tokenize=True) with the checkpoint's chat_template.jinja; no BOS added; same request shape as "
                     "mtp-accept/accept.py (chat_template_kwargs.enable_thinking=false, llama-server --jinja)",
        "continuation": f"{args.new_tokens} greedy tokens from the same reference with HF DynamicCache "
                        f"(chunked GDN prefill, recurrent GDN decode), EOS not stopped",
        "teacher_forcing": "one forward over input_ids without cache (chunked GDN, eager attention); logits saved "
                           "at the last 48 positions = last 16 prompt positions + 32 continuation positions",
        "comparison": "Teacher-force input_ids through the candidate; compare logits rows at identical positions. "
                      "Report KLD(reference||candidate) and top-1 agreement. The decode_vs_teacher_forced numbers give "
                      "the reference's own cached-vs-uncached path difference.",
        "prompts": records,
        "torch": torch.__version__, "transformers": transformers.__version__, "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "config_sha256": ref.sha256(checkpoint.path / "config.json"),
        "index_sha256": ref.sha256(checkpoint.path / "model.safetensors.index.json"),
        "chat_template_sha256": ref.sha256(checkpoint.path / "chat_template.jinja"),
        "prompts_jsonl_sha256": ref.sha256(args.prompts),
        "hf_model_source_sha256": ref.sha256(Path(inspect.getsourcefile(hf))),
        "reference_model_py_sha256": ref.sha256(Path(ref.__file__)),
        "gpu_reference_py_sha256": ref.sha256(Path(__file__)),
        "checkpoint_files": checkpoint.provenance, "gpu_decode_self_check": checked,
        "options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "calibration_hidden_dir": str(calib_dir),
        "calibration_prompt_ids": [r["id"] for r in calibration]}
    (out_dir / "long-code-reference.json").write_text(json.dumps(report, indent=2) + "\n")
    checkpoint.close()
    progress(f"long references complete: {artifact}")


def run_noise_floor(args):
    """Second FP32 realization of the long references: same arithmetic, different GEMM shape (the sequence is
    duplicated into a batch of 2, which changes cuBLAS tiling/summation order).  Without FP8 activation
    quantization such a change moves logits by ~1e-5; with it, rounding flips are amplified.  The per-position
    KL between the two realizations is the reference's own noise floor for candidate comparisons."""
    import torch
    from safetensors.torch import load_file, save_file

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    started = time.monotonic()
    primary = load_file(args.out / "long-code-reference.safetensors")
    meta = json.loads((args.out / "long-code-reference.json").read_text())
    checkpoint = GpuRawCheckpoint(args.checkpoint, device)
    self_check(checkpoint)
    model, hf = ref.build_streaming_model(checkpoint, device)
    alt, rows, all_kl = {}, [], []
    with torch.inference_mode():
        for record in meta["prompts"]:
            key = f"prompt_{record['index']}"
            ids = primary[key + ".input_ids"].to(device)
            batch = ids[None].expand(args.noise_batch, -1).contiguous()
            result = model(input_ids=batch, use_cache=False, return_dict=True)
            hidden = result.last_hidden_state[:1]
            del result
            last = primary[key + ".logits"].shape[0]
            logits = ref.project_vocabulary(checkpoint, hidden[:, -last:, :], args.head_tile).view(last, -1)
            kl = kl_rows(primary[key + ".logits"], logits)
            diff = (primary[key + ".logits"] - logits).abs().max(-1).values
            top1 = int((primary[key + ".logits"].argmax(-1) == logits.argmax(-1)).sum())
            alt[key + ".logits"] = logits.contiguous()
            rows.append({"index": record["index"], "id": record["id"], "kl_mean": float(kl.mean()),
                         "kl_max": float(kl.max()), "max_abs_logit_diff": float(diff.max()),
                         "median_rowmax_abs_diff": float(diff.median()), "top1_agree": top1, "positions": last,
                         "kl_per_position": [float(x) for x in kl]})
            all_kl.append(kl)
            progress(f"noise floor {record['id']}: KL mean {float(kl.mean()):.3e} max {float(kl.max()):.3e} "
                     f"top1 {top1}/{last}")
    kl = torch.cat(all_kl)
    artifact = args.out / "long-code-reference-alt.safetensors"
    save_file(alt, artifact, metadata={"model_id": ref.MODEL_ID, "revision": ref.REVISION,
              "computation": f"second realization: same as primary, teacher-forced batch of {args.noise_batch} duplicates"})
    summary = {"artifact": str(artifact), "artifact_sha256": sha256_file(artifact),
               "method": f"teacher-forced forward repeated with the sequence duplicated into a batch of {args.noise_batch} "
                         "(different GEMM shapes, identical arithmetic); KL(primary||alt) per saved position",
               "positions": int(kl.numel()), "kl_mean": float(kl.mean()), "kl_p99": float(kl.quantile(.99)),
               "kl_max": float(kl.max()), "top1_agree": sum(r["top1_agree"] for r in rows), "per_prompt": rows,
               "elapsed_seconds": time.monotonic() - started}
    (args.out / "long-code-reference-noise-floor.json").write_text(json.dumps(summary, indent=2) + "\n")
    meta["noise_floor"] = {k: v for k, v in summary.items() if k != "per_prompt"} | {
        "file": str(args.out / "long-code-reference-noise-floor.json")}
    (args.out / "long-code-reference.json").write_text(json.dumps(meta, indent=2) + "\n")
    checkpoint.close()
    progress(f"noise floor complete: KL mean {summary['kl_mean']:.3e}, p99 {summary['kl_p99']:.3e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("repro", "long", "noise"))
    parser.add_argument("--root", type=Path, required=True, help="output directory for repro mode")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--vram-fraction", type=float, default=0.0,
                        help="cap this process's CUDA allocator to this fraction of the card (lease guard)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calib-out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--positions", type=int, default=48)
    parser.add_argument("--head-tile", type=int, default=2048)
    parser.add_argument("--noise-batch", type=int, default=2)
    parser.add_argument("--prompt-batch-size", type=int, default=3, help="repro: reference_model.py's batch (3 = its original run)")
    parser.add_argument("--diagnostic-no-fp8-act", action="store_true",
                        help="DIAGNOSTIC ONLY: replace the calibrated FP8 activation hook by identity")
    args = parser.parse_args()
    os.environ["USE_HUB_KERNELS"] = "NO"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch

    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("assign CUDA_VISIBLE_DEVICES (the allocated RTX PRO 6000's UUID) through your allocation wrapper")
    name = torch.cuda.get_device_name(0)
    if "PRO 6000" not in name:
        raise RuntimeError(f"expected an RTX PRO 6000, got {name}")
    if args.vram_fraction:
        torch.cuda.set_per_process_memory_fraction(args.vram_fraction, 0)
    progress(f"device {name}, CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}, torch {torch.__version__}")
    if args.diagnostic_no_fp8_act:
        progress("DIAGNOSTIC: FP8 activation quantization disabled (not a reference)")
        ref.fp8_input_hook = lambda scale: (lambda module, inputs: None)
    if args.mode == "repro":
        device = args.device

        def make_checkpoint(path):
            checkpoint = GpuRawCheckpoint(path, device)
            self_check(checkpoint)
            return checkpoint
        ref.RawCheckpoint = make_checkpoint  # main() looks the class up at call time
        sys.argv = [ref.__file__, "--root", str(args.root), "--checkpoint", str(args.checkpoint), "--device", device, "--threads", str(args.threads),
                    "--prompt-batch-size", str(args.prompt_batch_size)]
        ref.main()
    else:
        torch.set_num_threads(args.threads)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        run_noise_floor(args) if args.mode == "noise" else run_long(args)


if __name__ == "__main__":
    main()
