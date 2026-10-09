#!/usr/bin/env python3
"""Run a reproducible Qwen3.5 trace against b12x or sglang.

The point of this script is to compare equivalent prompts, not "similar"
requests. It canonicalizes the prompt into one exact token-id sequence and
then sends that sequence to the selected backend via raw completion.

Examples:
    python tests/serve/qwen35_trace.py --backend b12x --out /tmp/b12x_trace.json
    python tests/serve/qwen35_trace.py --backend sglang --out /tmp/sglang_trace.json
    python tests/serve/qwen35_trace.py --backend b12x --chat --out /tmp/b12x_chat.json
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import torch
from transformers import AutoTokenizer

torch.set_grad_enabled(False)

DEFAULT_MODEL_PATH = "/data/models/Qwen3.5-397B-A17B-NVFP4"
DEFAULT_PROMPT = "The capital of France is"
DEFAULT_MESSAGES = [{"role": "user", "content": DEFAULT_PROMPT}]
RAW_COMPLETION_PROMPT_IDS = [760, 6511, 314, 9338, 369]


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _parse_prompt_ids(value: str | None) -> list[int] | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if text.startswith("["):
        return [int(x) for x in json.loads(text)]
    return [int(x) for x in text.split(",")]


def build_prompt_spec(
    *,
    model_path: str,
    prompt: str,
    prompt_ids: list[int] | None,
    chat: bool,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)

    if prompt_ids is not None:
        ids = prompt_ids
        formatted = None
        source = {"kind": "explicit_ids"}
    elif chat:
        formatted = tokenizer.apply_chat_template(
            DEFAULT_MESSAGES if prompt == DEFAULT_PROMPT else [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer.encode(formatted, add_special_tokens=False)
        source = {
            "kind": "chat_template",
            "messages": DEFAULT_MESSAGES if prompt == DEFAULT_PROMPT else [{"role": "user", "content": prompt}],
        }
    else:
        formatted = prompt
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        source = {"kind": "completion_text", "prompt": prompt}

    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    return {
        "model_path": model_path,
        "source": source,
        "formatted_text": formatted,
        "prompt_ids": ids,
        "decoded_prompt": decoded,
    }


def write_trace(out_path: str, data: dict) -> None:
    Path(out_path).write_text(json.dumps(data, indent=2) + "\n")


def _b12x_worker(
    tp_group,
    model_path: str,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    out_path: str,
    prompt_spec_json: str,
    mode: str,
) -> None:
    rank = tp_group.rank if tp_group else 0
    device = f"cuda:{tp_group.device.index}" if tp_group else "cuda"

    from serve.engine.sampling import SamplingParams
    from serve.engine.serving import ServingEngine

    engine = ServingEngine(
        model_path,
        device=device,
        tp_group=tp_group,
        graph_batch_sizes=[],
    )

    if rank != 0:
        engine.run_follower()
        return

    if mode == "generate":
        result = engine.generate(
            prompt_ids,
            SamplingParams(
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            ),
        )
        trace = {
            "backend": "b12x",
            "mode": mode,
            "prompt": json.loads(prompt_spec_json),
            "generation": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "generated_ids": result.generated_ids,
                "generated_text": engine.tokenizer.decode(
                    result.generated_ids, skip_special_tokens=True
                ),
                "finish_reason": result.finish_reason,
                "ttft_ms": result.time_to_first_token_ms,
                "total_time_ms": result.total_time_ms,
            },
        }
        print(json.dumps(trace["generation"], indent=2), flush=True)
    elif mode == "direct_prefill":
        token_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        logits = engine.runner.prefill(
            token_ids,
            request_ids=[0],
            q_seqlens=[len(prompt_ids)],
        )
        next_token = int(logits[0].argmax().item())
        topk = torch.topk(logits[0], k=10)
        trace = {
            "backend": "b12x",
            "mode": mode,
            "prompt": json.loads(prompt_spec_json),
            "prefill": {
                "next_token_id": next_token,
                "next_token_text": engine.tokenizer.decode([next_token], skip_special_tokens=True),
                "topk_ids": topk.indices.tolist(),
                "topk_tokens": [
                    engine.tokenizer.decode([tok], skip_special_tokens=True)
                    for tok in topk.indices.tolist()
                ],
                "topk_logits": [float(x) for x in topk.values.tolist()],
            },
        }
        print(json.dumps(trace["prefill"], indent=2), flush=True)
    else:
        raise ValueError(f"unsupported b12x mode: {mode}")

    write_trace(out_path, trace)
    engine.shutdown()


def run_b12x(
    *,
    model_path: str,
    prompt_spec: dict,
    max_new_tokens: int,
    temperature: float,
    out_path: str,
    gpu_ids: list[int],
    mode: str,
) -> None:
    from serve.tp.launch import launch_tp

    launch_tp(
        _b12x_worker,
        world_size=len(gpu_ids),
        args=(
            model_path,
            prompt_spec["prompt_ids"],
            max_new_tokens,
            temperature,
            out_path,
            json.dumps(prompt_spec),
            mode,
        ),
        gpu_ids=gpu_ids,
    )


def _wait_for_server(port: int, timeout_s: float = 600.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"sglang server on port {port} did not become ready in {timeout_s}s")


def run_sglang(
    *,
    model_path: str,
    prompt_spec: dict,
    max_new_tokens: int,
    temperature: float,
    out_path: str,
    tp: int,
    logprobs: int,
    visible_devices: str | None,
) -> None:
    port = _find_free_port()
    env = os.environ.copy()
    if visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = visible_devices
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--tp",
        str(tp),
        "--port",
        str(port),
        "--disable-cuda-graph",
        "--mem-fraction-static",
        "0.75",
    ]

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        _wait_for_server(port)
        body = {
            "model": model_path,
            "prompt": prompt_spec["prompt_ids"],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if logprobs > 0:
            body["logprobs"] = logprobs
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = json.loads(resp.read())

        trace = {
            "backend": "sglang",
            "prompt": prompt_spec,
            "generation": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "generated_text": payload["choices"][0]["text"],
                "raw_response": payload,
            },
        }
        write_trace(out_path, trace)
        print(json.dumps(trace["generation"], indent=2), flush=True)
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace Qwen3.5 generation on b12x or sglang.")
    parser.add_argument("--backend", choices=["b12x", "sglang"], required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-ids", default=None, help="Comma-separated ints or JSON list.")
    parser.add_argument("--chat", action="store_true", help="Canonicalize prompt through the tokenizer chat template.")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--gpu-ids",
        default="0,1,2,3",
        help="Used by b12x TP launch and as CUDA_VISIBLE_DEVICES for sglang.",
    )
    parser.add_argument("--logprobs", type=int, default=0, help="Only used by sglang /v1/completions.")
    parser.add_argument(
        "--b12x-mode",
        choices=["generate", "direct_prefill"],
        default="generate",
        help="Only used by --backend b12x.",
    )
    args = parser.parse_args()

    prompt_spec = build_prompt_spec(
        model_path=args.model,
        prompt=args.prompt,
        prompt_ids=_parse_prompt_ids(args.prompt_ids),
        chat=args.chat,
    )

    if args.backend == "b12x":
        run_b12x(
            model_path=args.model,
            prompt_spec=prompt_spec,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            out_path=args.out,
            gpu_ids=[int(x) for x in args.gpu_ids.split(",")],
            mode=args.b12x_mode,
        )
    else:
        run_sglang(
            model_path=args.model,
            prompt_spec=prompt_spec,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            out_path=args.out,
            tp=len(args.gpu_ids.split(",")),
            logprobs=args.logprobs,
            visible_devices=args.gpu_ids,
        )


if __name__ == "__main__":
    main()
