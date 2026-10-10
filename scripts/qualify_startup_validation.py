"""Exercise installed native startup rejection before CUDA or socket operations.

Manifest addresses are synthetic metadata, never valid allocation handles. Every
case must reject before CUDA driver loading or socket creation. strace guards
socket and ioctl calls while recording attempted operations in each subprocess.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flare.preparation.qwen38 import (
    CONFIG_SHA256, FRONTEND_SHA256, GENERATION_SHA256, MODEL_ID, REVISION,
)

SNAPSHOT = Path.home() / ".cache/huggingface/hub" / (
    "models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD"
) / "snapshots" / REVISION
ABI_VERSION = 2
GRAPH_CAPACITIES = {
    **{kind: ((1, 1), (2, 1), (4, 1)) for kind in ("init", "restore", "snapshot", "resume", "decode")},
    "prefill": tuple((1, n) for n in (128, 256, 512)),
    "vision": tuple((1, n) for n in (256, 1024, 4096, 16384)),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fixture(snapshot):
    """Return shape-valid metadata for the declared native Qwen contract."""
    vocabulary, context = 248320, 262144
    descriptions = [
        ("row_slot_ids", "int32", [4]),
        ("row_generations", "int64", [4]),
        ("live_rows", "int32", [1]),
        ("query_lengths", "int32", [4]),
        ("token_ids", "int64", [512]),
        ("prefill_positions", "int64", [512, 3]),
        ("page_tables", "int32", [4, context // 64]),
        ("sample_enabled", "int32", [4]),
        ("sampling_grammar_mask", "int32", [4, vocabulary // 32]),
        ("vision_token_indices", "int32", [512]),
        ("vision_pixel_values", "float32", [16384, 1536]),
        ("vision_interpolation_indices", "int64", [16384, 4]),
        ("vision_interpolation_weights", "float32", [16384, 4]),
        ("vision_position_ids", "int64", [16384, 2]),
        ("vision_cu_seqlens", "int32", [2]),
        ("vision_num_patches", "int32", [1]),
        ("vision_slot", "int32", [1]),
        ("vision_generation", "int64", [1]),
        ("vision_feature_offset", "int32", [1]),
        ("admission_count", "int32", [1]),
        ("admission_slots", "int32", [4]),
        ("admission_generations", "int64", [4]),
        ("admission_seed_bits", "int64", [4]),
        ("admission_parameters", "float32", [4, 6]),
        ("admission_top_k", "int32", [4]),
        ("admission_max_new_tokens", "int64", [4]),
        ("admission_eos_token_ids", "int64", [4, 8]),
        ("admission_eos_count", "int32", [4]),
        ("admission_histogram", "int32", [4, vocabulary]),
        ("admission_grammar_mask", "int32", [4, vocabulary // 32]),
        ("admission_grammar_enabled", "int32", [4]),
        ("state_source_slots", "int64", [4]),
        ("state_destination_slots", "int64", [4]),
        ("state_operation", "int32", [4]),
        ("state_qsa_slots", "int64", [4]),
        ("state_accepted_tokens", "int32", [4]),
        ("state_expected_consumed", "int64", [4]),
        ("state_num_ops", "int32", [1]),
        ("frame_index", "int32", [1]),
        ("submission_sequence", "int64", [1]),
        ("completion_records", "int64", [2, 4, 8]),
    ]
    buffers, bindings = [], []
    address, host = 0x10000, 0
    for name, dtype, shape in descriptions:
        nbytes = math.prod(shape) * (8 if dtype == "int64" else 4)
        buffers.append({"name": name, "address": address, "nbytes": nbytes,
            "dtype": dtype, "shape": shape,
            "strides": [math.prod(shape[i + 1:]) for i in range(len(shape))]})
        address += (nbytes + 7) & ~7
        if name != "completion_records":
            bindings.append({"buffer_name": name, "host_offset": host, "nbytes": nbytes})
            host += (nbytes + 7) & ~7
    graphs = []
    for kind, capacities in GRAPH_CAPACITIES.items():
        for batch, tokens in capacities:
            graphs.append({"name": f"{kind}-{batch}-{tokens}", "kind": kind,
                "sampling_mode": "general", "handle": len(graphs) + 1,
                "batch_capacity": batch, "token_capacity": tokens,
                "input_bindings": deepcopy(bindings)})
    manifest = {"abi_version": ABI_VERSION, "context": 1, "device": 0, "max_active": 4,
        "input_frame_bytes": host, "buffers": buffers, "graphs": graphs}
    assets = {"tokenizer": ("tokenizer.json", FRONTEND_SHA256["tokenizer.json"]),
        "tokenizer_config": ("tokenizer_config.json", FRONTEND_SHA256["tokenizer_config.json"]),
        "chat_template": ("chat_template.jinja", FRONTEND_SHA256["chat_template.jinja"]),
        "preprocessor_config": ("preprocessor_config.json", FRONTEND_SHA256["preprocessor_config.json"]),
        "model_config": ("config.json", CONFIG_SHA256),
        "generation_config": ("generation_config.json", GENERATION_SHA256)}
    config = {"model_id": MODEL_ID, "identity": "01" * 32, "max_sequence_tokens": context,
        "prefill_chunk_size": 512,
        "pages": 4096, "snapshots": 4, "eos_token_ids": [248046, 248044],
        "model_contract": {"schema": "qwen4exp.graph-inputs.v2", "vocabulary": vocabulary,
            "vision": {"features_per_request": 16384, "max_images": 4,
                "patches_per_feature": 4, "patch_capacities": [256, 1024, 4096, 16384]}},
        "bind": "127.0.0.1:0",
        "assets": {key: {"path": str((snapshot / filename).resolve(strict=True)), "sha256": sha}
            for key, (filename, sha) in assets.items()}}
    return manifest, config


def cases(snapshot):
    manifest, config = fixture(snapshot)
    result = []

    def add(name, expected, change):
        m, c = deepcopy(manifest), deepcopy(config)
        change(m, c)
        result.append({"name": name, "expected_error": expected, "manifest": m, "config": c})

    def buffer(m, name):
        return next(item for item in m["buffers"] if item["name"] == name)

    add("abi-version", "unsupported execution ABI version", lambda m, c: m.update(abi_version=4))
    add("null-context", "nonnull context", lambda m, c: m.update(context=0))
    add("overlapping-buffers", "overlap", lambda m, c:
        buffer(m, "row_generations").update(address=buffer(m, "row_slot_ids")["address"]))
    add("completion-layout", "required buffer layout", lambda m, c:
        buffer(m, "completion_records").update(shape=[4, 2, 8], strides=[16, 8, 1]))
    add("missing-resume4", "lacks Resume/General graph with batch 4, tokens 1", lambda m, c:
        m.update(graphs=[g for g in m["graphs"] if g["name"] != "resume-4-1"]))
    add("missing-prefill512", "lacks Prefill/General graph with batch 1, tokens 512", lambda m, c:
        m.update(graphs=[g for g in m["graphs"] if g["name"] != "prefill-1-512"]))

    add("missing-sampling-mode", "'sampling_mode'", lambda m, c: m["graphs"][0].pop("sampling_mode"))
    add("unknown-sampling-mode", 'unsupported graph sampling mode "unknown"', lambda m, c:
        m["graphs"][0].update(sampling_mode="unknown"))
    def ambiguous_variant(m, c):
        graph = deepcopy(next(g for g in m["graphs"] if g["name"] == "prefill-1-512"))
        graph.update(name="duplicate-prefill-1-512", handle=len(m["graphs"]) + 1)
        m["graphs"].append(graph)

    add("ambiguous-graph-capacity", "ambiguous prepared model graph capacity", ambiguous_variant)

    def missing_input(m, c):
        graph = next(g for g in m["graphs"] if g["name"] == "prefill-1-512")
        graph["input_bindings"] = [b for b in graph["input_bindings"]
            if b["buffer_name"] != "vision_token_indices"]

    add("missing-semantic-input", "lacks required input vision_token_indices", missing_input)
    add("wrong-page-table-layout", "semantic buffer page_tables", lambda m, c:
        buffer(m, "page_tables").update(shape=[2, 8192], strides=[8192, 1]))
    add("invalid-model-context", "invalid prepared Qwen4Exp model configuration", lambda m, c:
        c.update(max_sequence_tokens=262145))
    add("valid-shapes-reach-asset-validation", "assets must contain verified frontend descriptors",
        lambda m, c: c.pop("assets"))
    add("asset-checksum", "frontend asset SHA256 mismatch", lambda m, c:
        c["assets"]["generation_config"].update(sha256="00" * 32))
    add("frontend-worker-capacity", "invalid HTTP admission limits", lambda m, c:
        c.update(frontend_workers=0))
    add("request-queue-capacity", "invalid native request/event queue capacity", lambda m, c:
        c.update(queue_capacity=0))
    add("event-queue-capacity", "invalid native request/event queue capacity", lambda m, c:
        c.update(event_capacity=1))
    return result


def child(path):
    import flare_helios

    case = json.loads(path.read_text())

    class Owner:
        pass

    owner = Owner()
    retained = weakref.ref(owner)
    result = {"name": case["name"]}
    try:
        server = flare_helios.Server(owner, case["manifest"], json.dumps(case["config"]))
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error),
            resources_retained=getattr(error, "resources_retained", False),
            rejected=True, expected_error=case["expected_error"])
    else:
        result.update(rejected=False, error="unexpected native server construction")
        server.close()
    del owner
    gc.collect()
    result["owner_released"] = retained() is None
    result["passed"] = bool(result["rejected"] and
        case["expected_error"] in result["error"] and result["owner_released"] and
        not result.get("resources_retained"))
    print(json.dumps(result), flush=True)
    return 0 if result["passed"] else 1


def run(args):
    import flare_helios

    if not shutil.which("strace"):
        raise RuntimeError("strace is required for startup syscall evidence")
    args.artifacts.mkdir(parents=True, exist_ok=False)
    extension = next(Path(flare_helios.__file__).parent.glob("*.so"))
    sources = [Path(__file__).resolve(), ROOT / "flare/preparation/qwen38.py",
        *sorted((ROOT / "helios/src").rglob("*.rs")), ROOT / "helios/Cargo.toml", ROOT / "helios/Cargo.lock"]
    receipt = {"schema": "flare-helios-startup-validation-v1", "scope":
        "Installed constructor rejects invalid synthetic manifest/config metadata before CUDA or networking; no prepared model or GPU graph is created.",
        "extension": {"path": str(extension), "sha256": digest(extension)},
        "abi_version": ABI_VERSION,
        "graph_capacities": GRAPH_CAPACITIES,
        "python": sys.version, "executable": sys.executable, "checkpoint_assets": str(args.snapshot),
        "reference_revision": REVISION, "sources_before": {str(p): digest(p) for p in sources},
        "strace_version": subprocess.check_output(["strace", "-V"], text=True).splitlines()[0], "cases": []}
    save(args.artifacts / "receipt.json", receipt)
    for case in cases(args.snapshot):
        prefix = args.artifacts / case["name"]
        save(prefix.with_suffix(".input.json"), case)
        trace = prefix.with_suffix(".strace")
        command = ["strace", "-f", "-qq", "-s", "512", "-o", str(trace),
            "-e", "trace=network,open,openat,ioctl,clone,clone3",
            "-e", "inject=socket:error=EPERM", "-e", "inject=bind:error=EPERM",
            "-e", "inject=listen:error=EPERM", "-e", "inject=ioctl:error=EPERM",
            sys.executable, "-c",
            'import runpy, sys; runpy.run_path(sys.argv.pop(1), run_name="__main__")',
            str(Path(__file__).resolve()), "--child", str(prefix.with_suffix(".input.json"))]
        started = time.monotonic()
        process = subprocess.run(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1"}, text=True, capture_output=True, timeout=60)
        prefix.with_suffix(".stdout").write_text(process.stdout)
        prefix.with_suffix(".stderr").write_text(process.stderr)
        try:
            observed = json.loads(process.stdout)
        except (json.JSONDecodeError, ValueError):
            observed = {"name": case["name"], "passed": False, "error": "child output was not JSON"}
        text = trace.read_text()
        network = [line for line in text.splitlines() if re.search(r"\b(?:socket|bind|listen|connect)\(", line)]
        cuda = [line for line in text.splitlines() if re.search(r"libcuda(?:rt)?[./]|/dev/nvidia", line)]
        threads = [line for line in text.splitlines() if re.search(r"\bclone(?:3)?\(", line)]
        observed.update(returncode=process.returncode, elapsed_seconds=time.monotonic() - started,
            command=command, network_attempts=network, cuda_open_attempts=cuda,
            thread_creation_attempts=len(threads), trace_sha256=digest(trace),
            input_sha256=digest(prefix.with_suffix(".input.json")))
        observed["passed"] = bool(observed["passed"] and process.returncode == 0 and not network and not cuda)
        receipt["cases"].append(observed)
        save(args.artifacts / "receipt.json", receipt)
        print(json.dumps({key: observed[key] for key in ("name", "passed", "error", "elapsed_seconds")}), flush=True)
    receipt["sources_after"] = {str(p): digest(p) for p in sources}
    receipt["sources_unchanged"] = receipt["sources_before"] == receipt["sources_after"]
    receipt["extension_unchanged"] = receipt["extension"]["sha256"] == digest(extension)
    receipt["passed"] = all(case["passed"] for case in receipt["cases"]) and receipt["sources_unchanged"] and receipt["extension_unchanged"]
    save(args.artifacts / "receipt.json", receipt)
    return 0 if receipt["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--child", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child is not None:
        return child(args.child)
    if args.artifacts is None:
        parser.error("--artifacts is required")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
