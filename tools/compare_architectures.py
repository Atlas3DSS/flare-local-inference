#!/usr/bin/env python3
"""Capture ordinary Flare graphs on SM70/SM120, then compare identical contexts.

Run capture only inside an independently authorized GPU lease. This does not
launch a server, acquire a lease, export a package, or qualify native/MTP paths.
Both captures consume the SAME public short reference from reference_model.py.
That reference fixes the teacher-forced prefix after independent greedy paths
have diverged. Only full-vocabulary logits are accepted; there is no top-k KL.

Example (repeat capture with sm120 and its allocated UUID/cache/memory budget):
  python tools/compare_architectures.py capture --architecture sm70 \
    --checkpoint CHECKPOINT --reference REFERENCE/reference.safetensors \
    --output-dir RESULTS --gpu-uuid GPU-... --lease-gib 30 \
    --kernel-cache CACHE --compile-workers N --cpu-threads N
  python tools/compare_architectures.py compare \
    --sm70 RESULTS70/sm70.capture.json --sm120 RESULTS120/sm120.capture.json \
    --reference REFERENCE/reference.safetensors --output RESULTS/comparison.json

Short public reference generation is documented in docs/sm70/EXPORT.md.
Capture artifacts include the checkpoint's complete file-content hashes. Source,
reference, prompt IDs, teacher-forced IDs and execution mode must match. The
report retains genuine architecture-specific arithmetic/KV formats and reports
measurements without invented acceptance bounds or an exactness guarantee.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

if __package__:
    from . import reference_model, validate_candidate
else:
    import reference_model
    import validate_candidate

ARCHITECTURES = {"sm70": (7, 0), "sm120": (12, 0)}
FORMAT = "flare-ordinary-cross-architecture-v1"
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_new(path, value):
    with Path(path).open("x") as target:
        target.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_identity():
    # Include actual current code, including uncommitted public harness changes.
    paths = [ROOT / "pyproject.toml", ROOT / "uv.lock"]
    for directory in ("ignition", "heliopause", "helios", "tools"):
        for parent, directories, filenames in os.walk(ROOT / directory):
            directories[:] = sorted(name for name in directories if name not in ('target', '__pycache__', '.git'))
            paths.extend(Path(parent) / name for name in filenames
                         if Path(name).suffix in (".py", ".cu", ".cuh", ".cpp", ".h", ".hpp", ".rs", ".toml", ".lock"))
    files = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def checkpoint_identity(checkpoint, provenance):
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    names = set(index["weight_map"].values()) | {"config.json", "model.safetensors.index.json",
                                               "tokenizer.json", "tokenizer_config.json"}
    result = {}
    for name in sorted(names):
        if Path(name).name != name:
            raise ValueError("checkpoint must use flat indexed files")
        pin = checkpoint / ".cache" / "huggingface" / "download" / (name + ".metadata")
        lines = pin.read_text().splitlines()
        if len(lines) < 2 or lines[0] != reference_model.REVISION:
            raise ValueError("checkpoint file lacks the pinned revision: " + name)
        if provenance["checkpoint_files"].get(name) != {"revision": lines[0], "etag": lines[1]}:
            raise ValueError("checkpoint provenance differs from the common reference: " + name)
        result[name] = sha256(checkpoint / name)
        # HF LFS etags are the file SHA256; small Git blobs use SHA1 including
        # the Git object header. Bind the pin to bytes, not just a metadata label.
        etag = lines[1]
        if len(etag) == 64:
            actual_etag = result[name]
        elif len(etag) == 40:
            payload = (checkpoint / name).read_bytes()
            actual_etag = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        else:
            raise ValueError("unsupported checkpoint content etag: " + name)
        if etag != actual_etag:
            raise ValueError("checkpoint bytes differ from pinned content etag: " + name)
    for name, key in (("config.json", "config_sha256"), ("model.safetensors.index.json", "index_sha256")):
        if result[name] != provenance[key]:
            raise ValueError("checkpoint differs from the common reference: " + name)
    return result


def public_reference(path):
    prompts, _ = validate_candidate._reference(path)
    report = json.loads(Path(path).with_suffix(".json").read_text())
    if (report.get("status") != "complete" or report.get("model_id") != reference_model.MODEL_ID
            or report.get("revision") != reference_model.REVISION
            or len(prompts) != len(reference_model.PROMPTS)
            or len(report.get("prompts", [])) != len(prompts)):
        raise ValueError("requires the complete pinned public short reference")
    for index, (_, ids, generated, _, _) in enumerate(prompts):
        row = report["prompts"][index]
        if (row.get("source_text") != reference_model.PROMPTS[index]
                or row.get("prompt_ids") != ids.tolist() or row.get("generated_ids") != generated.tolist()):
            raise ValueError("public reference text/token identity mismatch")
    return prompts, report


def capture(args):
    import torch
    from tokenizers import Tokenizer

    capability = ARCHITECTURES[args.architecture]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid or not args.gpu_uuid.startswith("GPU-"):
        raise ValueError("CUDA_VISIBLE_DEVICES must equal the externally leased --gpu-uuid")
    if not args.output_dir.is_dir() or not args.kernel_cache.is_dir():
        raise ValueError("output and kernel-cache directories must already exist")
    manifest = args.output_dir / (args.architecture + ".capture.json")
    for path in (manifest, args.output_dir / (args.architecture + ".json"),
                 args.output_dir / (args.architecture + ".safetensors")):
        if path.exists():
            raise FileExistsError("refusing to overwrite capture evidence: " + str(path))
    if min(args.compile_workers, args.cpu_threads, args.lease_gib) < 1:
        raise ValueError("explicit CPU and lease budgets must be positive")
    prompts, provenance = public_reference(args.reference)
    identity = checkpoint_identity(args.checkpoint, provenance)
    config_doc = json.loads((args.checkpoint / "config.json").read_text())
    vocabulary = config_doc["text_config"]["vocab_size"]
    if any(prompt[3].shape[1] != vocabulary for prompt in prompts):
        raise ValueError("reference must contain the checkpoint's entire vocabulary")
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    for index, (_, ids, _, _, _) in enumerate(prompts):
        if tokenizer.encode(reference_model.PROMPTS[index], add_special_tokens=False).ids[:len(ids)] != ids.tolist():
            raise ValueError("public reference IDs differ from the checkpoint tokenizer")
    torch.set_num_threads(args.cpu_threads)
    torch.cuda.set_device(0)
    validate_candidate.require_capability(torch.cuda.get_device_capability(0), capability)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(ROOT))
    from ignition.preparation.package_storage import PackageStorage
    from ignition.recipes.qwen38_dense.bootstrap import PreparationConfig, prepare

    length = max(len(ids) + len(generated) - 1 for _, ids, generated, _, _ in prompts)
    context = ((length + 127) // 128) * 128
    total = torch.cuda.get_device_properties(0).total_memory
    # Bound allocation to the external lease, with additional safety headroom.
    headroom = max(0, total - (args.lease_gib << 30)) + (256 << 20)
    config = PreparationConfig(device=0, context=context, max_active=1, prefill_chunk_size=128,
        snapshot_budget_bytes=1 << 30, headroom_bytes=headroom, compile_workers=args.compile_workers,
        preparation_pages=context // 128, retain_graphs=True, mtp=False,
        kernel_cache=args.kernel_cache, cutlass_include=args.cutlass_include)
    source_hash = source_identity()
    with PackageStorage(device=0) as storage:
        prepared = prepare(args.checkpoint, config, package_storage=storage, progress=print)
    with prepared:
        report = validate_candidate.validate_candidate(prepared, args.reference, args.output_dir,
            mode="ordinary", artifact_stem=args.architecture, expected_capability=capability)
    if source_identity() != source_hash:
        raise RuntimeError("source changed during capture; comparison evidence not admitted")
    write_new(manifest, {"format": FORMAT, "status": "complete", "architecture": args.architecture,
        "compute_capability": list(capability), "gpu_uuid": args.gpu_uuid,
        "checkpoint_files_sha256": identity, "source_sha256": source_hash,
        "reference_sha256": sha256(args.reference),
        "reference_report_sha256": sha256(args.reference.with_suffix(".json")),
        "vocabulary": vocabulary, "mode": "ordinary", "mtp": False,
        "context": context, "prefill_chunk_size": 128, "max_active": 1,
        "arithmetic": {"continuation_dtype": "float32" if capability == (7, 0) else "bfloat16",
                       "kv_element_bytes": 2 if capability == (7, 0) else 1,
                       "note": "native recipe choices preserved; not forced to match architectures",
                       "tf32": False},
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "device_name": torch.cuda.get_device_name(0)},
        "candidate_report": args.architecture + ".json",
        "candidate_report_sha256": sha256(args.output_dir / (args.architecture + ".json")),
        "artifact": args.architecture + ".safetensors", "artifact_sha256": report["artifact_sha256"],
        "scope": "ordinary non-MTP retained Flare graphs; host argmax; native serving not tested"})
    return {"capture": str(manifest), "status": "complete"}


def load_capture(path, architecture):
    from safetensors import safe_open
    from safetensors.torch import load_file

    document = json.loads(path.read_text())
    if (document.get("format") != FORMAT or document.get("status") != "complete"
            or document.get("architecture") != architecture
            or document.get("compute_capability") != list(ARCHITECTURES[architecture])
            or document.get("mode") != "ordinary" or document.get("mtp") is not False):
        raise ValueError("requires a complete ordinary capture on the named architecture")
    for field in ("candidate_report", "artifact"):
        name = document[field]
        if Path(name).name != name or sha256(path.parent / name) != document[field + "_sha256"]:
            raise ValueError("capture file hash/path mismatch: " + field)
    report = json.loads((path.parent / document["candidate_report"]).read_text())
    if (report.get("status") != "measured" or report.get("state_restored") is not True
            or report.get("all_expected_positions_compared") is not True
            or report.get("independent_greedy", {}).get("status") != "complete"
            or report.get("compute_capability") != document["compute_capability"]
            or report.get("mode") != "ordinary"
            or report.get("reference_sha256") != document["reference_sha256"]
            or report.get("artifact_sha256") != document["artifact_sha256"]):
        raise ValueError("candidate report is incomplete or does not bind this capture")
    artifact = path.parent / document["artifact"]
    with safe_open(artifact, framework="pt", device="cpu") as source:
        metadata = source.metadata() or {}
    if (metadata.get("status") != "measured" or metadata.get("model_id") != reference_model.MODEL_ID
            or metadata.get("revision") != reference_model.REVISION
            or metadata.get("reference_sha256") != document["reference_sha256"]):
        raise ValueError("candidate tensor metadata does not bind the common reference")
    return document, load_file(artifact, device="cpu")


def compare_tensors(left, right, prompts, vocabulary):
    """Compare all positions on identical reference prefixes; greedy is separate."""
    import torch

    rows = []
    for index, (prefix, ids, generated, reference_prefill, _) in enumerate(prompts):
        matrices, greedy = [], []
        for tensors in (left, right):
            for name, expected in (("prompt_ids", ids), ("generated_ids", generated)):
                actual = tensors[f"{prefix}.{name}"]
                if actual.dtype != torch.int64 or not torch.equal(actual, expected):
                    raise ValueError("teacher-forced token contexts must match the common reference")
            prefill, decode = (tensors[f"{prefix}.{name}_logits"] for name in ("prefill", "decode"))
            if (reference_prefill.shape[1] != vocabulary or prefill.shape != (len(ids), vocabulary)
                    or decode.shape != (len(generated) - 1, vocabulary)):
                raise ValueError("requires every position and the entire checkpoint vocabulary")
            matrices.append(torch.cat((prefill, decode)))
            tokens, logits = tensors[f"{prefix}.greedy_ids"], tensors[f"{prefix}.greedy_logits"]
            if (tokens.dtype != torch.int64 or tokens.shape != generated.shape
                    or logits.shape != (len(generated), vocabulary) or not logits.is_floating_point()
                    or not bool(torch.isfinite(logits).all()) or not torch.equal(tokens, logits.argmax(-1))):
                raise ValueError("independent greedy IDs must match their full captured logits")
            if not torch.equal(logits[0], prefill[-1]):
                raise ValueError("greedy branch must start at the common prompt endpoint")
            greedy.append(tokens)
        forward, per_forward = validate_candidate.distribution_metrics(*matrices)
        reverse, per_reverse = validate_candidate.distribution_metrics(*reversed(matrices))
        equal = greedy[0] == greedy[1]
        mismatches = (~equal).nonzero().flatten().tolist()
        rows.append({"prompt_id": f"reference-public-{index:02d}", "prompt_token_ids": ids.tolist(),
            "teacher_forced_input_ids": torch.cat((ids, generated[:-1])).tolist(),
            "common_continuation_ids": generated.tolist(), "positions": forward["positions"],
            "sm70_to_sm120": forward, "sm120_to_sm70": reverse,
            "per_position_kl_sm70_to_sm120": per_forward["kld"].tolist(),
            "per_position_kl_sm120_to_sm70": per_reverse["kld"].tolist(),
            "teacher_forced_top1_sm70": per_forward["reference_top1"].tolist(),
            "teacher_forced_top1_sm120": per_forward["candidate_top1"].tolist(),
            "independent_greedy": {"sm70_ids": greedy[0].tolist(), "sm120_ids": greedy[1].tolist(),
                "exact_sequence_match": not mismatches, "common_prefix_tokens": mismatches[0] if mismatches else len(equal),
                "positional_token_agreement": float(equal.double().mean()),
                "note": "After first divergence these greedy positions have different contexts; no KL is computed on them."}})
    return rows


def compare(args):
    if args.output.exists():
        raise FileExistsError("refusing to overwrite comparison evidence")
    prompts, _ = public_reference(args.reference)
    left, left_tensors = load_capture(args.sm70, "sm70")
    right, right_tensors = load_capture(args.sm120, "sm120")
    fields = ("checkpoint_files_sha256", "source_sha256", "reference_sha256", "reference_report_sha256",
              "vocabulary", "mode", "context", "prefill_chunk_size", "max_active", "mtp")
    for field in fields:
        if left.get(field) is None or left[field] != right.get(field):
            raise ValueError("capture identities/configuration differ: " + field)
    if (left["reference_sha256"] != sha256(args.reference)
            or left["reference_report_sha256"] != sha256(args.reference.with_suffix(".json"))):
        raise ValueError("comparison requires the exact shared reference artifact and provenance")
    rows = compare_tensors(left_tensors, right_tensors, prompts, left["vocabulary"])
    result = {"format": FORMAT, "status": "measured", "numerical_pass": None,
        "scope": "ordinary retained Flare graphs; independent greedy and full-vocabulary common-prefix KL",
        "native_runtime_tested": False, "mtp_tested": False, "stop_on_eos": False,
        "interpretation": "No cross-architecture exactness guarantee or acceptance threshold is asserted. "
                          "Teacher forcing follows the same public FP32 reference continuation after divergence.",
        "captures": {"sm70": left, "sm120": right},
        "capture_manifest_sha256": {"sm70": sha256(args.sm70), "sm120": sha256(args.sm120)},
        "prompts": rows}
    write_new(args.output, result)
    return {"report": str(args.output), "status": "measured", "numerical_pass": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    capture_parser = commands.add_parser("capture", help="authorized GPU diagnostic; no server or export")
    capture_parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    for name in ("checkpoint", "reference", "output-dir", "kernel-cache"):
        capture_parser.add_argument("--" + name, type=Path, required=True)
    capture_parser.add_argument("--gpu-uuid", required=True)
    for name in ("lease-gib", "compile-workers", "cpu-threads"):
        capture_parser.add_argument("--" + name, type=int, required=True)
    capture_parser.add_argument("--cutlass-include", type=Path)
    comparison = commands.add_parser("compare", help="CPU-only comparison of captured full logits")
    for name in ("sm70", "sm120", "reference", "output"):
        comparison.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(capture(args) if args.command == "capture" else compare(args)))


if __name__ == "__main__":
    main()
