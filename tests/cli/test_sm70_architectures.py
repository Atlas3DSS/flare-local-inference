"""Synthetic CPU evidence-contract tests; no architecture/model qualification."""
import copy
import inspect
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from tools import compare_architectures as comparison
from tools import validate_candidate


class ArchitectureTests(unittest.TestCase):
    def test_prepared_catalog_preserves_sm70_history_without_claiming_mtp_execution(self):
        sm70 = comparison.preparation_catalog("sm70")
        self.assertIs(sm70["mtp"], True)
        self.assertEqual(sm70["speculative_widths"], (1, 2, 3, 4, 5))
        self.assertEqual(3 + max(sm70["speculative_widths"]) - 1, 7)
        self.assertIs(comparison.preparation_catalog("sm120")["mtp"], False)
        with self.assertRaises(ValueError):
            comparison.preparation_catalog("sm80")

    def test_mtp_catalog_selects_only_ordinary_graphs_and_rejects_missing_ordinary_variants(self):
        signatures = {}
        for kind in ("init", "prefill", "snapshot", "restore", "resume", "decode"):
            for enabled in (True, False):
                name = kind + "1" + ("" if enabled else "-no-mtp")
                signatures[name] = SimpleNamespace(kind=kind, batch_capacity=1, token_capacity=1,
                                                  speculation_enabled=enabled)
        signatures["prefill128-no-mtp"] = SimpleNamespace(kind="prefill", batch_capacity=1,
                                                         token_capacity=128, speculation_enabled=False)
        ordinary = validate_candidate._ordinary_graph_names(signatures, mode="ordinary")
        self.assertEqual(ordinary, {kind: kind + "1-no-mtp"
                                    for kind in ("init", "prefill", "snapshot", "restore", "resume", "decode")})
        self.assertEqual(validate_candidate._ordinary_graph_names(signatures, mode="chunked")["prefill"],
                         "prefill128-no-mtp")
        self.assertNotIn("decode", validate_candidate._ordinary_graph_names(signatures, mode="sequential"))
        for kind in ordinary:
            missing = {name: signature for name, signature in signatures.items()
                       if not (signature.kind == kind and signature.speculation_enabled is False)}
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "ordinary " + kind):
                validate_candidate._ordinary_graph_names(missing, mode="ordinary")
        signatures["resume-duplicate"] = signatures["resume1-no-mtp"]
        with self.assertRaisesRegex(ValueError, "one one-request captured resume"):
            validate_candidate._ordinary_graph_names(signatures, mode="ordinary")

    def fixture(self):
        ids, generated = torch.tensor([1, 2]), torch.tensor([3, 4, 4])
        prefill = torch.zeros(2, 32); prefill[0, 2] = 4; prefill[1, 3] = 4
        decode = torch.zeros(2, 32); decode[:, 4] = 4
        greedy_logits = torch.cat((prefill[-1:], decode)).clone()
        tensors = {"prompt_0.prompt_ids": ids, "prompt_0.generated_ids": generated,
                   "prompt_0.prefill_logits": prefill, "prompt_0.decode_logits": decode,
                   "prompt_0.greedy_ids": generated.clone(), "prompt_0.greedy_logits": greedy_logits}
        return tensors, [("prompt_0", ids, generated, prefill, decode)]

    def test_capability_default_stays_sm70_and_explicit_sm120_is_required(self):
        self.assertEqual(inspect.signature(validate_candidate.validate_candidate).parameters[
            "expected_capability"].default, (7, 0))
        validate_candidate.require_capability((7, 0))
        with self.assertRaisesRegex(ValueError, "SM70"):
            validate_candidate.require_capability((12, 0))
        validate_candidate.require_capability((12, 0), (12, 0))
        for invalid in (None, (8, 0), (True, 0), [12, 0]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_candidate.require_capability((12, 0), invalid)

    def test_full_vocabulary_kl_includes_tail_and_greedy_divergence_is_separate(self):
        left, prompts = self.fixture(); right = copy.deepcopy(left)
        # Change a tail logit while keeping top1 identical on shared contexts.
        right["prompt_0.decode_logits"][1, 31] = 1
        # Independent paths diverge at position 1; these logits must NOT enter KL.
        right["prompt_0.greedy_logits"][1:, 5] = 5
        right["prompt_0.greedy_ids"] = right["prompt_0.greedy_logits"].argmax(-1)
        result = comparison.compare_tensors(left, right, prompts, 32)[0]
        values = result["per_position_kl_sm70_to_sm120"]
        self.assertEqual(values[:3], [0.0, 0.0, 0.0]); self.assertGreater(values[3], 0)
        self.assertEqual(result["sm70_to_sm120"]["top1_agreement"], 1)
        self.assertEqual(result["independent_greedy"]["common_prefix_tokens"], 1)
        self.assertFalse(result["independent_greedy"]["exact_sequence_match"])
        self.assertEqual(result["teacher_forced_input_ids"], [1, 2, 3, 4])
        p = left["prompt_0.decode_logits"][1].double().softmax(-1)
        q = right["prompt_0.decode_logits"][1].double().softmax(-1)
        self.assertAlmostEqual(values[3], float((p * (p.log() - q.log())).sum()), places=12)

    def test_context_partial_vocabulary_nonfinite_and_forged_greedy_rejected(self):
        left, prompts = self.fixture()
        for kind in ("context", "partial", "nonfinite", "greedy", "endpoint"):
            right = copy.deepcopy(left)
            if kind == "context": right["prompt_0.generated_ids"][1] = 9
            if kind == "partial": right["prompt_0.decode_logits"] = right["prompt_0.decode_logits"][:, :20]
            if kind == "nonfinite": right["prompt_0.decode_logits"][0, 0] = float("nan")
            if kind == "greedy": right["prompt_0.greedy_ids"][1] = 9
            if kind == "endpoint": right["prompt_0.greedy_logits"][0, 0] = -1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                comparison.compare_tensors(left, right, prompts, 32)

    def test_artifact_hash_and_incomplete_capture_fail_closed(self):
        tensors, _ = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); artifact = directory / "sm70.safetensors"
            save_file(tensors, artifact, metadata={"status": "measured", "model_id": validate_candidate.MODEL_ID,
                "revision": validate_candidate.REVISION, "reference_sha256": "a" * 64})
            report = {"status": "measured", "state_restored": True, "all_expected_positions_compared": True,
                "independent_greedy": {"status": "complete"}, "compute_capability": [7, 0], "mode": "ordinary",
                "reference_sha256": "a" * 64, "artifact_sha256": comparison.sha256(artifact)}
            report_path = directory / "sm70.json"; report_path.write_text(json.dumps(report))
            document = {"format": comparison.FORMAT, "status": "complete", "architecture": "sm70",
                "compute_capability": [7, 0], "mode": "ordinary", "mtp": False,
                "prepared_mtp": True, "prepared_draft_tokens": 4,
                "prepared_speculative_widths": [1, 2, 3, 4, 5],
                "reference_sha256": "a" * 64, "candidate_report": report_path.name,
                "candidate_report_sha256": comparison.sha256(report_path), "artifact": artifact.name,
                "artifact_sha256": comparison.sha256(artifact)}
            path = directory / "sm70.capture.json"; path.write_text(json.dumps(document))
            comparison.load_capture(path, "sm70")
            for field, wrong in (("prepared_mtp", False), ("prepared_draft_tokens", 0),
                                 ("prepared_speculative_widths", [1, 2, 3, 4])):
                path.write_text(json.dumps({**document, field: wrong}))
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "prepared catalog"):
                    comparison.load_capture(path, "sm70")
            path.write_text(json.dumps(document))
            report["state_restored"] = False; report_path.write_text(json.dumps(report))
            document["candidate_report_sha256"] = comparison.sha256(report_path); path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                comparison.load_capture(path, "sm70")
            artifact.write_bytes(artifact.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "hash/path"):
                comparison.load_capture(path, "sm70")



    def test_checkpoint_pin_is_bound_to_file_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors")
            metadata = root / ".cache" / "huggingface" / "download"
            metadata.mkdir(parents=True)
            provenance = {"checkpoint_files": {}}
            for name in names:
                payload = json.dumps({"weight_map": {"weight": "model.safetensors"}}).encode()
                (root / name).write_bytes(payload)
                etag = (hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
                        if name == "tokenizer.json" else comparison.sha256(root / name))
                (metadata / (name + ".metadata")).write_text(validate_candidate.REVISION + "\n" + etag + "\n")
                provenance["checkpoint_files"][name] = {"revision": validate_candidate.REVISION, "etag": etag}
            provenance["config_sha256"] = comparison.sha256(root / "config.json")
            provenance["index_sha256"] = comparison.sha256(root / "model.safetensors.index.json")
            identity = comparison.checkpoint_identity(root, provenance)
            self.assertEqual(set(identity), set(names))
            (root / "model.safetensors").write_bytes(b"different weights with unchanged pin metadata")
            with self.assertRaisesRegex(ValueError, "content etag"):
                comparison.checkpoint_identity(root, provenance)

    def test_comparison_rejects_mismatched_source_checkpoint_and_reference(self):
        tensors, prompts = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            reference = directory / "reference.safetensors"; reference.write_bytes(b"synthetic")
            reference.with_suffix(".json").write_text("{}")
            left_path = directory / "left.json"; left_path.write_text("{}")
            right_path = directory / "right.json"; right_path.write_text("{}")
            args = SimpleNamespace(sm70=left_path, sm120=right_path, reference=reference,
                                   output=directory / "comparison.json")
            identity = dict(checkpoint_files_sha256={"synthetic": "b" * 64}, source_sha256="c" * 64,
                reference_sha256=comparison.sha256(reference),
                reference_report_sha256=comparison.sha256(reference.with_suffix(".json")),
                vocabulary=32, mode="ordinary", context=128, prefill_chunk_size=128, max_active=1, mtp=False)
            for field in ("source_sha256", "checkpoint_files_sha256", "reference_sha256", "mode"):
                changed = {**identity, field: "different"}
                with self.subTest(field=field), patch.object(comparison, "public_reference", return_value=(prompts, {})), \
                        patch.object(comparison, "load_capture", side_effect=[(identity, tensors), (changed, tensors)]), \
                        self.assertRaisesRegex(ValueError, "differ"):
                    comparison.compare(args)
                self.assertFalse(args.output.exists())
            with patch.object(comparison, "public_reference", return_value=(prompts, {})), \
                    patch.object(comparison, "load_capture", side_effect=[(identity, tensors), (identity, tensors)]):
                comparison.compare(args)
            report = json.loads(args.output.read_text())
            self.assertIsNone(report["numerical_pass"])
            self.assertFalse(report["mtp_tested"]); self.assertFalse(report["native_runtime_tested"])
            self.assertFalse(report["hardware_only_comparison"])
            self.assertTrue(report["prompts"][0]["independent_greedy"]["exact_sequence_match"])


if __name__ == "__main__":
    unittest.main()
