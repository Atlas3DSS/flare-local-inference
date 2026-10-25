"""CPU checks for bounded, exhaustive physical-pool cleanup validation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


spec = importlib.util.spec_from_file_location(
    "candidate_pool_validator", Path(__file__).resolve().parents[1] / "tools/validate_candidate.py")
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


class PoolValidationTests(unittest.TestCase):
    def pool(self):
        # Nonzero reserved tails are deliberately outside physical capacity.
        backing = torch.ones((4, 17), dtype=torch.uint8)
        backing[:2].zero_()
        pages = torch.ones((5, 11), dtype=torch.uint8)
        pages[:3].zero_()
        return SimpleNamespace(memory_capacity=torch.tensor([2, 3]), backing=backing, pages={"key": pages})

    def test_reductions_are_bounded_and_cover_only_physical_prefix(self):
        pool, sizes = self.pool(), []
        count = torch.count_nonzero

        def checked(tensor):
            sizes.append(tensor.numel())
            self.assertLessEqual(tensor.numel(), 7)
            return count(tensor)

        with patch.object(torch, "count_nonzero", side_effect=checked):
            candidate._assert_pool_cleared(pool, chunk_bytes=7)
        self.assertEqual(sum(sizes), 2 * 17 + 3 * 11)
        self.assertGreater(len(sizes), 2)

    def test_last_partial_chunk_corruption_is_detected(self):
        for name in ("continuations", "key"):
            pool = self.pool()
            if name == "continuations":
                pool.backing[1, -1] = 1
            else:
                pool.pages[name][2, -1] = 1
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, name):
                candidate._assert_pool_cleared(pool, chunk_bytes=7)

    def test_invalid_extents_or_noncontiguous_backing_fail_closed(self):
        for mutation in (lambda p: p.memory_capacity.fill_(99),
                         lambda p: setattr(p, "backing", p.backing[:, ::2])):
            pool = self.pool()
            mutation(pool)
            with self.assertRaises(ValueError):
                candidate._assert_pool_cleared(pool, chunk_bytes=7)
        with self.assertRaises(ValueError):
            candidate._assert_pool_cleared(self.pool(), chunk_bytes=0)


if __name__ == "__main__":
    unittest.main()
