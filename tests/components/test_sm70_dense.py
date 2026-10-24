"""CPU checks for the reviewed dense plan's admission and row independence."""

from dataclasses import dataclass, replace
import unittest

from ignition.components.sm70_dense import tactic


@dataclass(frozen=True)
class Query:
    kind: str = "dense"
    rows: int = 1
    in_features: int = 5120
    padded_in_features: int = 5120
    out_features: int = 1024
    weight_dtype: str = "bfloat16"
    input_dtype: str = "float32"
    output_dtype: str = "float32"
    has_bias: bool = False
    row_invariant: bool = True


class DensePlanTest(unittest.TestCase):
    def test_every_matrix_has_one_tactic_across_decode_verification_and_prefill(self):
        for k, n, expected in ((5120, 96, 22), (10240, 5120, 42), (5120, 12288, 36),
                               (5120, 1024, 54), (6144, 5120, 43),
                               (5120, 17408, 36), (17408, 5120, 42)):
            for rows in (*range(1, 21), 21, 41, 128, 4096):
                for row_invariant in (False, True):
                    query = Query(rows=rows, in_features=k, padded_in_features=k,
                                  out_features=n, row_invariant=row_invariant)
                    self.assertEqual(tactic(query), ("row_dense", expected))

    def test_unqualified_precision_bias_padding_and_shapes_are_not_admitted(self):
        for changes in ({"kind": "nvfp4"}, {"kind": "tensor_fp8"},
                        {"weight_dtype": "float16"}, {"weight_dtype": "float32"},
                        {"input_dtype": "bfloat16"}, {"output_dtype": "bfloat16"},
                        {"input_dtype": "float16"}, {"output_dtype": "float16"},
                        {"has_bias": True}, {"padded_in_features": 5376},
                        {"out_features": 14336}, {"out_features": 34816},
                        {"out_features": 384}):
            with self.subTest(changes=changes):
                self.assertIsNone(tactic(replace(Query(), **changes)))


    def test_single_pass_instantiations_are_compiled_defaults_in_the_reviewed_space(self):
        # astra-21: every multi-row (R, W) choice is a compiled DENSE_ROWS_CONFIGS entry of its tactic, inside the
        # reviewed template space (R1 kernels C = 0, R3 channel-blocked C >= 2), only for prefill/absorption tactics.
        import re
        from pathlib import Path
        from ignition.components.sm70_dense import ROWS

        source = (Path(__file__).resolve().parents[2] / "ignition/components/sm70_dense.cu").read_text()
        default = re.search(r"#define DENSE_ROWS_CONFIGS\(X\) ?(.*)\n", source).group(1)
        blocked = re.search(r"#define DENSE_CB_CONFIGS\(X\) ?(.*)\n", source).group(1)
        compiled = {(int(t), int(r), int(w), 0) for t, r, w in re.findall(r"X\((\d+),(\d+),(\d+)\)", default)}
        compiled |= {(int(t), int(r), int(w), int(c))
                     for t, r, w, c in re.findall(r"X\((\d+),(\d+),(\d+),(\d+)\)", blocked)}
        self.assertLessEqual(set(ROWS), {22, 42, 54})
        self.assertEqual(compiled, {(tactic, *choice) for tactic, choice in ROWS.items()})
        for tactic, (r, w, c) in ROWS.items():
            if c:
                self.assertTrue(r in (2, 4, 8) and c in (2, 4, 8) and (w == 4 if tactic == 54 else w in (2, 4, 8)))
            elif tactic == 54:
                self.assertTrue(w == 4 and r in (8, 16, 32))
            else:
                self.assertTrue(r in (4, 8, 16) and w in (4, 8))


if __name__ == "__main__":
    unittest.main()
