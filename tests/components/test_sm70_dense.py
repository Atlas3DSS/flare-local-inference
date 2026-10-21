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


if __name__ == "__main__":
    unittest.main()
