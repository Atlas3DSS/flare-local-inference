"""Lossless expansion of square-block UE8M0 scales into native MXFP8 rows."""

import torch

from ignition.components.weights import DirectWeightLoader, _aligned_range


class BlockFP8WeightLoader(DirectWeightLoader):
    @property
    def block_size(self):
        shape = self.config.quantization["weight_block_size"]
        if shape not in ([32, 32], [128, 128]):
            raise ValueError("native FP8 loading requires square block-32 or block-128 scales")
        return shape[0]

    def load_linear_group(self, prefixes, *, row_ranges=None, column_range=None):
        from b12x.gemm.blockscaled import pack_weight

        prefixes = tuple(prefixes)
        ranges = (None,) * len(prefixes) if row_ranges is None else row_ranges
        parts = [
            self._dense_blocks(prefix, rows=rows, columns=column_range)
            for prefix, rows in zip(prefixes, ranges, strict=True)
        ]
        self._session.flush()
        values = torch.cat([p[0] for p in parts]) if len(parts) > 1 else parts[0][0]
        scales = torch.cat(
            [
                p[1].repeat_interleave(self.block_size, dim=0).repeat_interleave(self.block_size // 32, dim=1)
                for p in parts
            ]
        )
        return pack_weight(values, scales, recipe="mxfp8")

    def _dense_blocks(self, prefix, *, rows=None, columns=None):
        self._require_open()
        weight, scale = self._sources[f"{prefix}.weight"], self._sources[f"{prefix}.scale"]
        block = self.block_size
        if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2:
            raise ValueError(f"{prefix}: expected native FP8 dense weights")
        n, k = weight.shape
        if scale.dtype != torch.float8_e8m0fnu or scale.shape != (
            (n + block - 1) // block,
            (k + block - 1) // block,
        ):
            raise ValueError(f"{prefix}: expected UE8M0 scales for {block}-by-{block} blocks")
        r0, r1 = _aligned_range(rows, n, block)
        c0, c1 = _aligned_range(columns, k, block)
        values = torch.empty((r1 - r0, c1 - c0), dtype=weight.dtype, device=self.device)
        scales = torch.empty(((r1 - r0) // block, (c1 - c0) // block), dtype=torch.uint8, device=self.device)
        self._copy(values, weight[r0:r1, c0:c1])
        self._copy(scales, scale[r0 // block : r1 // block, c0 // block : c1 // block].view(torch.uint8))
        return values, scales

    def load_attention_output(self, prefix, groups):
        from b12x.gemm import wo_projection

        a, sa = self._dense_blocks(prefix + ".wo_a", rows=groups)
        b, sb = self._dense_blocks(prefix + ".wo_b", columns=groups)
        self._session.flush()
        c = self.config.text
        return wo_projection.pack_weights(
            a,
            sa.view(torch.float8_e8m0fnu),
            b,
            sb.view(torch.float8_e8m0fnu),
            groups=c["o_groups"] // self.shard.world_size,
            group_width=c["num_attention_heads"] * c["head_dim"] // c["o_groups"],
            rank=c["o_lora_rank"],
            hidden=c["hidden_size"],
            block_size=(self.block_size, self.block_size),
        )
