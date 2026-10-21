"""Prepared b12x embedding lookup with fixed output and device live counts."""

from dataclasses import dataclass

import torch
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared
from b12x.sequence import embedding

from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import declare
from .collectives import PreparedChunkedAllGather
from .native_operator import native_contract, native_plan


_SM70_COMPONENT = "flare.sm70.embedding"


@dataclass(frozen=True)
class _SM70EmbeddingQuery:
    rows: int
    vocabulary: int
    width: int
    weight_dtype: str
    output_dtype: str
    device_count: bool


_SM70_CONTRACT = native_contract(component_id=_SM70_COMPONENT, query_type=_SM70EmbeddingQuery,
                                  tactics=lambda query: ("gather",))


@dataclass(frozen=True)
class _SM70Embedding:
    query: _SM70EmbeddingQuery

    def run(self, weight_bytes, ids, out, num_rows):
        weight = weight_bytes.view(getattr(torch, self.query.weight_dtype))
        if num_rows is None:
            out.copy_(torch.index_select(weight, 0, ids.reshape(-1)).view_as(out))
            return out
        # Match b12x's device errors for invalid counts and live token IDs.
        # Inactive IDs may be arbitrary and must never index the table.
        count = num_rows.reshape(())
        torch._assert_async((count >= 0) & (count <= ids.numel()),
                            "embedding live count exceeds its prepared capacity")
        active = torch.arange(ids.numel(), device=ids.device) < count
        safe_ids = torch.where(active, ids.reshape(-1), 0)
        selected = torch.index_select(weight, 0, safe_ids)
        out.copy_(torch.where(active[:, None], selected, 0).view_as(out))
        return out


@torch.library.custom_op("flare::sm70_embedding_out", mutates_args=("output",))
def _sm70_embedding_out(weight_bytes: torch.Tensor, ids: torch.Tensor, output: torch.Tensor,
                         num_rows: torch.Tensor | None, handle: int) -> None:
    state = require_prepared(plan_from_handle(handle), _SM70_COMPONENT, weight_bytes.device)
    state.run(weight_bytes, ids, output, num_rows)


@_sm70_embedding_out.register_fake
def _fake_sm70_embedding_out(weight_bytes, ids, output, num_rows, handle):
    pass


class PreparedEmbedding:
    """Gather token rows into caller storage and zero inactive output rows.

    Token IDs and the optional live count remain device inputs during graph
    replay. The fixed b12x kernel has no workspace or tuning candidates.
    """

    def __init__(self, weight, ids, output, *, session, name, num_rows=None):
        self.weight, self.ids, self.output, self.num_rows = weight, ids, output, num_rows
        self.sm70 = weight.device.type == "cuda" and torch.cuda.get_device_capability(weight.device) == (7, 0)
        if self.sm70:
            dtypes = (torch.bfloat16, torch.float16, torch.float32)
            if (weight.ndim != 2 or min(weight.shape) <= 0 or not weight.is_contiguous()
                    or weight.dtype not in dtypes or ids.dtype not in (torch.int32, torch.int64)
                    or not ids.is_contiguous() or ids.numel() == 0 or ids.numel() >= 2**31
                    or output.shape != (*ids.shape, weight.shape[1]) or output.dtype not in dtypes
                    or not output.is_contiguous() or ids.device != weight.device or output.device != weight.device):
                raise ValueError("SM70 embeddings require a floating table, contiguous integer IDs and matching floating output")
            if num_rows is not None and (num_rows.dtype != torch.int32 or num_rows.numel() != 1
                    or num_rows.device != weight.device or not num_rows.is_contiguous()):
                raise ValueError("embedding live count must be a device Int32 scalar")
            if any(torch._C._overlaps(output, tensor) for tensor in (weight, ids, num_rows) if tensor is not None):
                raise ValueError("embedding output must not alias its inputs")
            # Only selected rows expand; the complete checkpoint table stays BF16.
            # Bytes also keep BF16 storage outside Inductor's SM70 arithmetic checks.
            self.weight_bytes = weight.view(torch.uint8)
            record_numeric_tensor(self.weight_bytes)
            query = _SM70EmbeddingQuery(ids.numel(), weight.shape[0], weight.shape[1],
                str(weight.dtype).removeprefix("torch."), str(output.dtype).removeprefix("torch."), num_rows is not None)
            self.plan = native_plan(contract=_SM70_CONTRACT, query=query, device=weight.device,
                scratch_bytes=lambda tactic: 0, materialize=lambda tactic: _SM70Embedding(query))

            def prepare_sm70(state):
                prime_ids = torch.zeros_like(ids)
                prime_count = None if num_rows is None else torch.zeros_like(num_rows)
                return PreparedCall(run=lambda: state.run(self.weight_bytes, prime_ids, output, prime_count),
                                    owners=(prime_ids, prime_count))

            declare(session, (self.plan.request(name=name, prepare_call=prepare_sm70),))
            return
        self.arguments = {"out": output, "num_rows": num_rows}
        self.plan = embedding.plan(embedding.query_from_call(weight, ids, **self.arguments), device=weight.device)
        def prepare(state):
            prime_ids = torch.zeros_like(ids)
            arguments = {**self.arguments, "num_rows": None if num_rows is None else torch.zeros_like(num_rows)}
            return PreparedCall(run=lambda: state.run(weight, prime_ids, **arguments))

        declare(session, (self.plan.request(name=name, prepare_call=prepare),))

    def __call__(self):
        if self.sm70:
            _sm70_embedding_out(self.weight_bytes, self.ids, self.output, self.num_rows, self.plan.handle)
            return self.output
        if self.num_rows is not None:
            self.output.zero_()
        return embedding.run(self.weight, self.ids, plan=self.plan, **self.arguments)


class PreparedShardedEmbedding:
    """Gather column-sharded vocabulary rows through a prepared collective."""

    def __init__(self, weight, ids, local_output, output, *, runtime, coordinator, session, name, num_rows=None):
        self.lookup = PreparedEmbedding(weight, ids, local_output, session=session, name=name + ".lookup",
                                        num_rows=num_rows)
        self.gather = PreparedChunkedAllGather(local_output, output, runtime=runtime, coordinator=coordinator,
                                               session=session, name=name + ".gather")

    def __call__(self):
        return self.gather(self.lookup())
