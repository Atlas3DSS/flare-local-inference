"""Dense Qwen text inputs, recurrent continuations, and paged attention storage."""

from ignition.preparation import contract as c
from ignition.preparation import runtime_package_pb2 as p

from .config import PAGE_TOKENS


FRONTEND = c.frontend(tools=p.QWEN_CODER, wrapping_newlines=1, terminal_markers=["<tool_call>"],
    assistant_marker="<|im_start|>assistant\n", message_end_markers=["<|im_end|>"],
    prefix_boundary=c.prefix_boundary(["system", "developer"], append_role="user", minimum_messages=0,
        required_start="<|im_start|>system\n", required_marker="<|im_end|>"))


def prepare(signatures, buffers, pool, *, vocabulary, steering=None):
    inputs = {kind: c.text_inputs(kind) for kind in {s.kind for s in signatures.values()}}
    inputs["decode"].pop("token_ids", None)
    if "speculative" in inputs:
        inputs["speculative"].pop("token_ids", None)
    if steering is not None:
        for kind in ("init", "restore"):
            inputs[kind].update({name: c.binding(name, source, p.REQUEST_ROW) for name, source in (
                ("steering_addends", p.STEERING_ADDENDS), ("steering_phases", p.STEERING_PHASES),
                ("steering_prompt_tokens", p.STEERING_PROMPT_TOKENS))})
    continuation = c.state_view("state_pool", pool.layout.row_bytes)
    pending = c.state_view("request_pending_token", 8)
    regions = [c.history_region(buffers, f"cache.{name}", tensor, p.PAGED_KEY_VALUE)
               for name, tensor in pool.pages.items()]
    consumed = pool.layout.region("consumed_tokens")
    cursor = (dict(state=c.state_view("state_pool", pool.layout.row_bytes,
        offset=consumed.offset, nbytes=consumed.nbytes), lag_tokens=1)
        if "draft_page_tables" in buffers else None)
    contract = c.model_contract(vocabulary=vocabulary, page_tokens=PAGE_TOKENS,
        page_tables=[dict(buffer="page_tables", page_offset=0),
            *([dict(buffer="draft_page_tables", page_offset=1)] if cursor is not None else [])],
        regions=[dict(kind=p.CONTINUATION_STATE, storage=continuation, tokens_per_record=1, window_tokens=0),
                 *regions],
        # Replay-commit: speculative rounds keep one committed continuation row per request (verification writes
        # no candidate rows), so Helios plans and packs a single state row whatever the verification width.
        graphs=c.graph_contracts(signatures, inputs, continuation=continuation, pending=pending,
                                cursor=cursor, rows=1),
        frontend=FRONTEND)
    if steering is not None:
        contract["steering"] = steering.contract()
        for graph in contract["graphs"]:
            if graph["kind"] in (p.PREFILL, p.DECODE, p.SPECULATIVE):
                graph["state"].extend(dict(view=c.state_view(name, tensor[0].numel() * tensor.element_size()),
                                           access=p.READ) for name, tensor in steering.buffers().items())
    return contract
