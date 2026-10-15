"""Prepared graph inputs, continuation storage, and frontend syntax."""

from ignition.preparation import contract as c
from ignition.preparation import runtime_package_pb2 as p


VOCABULARY = 129280
INPUT_VOCABULARY = 129280
VISION = {'features_per_request': 0, 'max_images': 0, 'patches_per_feature': 0, 'patch_capacities': []}
FRONTEND = c.frontend(tools=p.DSML, tag_separator=" ", assistant_marker="<｜Assistant｜>",
    message_end_markers=["<｜User｜>", "<｜end▁of▁sentence｜>"], rejected_tokens=[129264],
    prefix_boundary=c.prefix_boundary(["system"]))


def prepare(signatures, buffers, pool, *, page_tokens, draft_lag, rows=1, embedding_tables=(), regions=()):
    regions = list(regions)
    for name, tensor in pool.pages.items():
        layer = int(name.split(".")[0].removeprefix("layer"))
        regions.append(c.history_region(buffers, "cache." + name, tensor, p.PAGED_LATENT,
            tokens_per_record=pool.config.text["compress_ratios"][layer]))
    inputs = {kind: c.text_inputs(kind) for kind in {s.kind for s in signatures.values()} if kind != "vision"}
    inputs["vision"] = c.vision_inputs({field for s in signatures.values() if s.kind == "vision" for field in s.inputs})
    continuation = c.state_view("state_pool", pool.layout.row_bytes)
    consumed = pool.layout.region("consumed_tokens")
    cursor = None if draft_lag is None else dict(state=c.state_view("state_pool", pool.layout.row_bytes,
        offset=consumed.offset, nbytes=consumed.nbytes), lag_tokens=draft_lag)
    pending = c.state_view("request_pending_token", 8)
    graphs = c.graph_contracts(signatures, inputs, continuation=continuation, pending=pending,
        cursor=cursor, rows=rows, embedding_tables=embedding_tables)
    for graph in graphs:
        if graph["kind"] in (p.PROPOSAL, p.VERIFICATION):
            graph["state"].extend(dict(view=c.state_view(name, tensor.numel() * tensor.element_size()),
                access=p.READ_WRITE) for name, tensor in buffers.items() if name.startswith("proposal_"))
    return c.model_contract(vocabulary=VOCABULARY, input_vocabulary=INPUT_VOCABULARY,
        page_tokens=page_tokens, page_tables=[dict(buffer="page_tables", page_offset=0),
            *([dict(buffer="draft_page_tables", page_offset=1)] if "draft_page_tables" in buffers else [])],
        regions=[dict(kind=p.CONTINUATION_STATE, storage=continuation, tokens_per_record=1, window_tokens=0), *regions],
        graphs=graphs, vision=VISION, frontend=FRONTEND)
