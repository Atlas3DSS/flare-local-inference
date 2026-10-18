"""KDA continuation rows and paged, unrotated E4M3 MLA latent records."""

import torch

from ignition.components.paged_state import PagedStateStorage
from ignition.components.recurrent import DeltaState
from ignition.components.state import StateLayout

from .config import PAGE_TOKENS, DSparkConfig
from .dcp import RECORD_BYTES

# Records of each DFlash2 sliding layer's ring. The block's first query attends the window's W - 1 context
# positions before it, which must never share a slot.
DRAFT_RING = 4096


def draft_layers(config):
    """DFlash2's sliding-window and full-attention draft layers; none without DFlash2."""
    if config.draft is None or isinstance(config.draft, DSparkConfig):
        return (), ()
    kinds = config.draft.text["layer_types"]
    if config.draft.text["sliding_window"] - 1 > DRAFT_RING:
        raise ValueError("DFlash2 sliding windows must fit the draft ring")
    return (tuple(i for i, kind in enumerate(kinds) if kind == "sliding_attention"),
            tuple(i for i, kind in enumerate(kinds) if kind == "full_attention"))


def page_specs(config, context_ranks=1):
    """Per-page storage; `context_ranks` ranks share each page's positions (decode context parallelism)."""
    # E4M3 records with a unit scale, as vLLM's Kimi K3 serving stores its latent cache (--kv-cache-dtype fp8).
    for index, kind in enumerate(config.layers):
        if kind == "mla":
            yield f"layer{index}.latent", (PAGE_TOKENS * RECORD_BYTES,)
    # DFlash2's full-attention layers attend every position on every rank, as vLLM replicates draft caches: each
    # page holds the E4M3 records of all the positions it covers. Sliding layers keep rings in continuation rows.
    for index in draft_layers(config)[1]:
        yield f"draft{index}.latent", (PAGE_TOKENS * context_ranks * RECORD_BYTES,)


def page_bytes(config, context_ranks=1):
    return sum(shape[0] for _, shape in page_specs(config, context_ranks))


def continuation_layout(config, world_size, *, speculative_tokens=0, context_ranks=1):
    """One continuation row; speculation widens each KDA convolution history by the verified drafts, DSpark
    adds every draft layer's key and value window and DFlash2 its sliding layers' rings, so restored
    continuations resume drafting. A copy of each page region holds a snapshot's partial page."""
    heads = config.text["linear_attn_config"]["num_heads"] // world_size
    if config.draft is not None and not isinstance(config.draft, DSparkConfig):
        speculative_tokens = config.draft.block_size - 1
    regions = []
    for index, kind in enumerate(config.layers):
        if kind == "kda":
            regions.extend(((f"layer{index}.convolution", (3 * heads * 128, 3 + speculative_tokens), torch.bfloat16),
                            (f"layer{index}.recurrent", (heads, 128, 128), torch.float32)))
    regions.extend((name, shape, torch.uint8) for name, shape in page_specs(config, context_ranks))
    regions.extend((f"draft{index}.ring", (DRAFT_RING, RECORD_BYTES), torch.uint8) for index in draft_layers(config)[0])
    if speculative_tokens and isinstance(config.draft, DSparkConfig):
        d = config.draft
        for layer in range(d.layers):
            for kind in ("key", "value"):
                regions.append((f"draft{layer}.{kind}", (d.window, d.kv_heads // world_size, d.head_dim),
                                torch.bfloat16))
    regions.extend((("last_hidden", (config.hidden,), torch.bfloat16), ("consumed_tokens", (1,), torch.int64)))
    return StateLayout.from_specs(regions)


class ContinuationPool(PagedStateStorage):
    @classmethod
    def for_model(cls, config, world_size, slots, pages, device, *, speculative_tokens=0, context_ranks=1, **kwargs):
        return cls.allocate(continuation_layout(config, world_size, speculative_tokens=speculative_tokens,
                                                context_ranks=context_ranks),
                            page_specs(config, context_ranks), slots, pages, device, **kwargs)

    def recurrent(self, layer):
        return DeltaState(self.reserved_region(f"layer{layer}.convolution"), self.reserved_region(f"layer{layer}.recurrent"))

    def attention(self, layer):
        return self.pages[f"layer{layer}.latent"].view(torch.float8_e4m3fn).view(-1, PAGE_TOKENS, RECORD_BYTES)

    def draft_attention(self, layer):
        """DFlash2 layer `layer`'s E4M3 records: a sliding layer's ring [rows, ring, 576] in every continuation
        row, else [pages, page positions, 576]."""
        name = f"draft{layer}.latent"
        if name in self.pages:
            pages = self.pages[name].view(torch.float8_e4m3fn)
            return pages.view(pages.shape[0], -1, RECORD_BYTES)
        return self.reserved_region(f"draft{layer}.ring").view(torch.float8_e4m3fn)

    def share_pages(self, storage, group):
        """Every rank's address of each MLA layer's page pool, {layer: [world] int64}, mapped through CUDA IPC.

        Context-parallel prefill reads earlier positions from the ranks that own them.
        """
        names = [name for name in self.pages if name.startswith("layer") and name.endswith(".latent")]
        regions = {region.address: region for region in storage.dynamic_regions}
        addresses = storage.peer_storage.share_regions(group, [regions[self.pages[name].data_ptr()] for name in names])
        return {int(name.split(".")[0].removeprefix("layer")): torch.tensor(pointers, dtype=torch.int64,
                device=self.backing.device) for name, pointers in zip(names, addresses, strict=True)}
