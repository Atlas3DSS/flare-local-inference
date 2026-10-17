"""KDA continuation rows and paged, unrotated MLA latent records."""

import torch

from ignition.components.paged_state import PagedStateStorage
from ignition.components.recurrent import DeltaState
from ignition.components.state import StateLayout

from .config import PAGE_TOKENS


def page_specs(config):
    for index, kind in enumerate(config.layers):
        if kind == "mla":
            yield f"layer{index}.latent", (PAGE_TOKENS * 576,)
    if config.draft is not None:
        for index in range(config.draft.layers):
            yield f"draft{index}.latent", (PAGE_TOKENS * 576,)


def page_bytes(config):
    return sum(shape[0] for _, shape in page_specs(config))


def continuation_layout(config, world_size):
    heads = config.text["linear_attn_config"]["num_heads"] // world_size
    regions = []
    for index, kind in enumerate(config.layers):
        if kind == "kda":
            columns = 3 if config.draft is None else 3 + config.draft.block_size - 1
            regions.extend(((f"layer{index}.convolution", (3 * heads * 128, columns), torch.bfloat16),
                            (f"layer{index}.recurrent", (heads, 128, 128), torch.float32)))
    regions.extend((name, shape, torch.uint8) for name, shape in page_specs(config))
    regions.extend((("last_hidden", (config.hidden,), torch.bfloat16), ("consumed_tokens", (1,), torch.int64)))
    return StateLayout.from_specs(regions)


class ContinuationPool(PagedStateStorage):
    @classmethod
    def for_model(cls, config, world_size, slots, pages, device, **kwargs):
        return cls.allocate(continuation_layout(config, world_size), page_specs(config), slots, pages, device, **kwargs)

    def recurrent(self, layer):
        return DeltaState(self.reserved_region(f"layer{layer}.convolution"), self.reserved_region(f"layer{layer}.recurrent"))

    def attention(self, layer):
        return self.pages[f"layer{layer}.latent"].view(torch.float8_e4m3fn).view(-1, PAGE_TOKENS, 576)

    def draft_attention(self, layer):
        return self.pages[f"draft{layer}.latent"].view(torch.float8_e4m3fn).view(-1, PAGE_TOKENS, 576)
