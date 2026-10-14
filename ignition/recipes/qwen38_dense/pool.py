"""Device-resident Gated DeltaNet histories and paged full-attention KV."""

import torch

from ignition.components.gated_delta import GDNState
from ignition.components.paged_state import PagedStateStorage
from ignition.components.state import StateLayout

from .config import PAGE_TOKENS


def page_specs(config):
    size = PAGE_TOKENS * config.text["num_key_value_heads"] * config.text["head_dim"] * 2
    for index, kind in enumerate(config.layers):
        if kind == "full_attention":
            for field in ("key", "value"):
                yield f"layer{index}.{field}", (size,)


def page_bytes(config):
    return sum(shape[0] for _, shape in page_specs(config))


def continuation_layout(config):
    regions = []
    for index, kind in enumerate(config.layers):
        if kind == "linear_attention":
            regions.extend(((f"layer{index}.convolution", (10240, 3), torch.bfloat16),
                            (f"layer{index}.recurrent", (48, 128, 128), torch.float32)))
    regions.extend((name, shape, torch.uint8) for name, shape in page_specs(config))
    regions.extend((("last_hidden", (config.hidden,), torch.bfloat16),
                    ("consumed_tokens", (1,), torch.int64)))
    return StateLayout.from_specs(regions)


class ContinuationPool(PagedStateStorage):
    @classmethod
    def for_model(cls, config, slots, pages, device, **kwargs):
        return cls.allocate(continuation_layout(config), page_specs(config), slots, pages, device, **kwargs)

    def recurrent(self, layer):
        return GDNState(self.reserved_region(f"layer{layer}.convolution"),
                        self.reserved_region(f"layer{layer}.recurrent"))

    def attention(self, layer, config):
        shape = (-1, PAGE_TOKENS, config.text["num_key_value_heads"], config.text["head_dim"])
        return tuple(self.pages[f"layer{layer}.{field}"].view(torch.bfloat16).view(shape)
                     for field in ("key", "value"))
