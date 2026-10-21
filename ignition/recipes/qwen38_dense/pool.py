"""Device-resident Gated DeltaNet histories and paged full-attention KV."""

import torch

from ignition.components.gated_delta import GDNState
from ignition.components.paged_state import PagedStateStorage
from ignition.components.state import StateLayout

from .config import PAGE_TOKENS


def page_specs(config, *, element_bytes=1, mtp=False):
    size = PAGE_TOKENS * config.text["num_key_value_heads"] * config.text["head_dim"] * element_bytes
    for index, kind in enumerate(config.layers):
        if kind == "full_attention":
            for field in ("key", "value"):
                yield f"layer{index}.{field}", (size,)
    if mtp:
        for field in ("key", "value"):
            yield f"draft.{field}", (size,)


def page_bytes(config, *, element_bytes=1, mtp=False):
    return sum(shape[0] for _, shape in page_specs(config, element_bytes=element_bytes, mtp=mtp))


def continuation_layout(config, *, dtype=torch.bfloat16, element_bytes=1, draft_tokens=0):
    if type(draft_tokens) is not int or not 0 <= draft_tokens <= 3:
        raise ValueError("dense MTP supports zero through three draft tokens")
    regions = []
    for index, kind in enumerate(config.layers):
        if kind == "linear_attention":
            regions.extend(((f"layer{index}.convolution", (10240, 3 + draft_tokens), dtype),
                            (f"layer{index}.recurrent", (48, 128, 128), torch.float32)))
    regions.extend((name, shape, torch.uint8) for name, shape in page_specs(
        config, element_bytes=element_bytes, mtp=draft_tokens > 0))
    regions.extend((("last_hidden", (config.hidden,), dtype),
                    ("consumed_tokens", (1,), torch.int64)))
    return StateLayout.from_specs(regions)


class ContinuationPool(PagedStateStorage):
    @classmethod
    def for_model(cls, config, slots, pages, device, *, draft_tokens=0, **kwargs):
        from ignition.components.sm70_projection import is_sm70
        sm70 = is_sm70(device)
        dtype, element_bytes = (torch.float32, 2) if sm70 else (torch.bfloat16, 1)
        return cls.allocate(continuation_layout(config, dtype=dtype, element_bytes=element_bytes,
                                               draft_tokens=draft_tokens),
                            page_specs(config, element_bytes=element_bytes, mtp=draft_tokens > 0),
                            slots, pages, device, **kwargs)

    def recurrent(self, layer):
        return GDNState(self.reserved_region(f"layer{layer}.convolution"),
                        self.reserved_region(f"layer{layer}.recurrent"))

    def attention(self, layer, config):
        shape = (-1, PAGE_TOKENS, config.text["num_key_value_heads"], config.text["head_dim"])
        dtype = torch.uint16 if self.last_hidden.dtype == torch.float32 else torch.float8_e4m3fn
        prefix = "draft" if layer == "draft" else f"layer{layer}"
        return tuple(self.pages[f"{prefix}.{field}"].view(dtype).view(shape)
                     for field in ("key", "value"))
