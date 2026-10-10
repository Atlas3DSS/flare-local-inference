"""Exact-prefix GLM state: KDA history, C4 tails, hidden output, and partial pages."""

from dataclasses import dataclass

import torch

from flare.components.recurrent import DeltaState
from flare.components.state import StateLayout, StateStorage

from .indexer import IndexState
from .mla import MLAState
from .state import DecoderState


def continuation_layout(layer_types, world_size):
    if world_size not in (2, 4):
        raise ValueError("GLM continuation storage requires TP2 or TP4")
    layer_types = tuple(layer_types)
    if not layer_types:
        raise ValueError("GLM continuation storage requires attention layers")
    heads = 64 // world_size
    regions = []
    for index, kind in enumerate(layer_types):
        if kind == "linear_attention":
            regions.extend((
                (f"layer{index}.recurrent", (heads, 128, 128), torch.float32),
                (f"layer{index}.convolution", (3 * heads * 128, 3), torch.bfloat16),
            ))
        elif kind == "indexed_attention":
            regions.extend((
                (f"layer{index}.index_tail", (2, 4, 128), torch.bfloat16),
                (f"layer{index}.latent_page", (4, 64, 528), torch.uint8),
                (f"layer{index}.index_page", (64, 132), torch.uint8),
            ))
        else:
            raise ValueError(f"unsupported canonical GLM attention type: {kind}")
    regions.extend((("last_hidden", (4096,), torch.bfloat16), ("consumed_tokens", (1,), torch.int64)))
    return StateLayout.from_specs(regions)


@dataclass(frozen=True)
class ContinuationPool(StateStorage):
    """One byte row per null/live/snapshot slot; complete KV pages remain separate.

    State slots 1–4 are live requests and slots 5 onward are immutable snapshots.
    KDA and C4 kernels bind directly to strided views into these rows. Every MLA
    snapshot has space for the complete 256-token partial page and its index page.
    """

    decoder: DecoderState
    consumed_tokens: torch.Tensor
    last_hidden: torch.Tensor

    @classmethod
    def allocate(cls, layer_types, world_size, slots, pages, device):
        layer_types = tuple(layer_types)
        if type(pages) is not int or pages <= 0:
            raise ValueError("GLM continuation storage requires positive page capacity")
        storage = StateStorage.allocate(continuation_layout(layer_types, world_size), slots, device)
        layers = []
        for index, kind in enumerate(layer_types):
            if kind == "linear_attention":
                layers.append(DeltaState(storage.region(f"layer{index}.convolution"),
                                         storage.region(f"layer{index}.recurrent")))
            else:
                cache = torch.zeros((pages * 4, 64, 528), dtype=torch.uint8, device=device)
                index_cache = torch.zeros((pages, 64, 132), dtype=torch.uint8, device=device)
                layers.append(MLAState(cache, IndexState(index_cache, storage.region(f"layer{index}.index_tail"))))
        return cls(storage.backing, storage.layout, DecoderState(tuple(layers)),
                   storage.region("consumed_tokens"), storage.region("last_hidden"))

    def clear_for_preparation(self):
        self.backing.zero_()
        for layer in self.decoder.layers:
            if isinstance(layer, MLAState):
                layer.cache.zero_()
                layer.index.cache.zero_()

    def partial_pages(self):
        """Pair each immutable page field with its live physical-page records."""
        for index, layer in enumerate(self.decoder.layers):
            if isinstance(layer, MLAState):
                pages = layer.index.cache.shape[0]
                yield f"layer{index}.latent_page", layer.cache.view(pages, -1)
                yield f"layer{index}.index_page", layer.index.cache.view(pages, -1)
