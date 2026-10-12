"""Shared compressed pages and complete per-request continuation state."""

from dataclasses import dataclass

import torch

from ignition.components.state import StateLayout, StateStorage

PAGE_TOKENS = 128
SWA_RECORD_BYTES = 528
COMPRESSED_RECORD_BYTES = 288
INDEX_RECORD_BYTES = 68


def page_specs(config):
    """One logical page covers the same token span in every source region."""
    for source in config.sources:
        records = source.records(PAGE_TOKENS)
        yield f"layer{source.layer}.latent_page", (records * COMPRESSED_RECORD_BYTES,)
        yield f"layer{source.layer}.index_page", (records * INDEX_RECORD_BYTES,)


def page_bytes(config):
    return sum(shape[0] for _, shape in page_specs(config))


def continuation_layout(config):
    regions = [(f"layer{i}.window", (config.text["sliding_window"] * SWA_RECORD_BYTES,), torch.uint8)
               for i in range(config.layers)]
    for source in config.sources:
        if source.ratio == 2:
            regions.extend((
                (f"layer{source.layer}.pending_values", (512,), torch.float32),
                (f"layer{source.layer}.pending_gates", (512,), torch.float32),
                (f"layer{source.layer}.pending_position", (1,), torch.int64),
            ))
    regions.extend((name, shape, torch.uint8) for name, shape in page_specs(config))
    regions.extend((
        ("last_hidden", (config.text["hidden_size"],), torch.bfloat16),
        ("consumed_tokens", (1,), torch.int64),
    ))
    return StateLayout.from_specs(regions)


@dataclass(frozen=True)
class ContinuationPool(StateStorage):
    config: object
    pages: dict[str, torch.Tensor]
    window_backing: torch.Tensor

    @classmethod
    def allocate(cls, config, slots, pages, device, *, storage=None, maximum_pages=None):
        if type(pages) is not int or pages <= 0:
            raise ValueError("DeepSeek cache requires positive logical page capacity")
        backing = StateStorage.allocate(continuation_layout(config), slots, device, storage=storage)
        maximum = pages if storage is None else maximum_pages
        if maximum is None or maximum < pages:
            raise ValueError("virtual cache extent must cover prepared physical pages")
        regions = {}
        for name, shape in page_specs(config):
            regions[name] = (torch.zeros((pages, *shape), dtype=torch.uint8, device=device)
                if storage is None else storage.device_allocate(name, (maximum, *shape), torch.uint8,
                    group="KV_PAGES", maximum=maximum, units=pages))
        backing.memory_capacity[1] = pages
        window_backing = backing.backing
        if storage is not None:
            from b12x.sequence._shared.disk_table import _tensor_from_pointer

            region = next(region for region in storage.dynamic_regions
                          if region.address == backing.backing.data_ptr())
            window_backing = _tensor_from_pointer(region.address,
                shape=(region.maximum, backing.layout.row_bytes), dtype=torch.uint8,
                device=backing.backing.device, nbytes=region.maximum * backing.layout.row_bytes)
        return cls(backing.backing, backing.layout, backing.memory_capacity, config, regions, window_backing)

    def window(self, layer):
        """Expose the reserved row extent to cache writers; admission bounds physical rows."""
        region = self.layout.region(f"layer{layer}.window")
        return self.window_backing[:, region.offset:region.offset + region.nbytes]

    @property
    def last_hidden(self):
        return self.region("last_hidden")

    def partial_pages(self):
        return self.pages.items()

    def clear_for_preparation(self):
        self.backing.zero_()
        count = int(self.memory_capacity[1].item())
        for value in self.pages.values():
            value[:count].zero_()

