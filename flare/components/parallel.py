"""Contiguous tensor partitions for b12x TP2 and TP4 execution."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TensorParallelShard:
    rank: int
    world_size: int

    def __post_init__(self):
        if (type(self.world_size) is not int or self.world_size not in (2, 4)
                or type(self.rank) is not int or not 0 <= self.rank < self.world_size):
            raise ValueError("tensor parallelism requires a valid rank in TP2 or TP4")

    def partition(self, extent: int, *, alignment: int = 1) -> tuple[int, int]:
        if (type(extent) is not int or extent <= 0 or type(alignment) is not int
                or alignment <= 0 or extent % (self.world_size * alignment)):
            raise ValueError("tensor extent must divide into aligned rank partitions")
        width = extent // self.world_size
        return self.rank * width, (self.rank + 1) * width
