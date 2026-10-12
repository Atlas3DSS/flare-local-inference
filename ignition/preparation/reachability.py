"""Retain graph-reachable CUDA allocations, treating typed numeric data as leaves."""

from bisect import bisect_right
from collections import deque
import struct

from .tensor_metadata import merge_ranges, subtract_ranges


class Reachability:
    def __init__(self, allocations, *, numeric=(), inactive=(), roots=()):
        self.allocations = {allocation.name: allocation for allocation in allocations}
        self.intervals = sorted((allocation.address, allocation.address + allocation.nbytes, allocation.name)
                                for allocation in allocations)
        self.starts = [start for start, _, _ in self.intervals]
        self.numeric = merge_ranges(numeric)
        self.numeric_starts = [start for start, _ in self.numeric]
        self.inactive = merge_ranges(inactive)
        self.inactive_starts = [start for start, _ in self.inactive]
        self.reached = set()
        self.pending = deque()
        self._blobs = set()
        self._tail = None
        for name in roots:
            self.reference({"allocation": name})

    def reference(self, reference):
        name = reference["allocation"]
        if name not in self.allocations:
            raise ValueError(f"unregistered graph allocation: {name}")
        if name not in self.reached:
            self.reached.add(name)
            self.pending.append(name)

    def pointer(self, value):
        free = bisect_right(self.inactive_starts, value) - 1
        if free >= 0 and value < self.inactive[free][1]:
            return
        index = bisect_right(self.starts, value) - 1
        if index >= 0 and value < self.intervals[index][1]:
            self.reference({"allocation": self.intervals[index][2]})

    def blob(self, data, *, numeric=()):
        data = bytes(data)
        key = data, tuple(numeric)
        if key in self._blobs:
            return
        self._blobs.add(key)
        for start, end in subtract_ranges([(0, len(data))], merge_ranges(numeric)):
            for offset in range(start, end - 7):
                self.pointer(struct.unpack_from("<Q", data, offset)[0])

    def graph(self, description, kernels):
        for node in description["nodes"]:
            if node["kind"] == "kernel":
                kernel = kernels[node["function"]]
                for blob, argument in zip(node["arguments"], kernel.arguments, strict=True):
                    for relocation in blob["relocations"]:
                        self.reference(relocation["target"])
                    self.blob(blob["data"], numeric=argument.numeric)
                # CUDA's other exported node attributes contain only numeric fields.
                for attribute in node["attributes"]:
                    for relocation in attribute["value"]["relocations"]:
                        self.reference(relocation["target"])
            elif node["kind"] == "copy":
                self.reference(node["source"])
                self.reference(node["destination"])
            elif node["kind"] == "memset":
                self.reference(node["destination"])

    def contents(self, allocation, offset, data):
        """Follow opaque words during the existing bounded tensor download."""
        import numpy as np

        address = allocation.address + offset
        if self._tail is not None:
            name, previous_end, tail = self._tail
            if name == allocation.name and previous_end == address:
                data = tail + data
                address -= len(tail)
        end = address + len(data)
        self._tail = allocation.name, end, data[-7:]
        first = max(0, bisect_right(self.numeric_starts, address) - 1)
        last = bisect_right(self.numeric_starts, end)
        spans = subtract_ranges([(address, end)], self.numeric[first:last])
        for start, stop in spans:
            for shift in range(8):
                count = (stop - start - shift) // 8
                if count <= 0:
                    continue
                words = np.frombuffer(data, dtype="<u8", count=count, offset=start - address + shift)
                candidates = words[(words >= self.starts[0]) & (words < self.intervals[-1][1])]
                for value in np.unique(candidates):
                    self.pointer(int(value))

    def take(self):
        if not self.pending:
            return None
        allocation = self.allocations[self.pending.popleft()]
        for relocation in allocation.relocations:
            self.reference({"allocation": relocation.target.allocation})
        return allocation
