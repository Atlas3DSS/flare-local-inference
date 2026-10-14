"""Run one rank's share of rank zero's direct steps: copy inputs, then replay a graph or call its forward.

A step is a plain message: the graph's name, the input byte ranges to write
(the same partial uploads Helios writes, leaving every other byte of the
resident input buffers unchanged), and the frame index and submission sequence
the graph's completion record carries. Rank zero decides every step and sends
it to each peer over the rank's executor connection before running it itself,
so every rank executes the same graphs in the same order; the collectives
inside each graph join the ranks' GPU work. A rank finishes each step before
taking the next, which keeps its pinned staging and the next step's uploads
ordered without events.
"""

import struct

import torch

from ignition.preparation.host_ranks import receive_object, send_object

_RESERVED = ("frame_index", "submission_sequence")


class StepRunner:
    """Execute steps on one rank's prepared model."""

    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda", model.device)
        self.inputs = {name: frozenset((*signature.inputs, *_RESERVED))
                       for name, signature in model.signatures.items()}
        self.staging = {}
        # Collective runtimes that report transport failures only when asked, after each step.
        health, seen = [], set()
        for resource in model.resources:
            if callable(getattr(resource, "check_health", None)) and id(resource) not in seen:
                seen.add(id(resource))
                health.append(resource)
        self.health = tuple(health)
        torch.cuda.set_device(self.device)
        torch.cuda.synchronize(self.device)
        self.stream = torch.cuda.Stream(self.device)
        self.started = torch.cuda.Event(enable_timing=True)
        self.finished = torch.cuda.Event(enable_timing=True)

    def _upload(self, name, offset, data):
        target = self.model.buffers[name].view(-1)
        size = target.element_size()
        if offset % size or len(data) % size or offset + len(data) > target.numel() * size or not data:
            raise ValueError(f"misaligned or out-of-range upload to {name}")
        staging = self.staging.get(name)
        if staging is None:
            staging = self.staging[name] = torch.empty(target.numel(), dtype=target.dtype, pin_memory=True)
        start, count = offset // size, len(data) // size
        staging[start:start + count].copy_(torch.frombuffer(bytearray(data), dtype=target.dtype))
        target[start:start + count].copy_(staging[start:start + count], non_blocking=True)

    def run(self, step):
        """Execute one step and return its GPU milliseconds, from the first upload through the graph."""
        name = step["graph"]
        allowed = self.inputs.get(name)
        if allowed is None:
            raise ValueError(f"step names unknown graph {name!r}")
        with torch.cuda.stream(self.stream), torch.no_grad():
            self.started.record(self.stream)
            for buffer, offset, data in step["uploads"]:
                if buffer not in allowed or buffer in _RESERVED:
                    raise ValueError(f"graph {name} does not bind input {buffer}")
                self._upload(buffer, offset, data)
            self._upload("frame_index", 0, struct.pack("<i", step["frame"]))
            self._upload("submission_sequence", 0, struct.pack("<q", step["sequence"]))
            self.model.graphs[name].replay()
            self.finished.record(self.stream)
        self.finished.synchronize()
        for runtime in self.health:
            runtime.check_health()
        return self.started.elapsed_time(self.finished)

    def records(self, frame):
        """The completion records the last step wrote to `frame`, one row per request slot."""
        return self.model.buffers["completion_records"][frame].cpu().numpy()

    def close(self):
        torch.cuda.synchronize(self.device)


class RankSteps:
    """Rank zero's side: send every step to each peer, then run it here."""

    def __init__(self, model, peers):
        self.runner = StepRunner(model)
        self.peers = tuple(peers)
        self.stopped = False

    def run(self, step):
        for peer in self.peers:
            send_object(peer, step)
        return self.runner.run(step)

    def records(self, frame):
        return self.runner.records(frame)

    def stop(self):
        """End every peer's step loop; peers then close their models with this rank."""
        if self.stopped:
            return
        self.stopped = True
        for peer in self.peers:
            send_object(peer, None)
        self.runner.close()


def serve_peer(model, connection):
    """Execute rank zero's steps, in order, until it sends the stop message."""
    runner = StepRunner(model)
    while (step := receive_object(connection)) is not None:
        runner.run(step)
    runner.close()
