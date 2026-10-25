"""Request-owned additive interventions captured inside existing model forwards."""

from dataclasses import dataclass

import torch

from ignition.components.sm70_precision import round_bf16


@dataclass(frozen=True)
class SteeringBuffers:
    addends: torch.Tensor
    phases: torch.Tensor
    prompt_tokens: torch.Tensor
    sites: tuple
    offsets: tuple
    max_sites: int

    @classmethod
    def allocate(cls, layers, hidden, capacity, max_sites, device, *, intermediate):
        sites = tuple(dict(name=f"layers.{layer}.{kind}", width=width)
                      for layer in range(layers)
                      for kind, width in (("mlp_out", hidden), ("residual", hidden), ("mlp_input", intermediate)))
        if type(max_sites) is not int or not 1 <= max_sites <= len(sites):
            raise ValueError("steering max sites must fit the prepared activation site catalog")
        offsets = [0]
        for site in sites:
            if type(site["width"]) is not int or not 1 <= site["width"] <= 65536:
                raise ValueError("steering site widths must be positive bounded integers")
            offsets.append(offsets[-1] + site["width"])
        return cls(torch.zeros((capacity, offsets[-1]), dtype=torch.float32, device=device),
                   torch.zeros((capacity, len(sites)), dtype=torch.int32, device=device),
                   torch.zeros(capacity, dtype=torch.int64, device=device), sites, tuple(offsets), max_sites)

    def site_addends(self, site):
        """Physical-request rows for one site, aliasing the ragged persistent buffer."""
        return self.addends[:, self.offsets[site]:self.offsets[site + 1]]

    def buffers(self):
        return dict(steering_addends=self.addends, steering_phases=self.phases,
                    steering_prompt_tokens=self.prompt_tokens)

    def contract(self):
        return dict(sites=list(self.sites), max_sites=self.max_sites)

    def bind(self, inputs, requests, metadata):
        return PreparedSteering(self, inputs, requests, metadata)


class PreparedSteering:
    def __init__(self, buffers, inputs, requests, metadata):
        self.buffers, self.inputs, self.requests, self.metadata = buffers, inputs, requests, metadata

    def __call__(self, value, site):
        b, m, inputs = self.buffers, self.metadata, self.inputs
        packed = m.requests
        slots = inputs.row_slot_ids[packed].long()
        safe = slots.clamp(0, b.addends.shape[0] - 1)
        live = ((m.positions >= 0) & (slots >= 0) & (slots < b.addends.shape[0])
                & (inputs.row_generations[packed] == self.requests.generation[safe]))
        prompt = m.positions < b.prompt_tokens[safe]
        phase = torch.where(prompt, 1, 2)
        enabled = live & ((b.phases[safe, site] & phase) != 0)
        delta = b.site_addends(site)[safe]
        # Select original bits for disabled sites and zero coordinates, including signed zero.
        return torch.where(enabled[:, None] & (delta != 0), round_bf16(value + delta), value)
