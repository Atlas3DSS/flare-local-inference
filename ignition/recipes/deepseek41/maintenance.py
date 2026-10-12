"""Collective admission and complete continuation snapshots."""

from ignition.components.collectives import PreparedAllReduce
from ignition.components.maintenance import PreparedMaintenance as StateMaintenance
from ignition.components.state_transfer import PreparedStateTransfer

from .pool import PAGE_TOKENS


class PreparedMaintenance:
    def __init__(self, mode, requests, pool, commands, inputs, records, frame, sequence, *,
                 runtime, coordinator, session, name, capacity, admission_buffers):
        self.pool, self.commands, self.inputs, self.requests = pool, commands, inputs, requests
        if mode == "snapshot":
            slots, epochs, count = inputs.row_slot_ids, inputs.row_generations, inputs.live_rows
        else:
            slots, epochs, count = (admission_buffers.admission_slots, admission_buffers.admission_generations,
                                   admission_buffers.admission_count)
        self.state = StateMaintenance(mode, requests, self._transfer(), slots, epochs, count,
            records, frame, sequence, world_size=runtime.world_size, capacity=capacity,
            admission_buffers=admission_buffers)
        self.state.prepare()
        self.validation = PreparedAllReduce(self.state.validation_input, self.state.validation_sum,
            runtime=runtime, coordinator=coordinator, session=session, name=name + ".validation")

    def _transfer(self):
        return PreparedStateTransfer(self.pool, self.commands, self.inputs.page_tables,
            pages=self.pool.partial_pages(), page_tokens=PAGE_TOKENS,
            state_rows=self.inputs.request_state_rows, requests=self.requests)

    def __call__(self):
        self.validation(self.state.validate())
        self.state.apply()
