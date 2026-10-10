"""GLM continuation transitions with b12x rank agreement before mutation."""

from flare.components.collectives import PreparedAllReduce
from flare.components.maintenance import PreparedMaintenance as StateMaintenance
from flare.components.state_transfer import PreparedStateTransfer


class PreparedMaintenance:
    def __init__(self, mode, requests, pool, commands, inputs, records, frame, sequence, *,
                 runtime, coordinator, session, name, capacity=4, admission_buffers=None):
        transfer = PreparedStateTransfer(pool, commands, inputs.page_tables, pages=pool.partial_pages(), page_tokens=256)
        if mode == "snapshot":
            slots, epochs, count = inputs.row_slot_ids, inputs.row_generations, inputs.live_rows
        elif admission_buffers is not None:
            slots, epochs, count = (admission_buffers.admission_slots, admission_buffers.admission_generations,
                                   admission_buffers.admission_count)
        else:
            raise ValueError("GLM init and restore require request admission buffers")
        self.state = StateMaintenance(mode, requests, transfer, slots, epochs, count, records, frame, sequence,
            world_size=runtime.world_size, capacity=capacity, admission_buffers=admission_buffers)
        self.state.prepare()
        self.validation = PreparedAllReduce(self.state.validation_input, self.state.validation_sum,
            runtime=runtime, coordinator=coordinator, session=session, name=f"{name}.validation")

    def __call__(self):
        self.validation(self.state.validate())
        self.state.apply()
