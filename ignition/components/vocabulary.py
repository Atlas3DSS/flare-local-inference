"""Vocabulary-sharded projection followed by native PCIe logit gathering."""

from .collectives import PreparedAllGather
from .linear import UnquantizedProjection


class PreparedVocabularyProjection:
    """Project replicated hidden rows with a contiguous vocabulary weight shard.

    Local and full-vocabulary outputs are caller-owned. Every rank participates
    in the gather; the recipe chooses which rank samples from the full logits.
    """

    def __init__(self, source, weight, local_output, output, *, runtime, session,
                 coordinator, name):
        self.projection = UnquantizedProjection(source, weight, local_output,
            session=session, name=f"{name}.projection")
        self.gather = PreparedAllGather(local_output, output, runtime=runtime,
            session=session, coordinator=coordinator, name=f"{name}.gather")
        self.output = output

    def __call__(self, source):
        return self.gather(self.projection(source))
