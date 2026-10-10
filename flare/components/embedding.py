"""Prepared b12x embedding lookup with fixed output and device live counts."""

from b12x.preparation import PreparedCall
from b12x.sequence import embedding


class PreparedEmbedding:
    """Gather token rows into caller storage and zero inactive output rows.

    Token IDs and the optional live count remain device inputs during graph
    replay. The fixed b12x kernel has no workspace or tuning candidates.
    """

    def __init__(self, weight, ids, output, *, session, name, num_rows=None):
        self.weight, self.ids, self.output, self.num_rows = weight, ids, output, num_rows
        self.arguments = {"out": output, "num_rows": num_rows}
        self.plan = embedding.plan(embedding.query_from_call(weight, ids, **self.arguments), device=weight.device)
        session.prepare((self.plan.request(name=name, prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(weight, ids, **self.arguments))),))

    def __call__(self):
        if self.num_rows is not None:
            self.output.zero_()
        return embedding.run(self.weight, self.ids, plan=self.plan, **self.arguments)
