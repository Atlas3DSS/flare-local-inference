# CUDA graph node attribution

The capture mapper in [`qualify/graph_nodes.py`](../../qualify/graph_nodes.py)
assigns nodes to operator scopes using CUDA graph queries and CUPTI node IDs.
The analyzer in [`qualify/kernel_timeline.py`](../../qualify/kernel_timeline.py)
joins captured IDs to executable nodes through Nsight's explicit
`CUDA_GRAPH_NODE_EVENTS.originalGraphNodeId` relationships. Pointer values,
kernel names, and launch order are not used as node identities.

With Nsight Systems 2026.1.3 and CUDA 13 on SM120, a two-operation graph replays
three times with complete node coverage, one host launch per replay, correct
outputs, and unchanged allocation counters. The trace contains two kernel
nodes per replay. This checks the profiling mechanism; ordinary Qwen B1/B4
kernel attribution and tracing-overhead measurements remain pending.

Run the mechanism probe on an explicitly assigned GPU:

```sh
CUDA_VISIBLE_DEVICES=GPU-... CUDA_HOME=/opt/cuda \
nsys profile --trace=cuda-sw --cuda-graph-trace=node:host-only \
  --sample=none --cpuctxsw=none --export=sqlite --output=/tmp/graph-node-probe \
  .venv/bin/python -m qualify.graph_node_probe --output /tmp/graph-node-map.json

.venv/bin/python -m qualify.kernel_timeline \
  --database /tmp/graph-node-probe.sqlite --map /tmp/graph-node-map.json \
  --output /tmp/graph-node-analysis.json
```

The analyzer rejects incomplete, ambiguous, cyclic, or foreign-process clone
provenance. It reports kernel durations, interval unions, and first-to-last
device-node spans separately. A node span is not a whole-graph envelope;
uncovered intervals do not establish a host scheduling cause. Trace diagnostics
are retained for review. Node tracing requires paired untraced measurements
before its durations can support performance conclusions.
