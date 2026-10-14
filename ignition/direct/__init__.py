"""Ignition's direct execution: a diagnostic and qualification path, not production serving.

Production serving loads an exported package in Helios, which owns scheduling,
input packing and graph replay without Python (see AGENTS.md, "Python
prepares; Helios serves"). At the project owner's request, `ignition serve
--direct` instead prepares a model in process and drives its prepared
execution from Python: a small fixed scheduler packs the same graph inputs
Helios packs, and every rank replays the captured graphs in lockstep, with rank
zero deciding each step. It exists to reproduce and bisect model-level faults
without Helios, never as a fallback; nothing selects it implicitly.

`scheduler` holds the CPU-only scheduling, cache bookkeeping and input packing,
`executor` runs one rank's steps, and `server` exposes a minimal non-streaming
OpenAI-compatible endpoint over them.
"""
