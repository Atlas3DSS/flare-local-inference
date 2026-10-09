"""Public package surface for `serve` with lazy imports."""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["ServingEngine", "SamplingParams", "GenerationResult", "__version__"]


def __getattr__(name: str):
    if name in {"ServingEngine", "SamplingParams", "GenerationResult"}:
        from serve.engine import GenerationResult, SamplingParams, ServingEngine

        exports = {
            "ServingEngine": ServingEngine,
            "SamplingParams": SamplingParams,
            "GenerationResult": GenerationResult,
        }
        return exports[name]
    raise AttributeError(f"module 'serve' has no attribute {name!r}")
