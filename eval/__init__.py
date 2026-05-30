"""Capability-probe suite for the architecture-comparison experiment.

Each probe lives in its own module and exposes a top-level `run(...)` function
returning a flat `dict[str, float]` of metric_name → value. `run_capability_suite.py`
loads a trained checkpoint and aggregates results across all probes.

See EXPERIMENTS.md §5 for the rationale behind each probe.
"""

from __future__ import annotations
