"""AMTB — Agent Memory Transfer Benchmark.

Six-axis benchmark for agent memory systems. Pre-registered 2026-05-07
at git commit 745b054 on github.com/adriangrassi/ocrr-benchmark.

The pre-registration locks experimental design (axes, datasets, hypotheses,
metrics, baselines) before any measurements. Any retroactive change to
this design invalidates the published result. See ../PRE-REGISTRATION.md
for full details and the six conditions that constitute scientific
misconduct under this framework.
"""

__version__ = "0.1.0-pre"

from amtb.types import (
    AMTBMatrix,
    AxisName,
    AxisResult,
    MemorySystem,
    SystemReport,
)

__all__ = [
    "AMTBMatrix",
    "AxisName",
    "AxisResult",
    "MemorySystem",
    "SystemReport",
]
