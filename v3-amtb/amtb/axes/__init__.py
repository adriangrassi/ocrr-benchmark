"""AMTB axis evaluators.

One module per axis. Each module exposes:
- A `run(system, **config)` function returning AxisResult
- An `EvaluationContract` describing what the system must implement

Per the pre-registration, axis definitions are FROZEN at v0.1. Bug fixes
in evaluators that don't change the metric definition are allowed (and
should be documented in commit messages). Any change to a metric
definition requires a v0.2 release.
"""
