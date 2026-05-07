# OCRR v3 / AMTB — Agent Memory Transfer Benchmark

[![arXiv](https://img.shields.io/badge/arXiv-2605.03153-b31b1b.svg)](https://arxiv.org/abs/2605.03153)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Pre-Registered](https://img.shields.io/badge/status-pre--registered-blue)](#pre-registration)

**A six-axis benchmark for agent memory systems. Measures recall, retention, auditability, cross-modal recall, scale, and adversarial-revision robustness.**

> **⚠ Status as of 2026-05-07:** This directory contains the **pre-registration only**. Implementation, baselines, and leaderboard are forthcoming. The pre-registration is time-stamped to lock the experimental design before measurements begin.

## Why this exists

Production agent memory systems must satisfy **at least six** distinct properties simultaneously. The current state of the field evaluates one: factoid recall in long conversations (LOCOMO). Self-reported leaderboard numbers (Mem0 91.6, MemMachine 91.69) compete on a single brittle metric with custom judge prompts.

This benchmark measures all six properties with deterministic metrics, public datasets, and a pre-registered evaluation protocol.

## Lineage

This is the third paper in the OCRR research program:

| Version | Scope | Status |
|---|---|---|
| **OCRR v1** | Retention only — single axis, two NLP datasets, 13 systems | Public ([arXiv:2605.03153](https://arxiv.org/abs/2605.03153)) |
| **OCRR v2** | Retention + cross-modal + adversarial corrections + 10-stage chain | In progress, mid-2026 release |
| **OCRR v3 / AMTB** | All six axes unified into a transfer benchmark | **Pre-registered 2026-05-07** |

OCRR v1 introduced retention as a measurable property of memory systems. OCRR v2 extends to cross-modal substrates and adversarial corrections. OCRR v3 is the unification: every property OCRR has measured plus three more (auditability, scale, adversarial revision) become a single benchmark suite.

## The six axes

Each axis has 1–3 datasets with deterministic metrics. **No LLM-as-judge in any axis** — all gold answers are factoids, classification labels, or boolean checks. Reproducible across machines without API spend on judge models.

| # | Axis | Tests | Datasets |
|---|---|---|---|
| 1 | **Recall** | Factoid retrieval over long context | LOCOMO, HotpotQA, NaturalQuestions, TriviaQA |
| 2 | **Retention** | Correction-stream learning without forgetting | Banking77, CLINC150, MASSIVE-en, 20-newsgroups |
| 3 | **Auditability** | Cryptographic tamper detection | Synthetic 10K-entry tamper test |
| 4 | **Cross-modal** | Substrate-agnostic recall (text + image + audio) | LOCOMO-CrossModal, CLIP-CIFAR-100, CLAP-ESC50 |
| 5 | **Scale** | Long-tail decay with growing corpus | Synthetic 10K → 10M factoid retrieval |
| 6 | **Adversarial revision** | Original-fact preservation under contradicting inputs | LOCOMO + revision injection at 5%, 10%, 20% |

## Aggregate scoring

Per-axis scores are reported as a **transparent 6-tuple matrix**. The benchmark refuses to publish a single ranking — different deployments weight axes differently. We provide:

- Per-axis percentile ranks (when ≥ 3 systems are evaluated)
- AMTB-mean (unweighted average) as a summary, not a ranking
- Pareto-frontier analysis across axes

## Methodological commitments

These commitments are pre-registered. Violating any invalidates the published result:

1. **Frozen-system evaluation.** A system runs the SAME configuration on every axis. No per-dataset or per-axis hyperparameter tuning.
2. **Held-out test splits only.** Benchmarks with public train/test splits use ONLY test. No looking at test data when designing the system.
3. **Pre-registered hypotheses.** All hypotheses (H1–H6) are published before measurement. Failures are reported with the same prominence as wins.
4. **Schema lifecycle.** The v0.1 schema (typed-slot consolidation: 7 categories) is frozen for v0.1. Future changes require a v0.2 release with separate baselining.
5. **All baselines run by us.** Baseline self-reported numbers are cited but not entered into the leaderboard. We run open-source baselines directly on AMTB axes.
6. **Transparent failure reporting.** Cells where a system has no architectural answer are reported as 0.0 (not omitted). The benchmark makes architectural blind spots visible.

## Pre-registration

The full pre-registration document is at [`PRE-REGISTRATION.md`](./PRE-REGISTRATION.md).

The pre-registration was committed to this public repository on **2026-05-07**. The git commit timestamp is the auditable pre-registration date. Any retroactive change to the design (axes, hypotheses, methodology) after measurements begin invalidates the published result.

## Planned baselines (v0.1)

| System | Source | Applicable axes |
|---|---|---|
| Horizon (substrate + schema-driven consolidation) | This work | 1, 2, 3, 4, 5, 6 |
| Cortex (consolidation pipeline) | github.com/adriangrassi/Cortex | 1, 2 |
| Mem0 base (peer-reviewed) | github.com/mem0ai/mem0 | 1, 4 |
| Mem0g (graph variant) | github.com/mem0ai/mem0 | 1 |
| MemMachine v0.2 | github.com/MemMachine/MemMachine | 1 |
| Naive flat-vector + LLM reader | This repo | 1 |
| Full-context upper bound | LOCOMO standard | 1 |

Cells where a baseline has no architectural answer (e.g. text-only systems on Axis 4) are reported as 0.0 — not omitted. **Making architectural blind spots visible is the entire methodological point.**

## Submission protocol (forthcoming)

Once v0.1 ships:
- **Open-source submission:** systems that are open-source can be submitted by anyone with a config + reproducibility script. We re-run them ourselves to enter on the leaderboard.
- **Self-reported submission:** closed-source systems can self-report with attestation; flagged on the leaderboard as "self-reported" (not directly comparable).

## Roadmap

| Milestone | Target |
|---|---|
| v0.1 pre-registration committed | 2026-05-07 ✓ |
| v0.1 axis-1 implementation (deterministic LOCOMO + HotpotQA + NQ + TriviaQA) | Q2 2026 |
| v0.1 axes 2–6 implementations | Q2-Q3 2026 |
| v0.1 Horizon evaluation across all 6 axes | Q3 2026 |
| v0.1 baseline evaluations (Cortex, Mem0, MemMachine) | Q3 2026 |
| Paper draft (NeurIPS 2026 D&B) | Late June 2026 target |
| Public leaderboard launch | Q3-Q4 2026 |

## Citation

If you reference this pre-registration before the v0.1 paper releases:

```bibtex
@misc{grassi2026amtb-prereg,
  author       = {Adrian Grassi},
  title        = {{OCRR v3 / AMTB}: A Pre-Registered Six-Axis Benchmark for Agent Memory Systems},
  year         = {2026},
  howpublished = {\url{https://github.com/adriangrassi/ocrr-benchmark/tree/main/v3-amtb}},
  note         = {Pre-registration committed 2026-05-07. Implementation forthcoming.}
}
```

## Contact

- Adrian Grassi — `adriangrassi@gmail.com` — ORCID: [0009-0007-4890-5393](https://orcid.org/0009-0007-4890-5393)
- Issues / questions: open a GitHub issue on this repo
- Submission inquiries: open a GitHub issue tagged `submission`

## License

MIT (matches the OCRR v1 release).
