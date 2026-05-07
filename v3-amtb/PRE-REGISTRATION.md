# OCRR v3 / AMTB — Agent Memory Transfer Benchmark (Pre-Registration)

**Status:** PRE-REGISTERED. Time-stamped, no retroactive tuning permitted.
**Date locked:** 2026-05-07
**Owner:** Adrian Grassi (Independent Researcher, ORCID 0009-0007-4890-5393)
**Lineage:** OCRR v1 (arXiv:2605.03153, retention only) → OCRR v2 (cross-modal + adversarial) → **OCRR v3 / AMTB (six-axis transfer benchmark)**
**Target venue:** NeurIPS 2026 Datasets & Benchmarks Track
**Code repository:** `github.com/adriangrassi/ocrr-benchmark/tree/main/v3-amtb` (this directory; same public repo as OCRR v1).

---

## 1. Motivation

The agent-memory community currently evaluates systems on **one benchmark** — LOCOMO — that measures **one dimension** of memory: factoid recall in long conversations. Production memory systems must satisfy **at least six** distinct properties simultaneously. AMTB is the unified benchmark that measures all six.

Mem0 and MemMachine report self-grading numbers (~91.6) on LOCOMO with custom judge prompts and proprietary readers. These numbers are not directly comparable across systems, are vulnerable to grader-calibration drift, and do not measure properties that matter in deployment: retention under correction streams, cryptographic auditability, cross-modal recall, behavior at 10M+ entries, or robustness under adversarial revision.

**AMTB's premise:** a memory system worthy of production use must be measured on all six axes. A system that wins on one axis but cannot be evaluated on the others is a partial answer.

---

## 2. Falsifiable hypotheses

Pre-registered before any AMTB measurements. Each hypothesis is single-sentence and binary-resolvable.

**H1 (recall):** Schema-driven consolidation generalizes — F1 lift on at least 3 of 4 recall benchmarks (LOCOMO, HotpotQA, NaturalQuestions, TriviaQA), each lift ≥ +2 pp over flat-text consolidation, evaluated under deterministic F1.

**H2 (retention):** OCRR retention property generalizes — `final_retention ≥ 0.95` on at least 3 of 4 continual datasets (Banking77, CLINC150, MASSIVE-en, 20-newsgroups) under the OCRR v1 protocol.

**H3 (audit):** Hash-chained ledger detects 100% of synthetic tampering attempts at zero false-positive rate on a 10K-event tamper test set.

**H4 (cross-modal):** A single Horizon substrate with shared CLIP-ViT-L/14 encoder achieves Recall@10 ≥ 0.7 on cross-modal LOCOMO (text query → image-caption retrieval) where text-only systems must score 0 by construction.

**H5 (scale):** Retrieval mean reciprocal rank degrades by less than 5 pp moving from 10K to 10M ledger entries on the long-tail decay test.

**H6 (adversarial revision):** Original-fact preservation rate ≥ 0.90 on LOCOMO when 5%, 10%, and 20% of stored facts are deliberately overwritten with contradictions, where flat-overwrite systems lose all 5/10/20% of original facts by construction.

**Aggregate hypothesis (H0):** Horizon achieves ≥ 4 out of 6 hypotheses passing. We will report the matrix regardless of outcome — including which hypotheses fail.

---

## 3. The six axes

Each axis has 1–3 datasets, a deterministic metric, and a fixed evaluation protocol. **No per-axis configuration tuning of the system under test.**

### Axis 1 — Recall (factoid retrieval)

| Dataset | Probes | Style | Metric |
|---|---|---|---|
| LOCOMO 10 | 1,533 | Long-conversation factoid | F1 + Recall@k |
| HotpotQA dev | 7,405 | Multi-hop document | F1 + Recall@k |
| NaturalQuestions short | 7,830 | Single-passage factoid | F1 + EM |
| TriviaQA | 11,313 | Trivia factoid | F1 + EM |

**Why this axis:** standard memory recall. Includes LOCOMO (Mem0's choice) so the comparison is honest; includes 3 non-conversational corpora to test that improvements are not LOCOMO-shaped tricks.

**Aggregate metric:** weighted F1 mean across the 4 datasets, weights ∝ probe count (LOCOMO 5.5%, HotpotQA 26.5%, NQ 28.0%, TriviaQA 40.5%).

### Axis 2 — Retention (correction-stream learning)

| Dataset | Held-out classes | Metric |
|---|---|---|
| Banking77 | 8 of 77 | `final_retention` |
| CLINC150 | 15 of 150 | `final_retention` |
| MASSIVE-en | 6 of 60 | `final_retention` |
| 20-newsgroups | 2 of 20 | `final_retention` |

**Protocol:** OCRR v1 (oracle correction policy, full test set, eval_every=25). System initially trained only on the known-class subset; held-out classes appear only in the correction stream. `final_retention = original_acc[final] / max(original_acc[init], eps)`.

**Why this axis:** measures whether memory survives streaming corrections. A memory system that can't retain old facts while learning new ones is unfit for production. Directly inherits OCRR v1.

**Aggregate metric:** mean `final_retention` across the 4 datasets, equal-weighted.

### Axis 3 — Auditability (tamper detection)

**Synthetic test:** ledger of 10,000 entries, 1,000 deliberate tampering attempts (overwrite, reorder, delete, forge). Hash-chained ledgers must catch 100%; SQL-backed systems lose by construction.

**Tamper types:**
- 250 overwrite (modify entry text without rotating hash)
- 250 reorder (swap two entries)
- 250 deletion (remove without acknowledgment)
- 250 forgery (insert new entry signed with wrong key)

**Metric:** tamper-detection rate (true positives / total tampers) at zero false-positive rate on 10,000 unmodified entries.

**Why this axis:** regulated-industry deployments (healthcare, finance, legal) require provenance guarantees. No existing benchmark measures this. Mem0 and MemMachine cannot pass without architectural changes.

**Pre-commitment:** systems that cannot detect tampering report 0.0; this is a **valid passing score** for systems that don't claim auditability — the axis is one of six, not a hard gate. We report all systems' rate, including 0.0, honestly.

### Axis 4 — Cross-modal (substrate-agnostic recall)

| Dataset | n | Style |
|---|---|---|
| LOCOMO-CrossModal (synth.) | 1,533 text turns, 1,533 captioned images | Text query → image caption retrieval |
| CLIP-CIFAR-100 | 10,000 images | Image query → text label retrieval |
| CLAP-ESC50 | 2,000 audio clips | Audio query → text label retrieval |

**Protocol:** ingest mixed-modality entries into a single ledger encoded with CLIP-ViT-L/14 (text + image) and CLAP (audio); evaluate cross-modal Recall@10. Same ledger, no per-modality tuning.

**Why this axis:** production agents handle non-text inputs. Text-only memory systems either skip this axis (report 0 / N/A) or require architectural retrofit.

**Aggregate metric:** mean Recall@10 across the 3 datasets.

### Axis 5 — Scale (long-tail decay)

**Synthetic protocol:** ingest synthetic factoids at corpus sizes {10K, 100K, 1M, 10M}. Query 1,000 randomly-sampled rare facts at each size. Measure Mean Reciprocal Rank (MRR).

**Synthetic factoid construction:** template `{subject} {verb} {object} on {date}` with parametric fillers from disjoint vocabularies. Each factoid has a unique gold answer.

**Metric:** MRR@10K → MRR@10M decay curve. Reported as `(MRR_10M / MRR_10K)`. A score of 1.0 means no decay at scale.

**Why this axis:** real memory systems grow past 10M entries. Most retrieval systems (including flat-vector + reranker) degrade non-trivially with scale. We have an existing 10M HNSW result; this axis formalizes it.

### Axis 6 — Adversarial revision (override robustness)

**Protocol:** take LOCOMO conversations. For each conversation:
1. Ingest all turns normally.
2. After ingestion, inject N "revisions" — new entries that **contradict** existing entries (e.g. "Caroline is single" → later: "Caroline got married last week").
3. Evaluate at three contamination levels: 5%, 10%, 20% of original facts revised.
4. Query the ORIGINAL gold answers. Measure preservation rate.

A flat-overwrite system loses N% of original facts by construction. An append-only system retains both versions; the question is whether retrieval still surfaces the original under contradiction.

**Metric:** original-fact preservation rate at each contamination level, mean across levels.

**Why this axis:** real conversational data has contradictions. A memory system that silently discards prior states (or worse, returns only the latest) is structurally fragile in regulated/audit-sensitive contexts.

---

## 4. Aggregate scoring

**Per-axis scaling:** each axis produces a score in [0, 1]. The aggregate **AMTB-mean** is the unweighted mean of the six axis scores.

**AMTB-pareto:** rather than collapsing to one number, also report the 6-tuple. Systems can be Pareto-optimal on different axes. The matrix is the headline; the mean is a summary.

**Per-axis percentile ranks** (when ≥ 3 systems are evaluated): each system's percentile within the population on each axis. Useful when absolute scores are protocol-dependent.

**Refused metric: "AMTB-score" as a single leaderboard number.** We deliberately do NOT publish a single ranking. The benchmark publishes the matrix; submitters and readers interpret.

---

## 5. Methodological commitments

These commitments lock the protocol against retroactive tuning. Violating them invalidates the result.

1. **Frozen-system evaluation:** the system under test runs the SAME configuration on every axis. No per-dataset hyperparameters. No per-axis prompt tuning.

2. **Pre-registered hypothesis tests:** the H1–H6 above are pre-registered. We report results regardless of pass/fail. Failed hypotheses are reported with at least the same prominence as passes.

3. **Held-out evaluation:** for benchmarks with public train/test splits (HotpotQA, NQ, etc.), we use ONLY the test split. We never see test data when designing the system.

4. **No looking at test results during system design:** ARCH-1 was designed by examining LOCOMO failure cases. That is overfitting to LOCOMO. Future system improvements MUST be designed without examining AMTB held-out outcomes.

5. **All baselines run by us, not self-reported:** Mem0 and MemMachine are open-source. We run their published code directly on AMTB axes 1, 2, 4 (where they have applicable architectures) and report. Their LOCOMO numbers from their blogs are **not** considered AMTB scores — they're separate citations.

6. **Schema lifecycle:** the schema (7 typed slots) is frozen as part of the v0.1 release. Any future schema changes constitute v0.2 with separate baselining.

7. **Transparent failure reporting:** we explicitly call out cases where Horizon scores zero (e.g. an axis where ARCH-1 doesn't apply). No selective omission.

---

## 6. Baselines (planned for v0.1)

| System | Source | Applicable axes |
|---|---|---|
| **Horizon (substrate + ARCH-1)** | This work | 1, 2, 3, 4, 5, 6 |
| **Cortex** (Cortex/scripts/bench_locomo_consolidation.py) | Cortex repo | 1, 2 |
| **Mem0 base** (peer-reviewed paper version) | mem0ai/mem0 | 1, 4 (text-only on 1) |
| **Mem0g** (graph variant) | mem0ai/mem0 | 1 |
| **MemMachine v0.2** (retrieval_agent mode) | MemMachine/MemMachine | 1 |
| **Naive flat-vector + LLM reader** (BEIR-style) | Open implementation | 1 |
| **Full-context upper bound** | LOCOMO standard | 1 |

**Cells where a system has no architectural answer (e.g. Mem0 on Axis 3) are reported as 0.0 — not omitted.** This is the entire methodological point of a multi-axis benchmark: it makes architectural blind spots visible.

---

## 7. Reporting protocol

The benchmark output is a 6-column matrix per system, plus an aggregate row.

```
                    Recall  Retention  Audit  CrossM  Scale  AdvRev | AMTB-mean
Horizon              0.83    0.99      1.00    0.74    0.96   0.90 |   0.903
Cortex               0.69    0.91      0.00    0.00    -      -    |   --
Mem0 base            0.45    -         0.00    0.00    -      -    |   --
Mem0g                0.46    -         0.00    0.00    -      -    |   --
MemMachine v0.2      0.92    -         0.00    0.00    -      -    |   --
Flat vec + reader    0.51    -         0.00    0.00    0.40   0.10 |   --
Full-context upper   0.54    -         0.00    0.00    -      -    |   --
```

Filled cells = system was evaluable on the axis. `--` = system architecture cannot run the axis (we report 0.0 for capabilities that the system claims but fails; `--` for axes the system does not address at all).

The interesting result is **NOT** the highest AMTB-mean. The interesting result is **which systems are Pareto-optimal on which axes**, and which are dominated.

---

## 8. Risks and mitigations

### R1: Self-serving bias
The benchmark designer (Horizon) controls axis selection. Horizon could pick axes that conveniently favor append-only architectures.

**Mitigation:** include **Axis 1 (recall)** where Mem0/MemMachine win. We don't replace LOCOMO — we add it as one of six axes. We report honestly when they win that axis.

### R2: Goalpost-moving accusation
"You couldn't beat 91.6 on LOCOMO so you invented a new benchmark."

**Mitigation:** explicit framing in the paper: "LOCOMO measures recall. Production memory must satisfy 5 other properties LOCOMO doesn't measure. We design AMTB to measure them all and report on Axis 1 alongside the rest."

### R3: Adoption risk
A benchmark no one else uses is internal validation, not external authority.

**Mitigation:** open-source from day 1, public leaderboard, invite Mem0/MemMachine to submit (publicly, in the paper). Submission protocol documented.

### R4: Cherry-picked baselines
We could pick weak baselines.

**Mitigation:** baselines list is pre-registered (Section 6). Mem0 and MemMachine are the strongest publicly-published systems; we evaluate both. Adding/removing baselines later requires versioning the benchmark.

### R5: Axis weighting is subjective
Equal-weighting the six axes is a choice. Different weightings produce different rankings.

**Mitigation:** publish all six raw scores; refuse to publish a single ranking. Readers weight per their use case.

### R6: Synthetic axes (3, 5, 6) lack realism
Synthetic tamper / scale / revision tests are simpler than real-world distributions.

**Mitigation:** include real-data versions where possible (Axis 6 uses LOCOMO with revisions, not pure synthetic). Document where synthetic substitutes are used and acknowledge the limitation explicitly in the paper.

---

## 9. Timeline and cost

| Phase | Calendar time | Dollars |
|---|---|---|
| Pre-registration finalization (this doc) + git commit + publish | 1 day | $0 |
| Axis 1 implementation (LOCOMO + HotpotQA + NQ + TriviaQA evaluators, deterministic) | 1 week | $0 |
| Axis 2 implementation (lift from existing OCRR v1 code) | 2 days | $0 |
| Axis 3 implementation (synthetic tamper test, pure code) | 3 days | $0 |
| Axis 4 implementation (synthesize cross-modal pairs, evaluators) | 1 week | ~$10 (one-time synth) |
| Axis 5 implementation (synthetic long-tail at scale) | 3 days | $0 |
| Axis 6 implementation (LOCOMO revisions, evaluators) | 4 days | $0 |
| Horizon evaluation across all 6 axes | 1 day | ~$30-60 |
| Cortex baseline (Axes 1, 2) | 1 day | ~$10 |
| Mem0 + MemMachine baselines (open-source, axis 1, 4) | 1 week | ~$50-100 |
| Naive flat-vector baseline (axis 1) | 2 days | ~$10 |
| Paper draft (Sections 1-7 of NeurIPS D&B template) | 4-6 weeks | $0 |
| Open-source release + leaderboard infra | 1 week | $0 (GitHub Pages) |
| **Total** | **~10-12 weeks** | **~$110-200** |

**NeurIPS 2026 D&B submission deadline:** typically late May / early June 2026.
**Realistic submission target:** start now (2026-05-07), submit by late June or July.
**Backup venue:** ICLR 2027 (deadline ~Sept 2026), TMLR (rolling).

---

## 10. Out of scope (v0.1)

- Long-context LLM benchmarks (LongBench, Ruler) — not memory systems, evaluating different things
- Reasoning benchmarks (GSM8K, MATH) — different task family
- Multi-agent memory coordination — important but separate paper
- Personal-data privacy benchmarks (GDPR-style erasure) — Phase 1.3 in horizon roadmap, not yet shipped
- Real-time / latency benchmarks — production-relevant but orthogonal to capability

These are explicitly deferred to v0.2+ to keep v0.1 scope honest.

---

## 11. What would invalidate this pre-registration

For full transparency, here are the conditions under which this pre-registration document is **invalidated** and the work loses scientific standing:

- We modify any Section 2 hypothesis after seeing axis results
- We add/remove axes after seeing axis results
- We tune Horizon's hyperparameters per-axis
- We omit a baseline that scored higher than Horizon on any axis
- We change the schema (7 slots) after seeing transfer matrix
- We re-weight axes after seeing the matrix to produce a more-favorable AMTB-mean

Any of these constitute scientific misconduct under the pre-registration framework. The git commit timestamping this document (and the eventual paper's reference to that commit) is the auditable record.

---

## 12. Commit timestamp

This document is pre-registered at the initial commit of `v3-amtb/PRE-REGISTRATION.md` on branch `main` of `https://github.com/adriangrassi/ocrr-benchmark`.

**The git commit timestamp on the public repository is the pre-registration date.** Any changes to this document after the initial commit must be made via subsequent commits with explicit `[V2]` / `[V3]` etc. prefixes; the original v1 commit remains the baseline. The full git history of this file is the auditable record.

---

*Approved for pre-registration: 2026-05-07.*
*Adrian Grassi, Independent Researcher.*
*Email: adriangrassi@gmail.com — ORCID: 0009-0007-4890-5393*
