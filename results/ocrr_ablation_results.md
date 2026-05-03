# OCRR vote-rule ablation — substrate variants

**Date:** 2026-05-02
**Script:** `scripts/run_ocrr_ablation.py`
**Cell:** banking77 / oracle / seeds {0, 1, 2}
**Wall time:** 242 s

Tests whether the substrate's full vote rule (margin-band majority count +
max-similarity tiebreak + recency tiebreak) is load-bearing or whether
simpler variants achieve the same accuracy.

## Variants

| Variant | Vote rule |
|---|---|
| substrate_k1 | k=1 nearest-neighbour, no voting |
| substrate_sumsim | sum-of-similarities (no margin band, no count) |
| substrate_count_only | margin-band count + insertion-order tiebreak |
| substrate_no_recency | margin-band count + max_sim tiebreak (no recency) |
| substrate (full) | margin-band count + max_sim + recency |

## Aggregated results (3 seeds, mean ± std)

| Variant | Final novel | Final orig | →10% | →50% | →70% | →90% |
|---|---:|---:|---:|---:|---:|---:|
| substrate_k1 | 0.907 ± 0.031 | 0.938 ± 0.009 | 30 | 38 | 74 | 138 |
| **substrate (full)** | **0.905 ± 0.027** | **0.950 ± 0.007** | 35 | 66 | 103 | 174 |
| substrate_count_only | 0.905 ± 0.027 | 0.950 ± 0.007 | 35 | 66 | 103 | 174 |
| substrate_no_recency | 0.905 ± 0.027 | 0.950 ± 0.007 | 35 | 66 | 103 | 174 |
| substrate_sumsim | 0.893 ± 0.020 | 0.947 ± 0.006 | 39 | 73 | 123 | 198 |

## Findings

### 1. In the dense-substrate regime, vote-rule details barely matter

`substrate_count_only`, `substrate_no_recency`, and `substrate (full)` are
**identical to 4 decimal places** on every metric. With ~9000 seed entries
across 67 known classes (~130 entries per class), the margin band almost
always contains 5 entries with the same label — so count is decisive
before any tiebreak fires. **In this regime, the vote rule's complexity is
not load-bearing.**

### 2. The full vote rule pays off in *sparse* regimes

The bounded-substrate storage sweep (`research/ocrr_storage_sweep_results.md`)
shows that at budgets ≤ 1000 entries (~13 per class or fewer), recency
tiebreaks become decisive — that's where the demo's "I just corrected
this exact query, now classify the same query" loop runs. The full vote
rule was designed for that case. **In the OCRR benchmark's dense regime,
count_only is a perfectly fine substrate; recency is paper-justified by
the sparse-regime ablations.**

### 3. k=1 is competitive but slightly worse on retention

`substrate_k1` matches the voting variants on novel-class accuracy
(0.907 vs 0.905) but loses 1.2 pp on original-distribution accuracy
(0.938 vs 0.950). Single-neighbour predictions are more sensitive to
near-duplicate stream entries pushing the seed corpus's nearest-neighbour
to a wrong class. The 5-NN vote averages this out.

### 4. Sum-of-similarities is the clear loser

`substrate_sumsim` (no margin band, no count gate, just summed cosines)
falls 1.2 pp behind on novel and is consistently slowest to recover. This
is the bug we identified in the live demo (4 mediocre matches outvote 1
strong match) reproduced quantitatively. Margin-band gating is the
load-bearing piece.

## Implications for the paper

The ablation is honest but doesn't justify the full complexity at dense
budgets. Right framing for the paper's Section 5:

> "In the dense-substrate regime (≥ 100 entries per class), all margin-
> band variants converge. The recency and max-sim tiebreaks become
> decisive only at sparse budgets, characterised in Section 5.2 (storage
> sweep). We retain the full vote rule because it preserves correctness
> across both regimes."

So the vote rule isn't a contribution per se — it's the simplest rule
that works across the operating envelope. The substrate's *architectural*
contribution is the append-only ledger plus encoder-agnosticism.

## Output

- `research/ocrr_ablation_summary.csv` — aggregated 5-row table
- `research/ocrr_ablation.log` — full stdout

Closes part of #54. LoRA-DeBERTa baseline is the other half.
