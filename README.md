# OCRR — Online Correction Recovery Rate

**A benchmark for measuring how fast classification systems recover from
distribution shift via online correction.**

Static benchmarks (Banking77, GLUE, MMLU) measure a model frozen at training
time. They cannot characterise the property that matters most for real-world
deployment: how fast does the system recover when the input distribution
shifts and a user provides corrections? OCRR measures exactly that.

## Headline result — Banking77, oracle correction policy, 3 seeds

| System | Buffer | Final novel | Final orig | →70 % novel |
|---|---:|---:|---:|---:|
| **substrate** | ∞ | **0.905 ± 0.027** | **0.950 ± 0.007** | **103** |
| bounded reservoir 5000 | 5000 | 0.883 ± 0.029 | 0.943 ± 0.003 | 135 |
| bounded reservoir 1000 | 1000 | 0.807 ± 0.016 | 0.897 ± 0.003 | 351 |
| river_logreg | params | 0.867 ± 0.106 | **0.000** | 134 |
| knn_lm | ∞ | 0.823 ± 0.045 | 0.963 ± 0.005 | 271 |
| lora_deberta_v3_large | params + LoRA | 0.771 ± 0.086 | 0.108 ± 0.008 | 297 |
| online_linear | params | 0.544 ± 0.081 | 0.928 ± 0.012 | never |
| a_gem | params + 1000 | 0.484 ± 0.065 | 0.938 ± 0.014 | never |
| ewc | params | 0.405 ± 0.069 | 0.946 ± 0.007 | never |
| lwf | params | 0.118 ± 0.025 | 0.949 ± 0.004 | never |
| static_knn | (seed) | 0.000 | 0.957 | never |
| static_linear | params | 0.000 | 0.952 | never |

**Substrate sits alone on the storage-vs-recovery Pareto frontier across all
6 (dataset × policy) cells.** No alternative system simultaneously matches it
on novel-class recovery and original-distribution retention.

See [`paper/paper.md`](paper/paper.md) for the full draft and
[`results/`](results/) for per-seed CSVs, logs, and figures.

## What OCRR measures

A classification system is presented with a stream of `(text, label)` pairs
drawn from a distribution that has shifted away from its initial training
set. After each prediction:

- If wrong, a correction policy decides whether to call
  `system.correct(text, label)`.
- The system updates its state in real time.
- We track accuracy on **both** the held-out novel distribution AND the
  original distribution (forgetting check) over the correction-count axis.

Reported metrics: final novel accuracy, final original accuracy,
corrections-to-N % thresholds, and per-system storage footprint.

## Streaming-learning constraints

| Property | Required by OCRR? |
|---|---|
| Data arrives sequentially | **Yes** |
| Model updates in real time on each correction | **Yes** |
| Memory bounded (no historical-data storage) | Optional — reported per system |

The third constraint is what classical online-ML libraries (`river`)
require. OCRR does not enforce it but **reports each system's storage
footprint** so the comparison is honest about the trade-off. The
`bounded_reservoir_*` and `bounded_fifo_*` variants probe the entire
storage-vs-recovery Pareto.

## Repository layout

```
ocrr-benchmark/
├── ocrr_benchmark/         # importable Python package
│   ├── eval/               # harness, systems, baselines, ablations
│   ├── memory/             # ImmutableLedger (append-only + Merkle hash chain)
│   └── datasets/           # Banking77 / CLINC150 loaders
├── scripts/                # run_ocrr*.py — one per result cell
├── results/                # CSVs, logs, figures from the paper
├── paper/                  # paper draft + figures
├── REPRODUCING.md          # step-by-step reproduction
├── pyproject.toml          # dependencies
└── LICENSE                 # MIT (code) — paper is CC BY 4.0
```

## Quick start

```bash
# Install
pip install -e .

# Run the v1 single-cell sanity check (Banking77, oracle, 4 systems, 1 seed)
python scripts/run_ocrr.py --output results/_sanity.csv

# Reproduce the headline 9-system × 18-cell sweep
python scripts/run_ocrr_full_sweep.py --output results/_repro_full.csv
```

See [REPRODUCING.md](REPRODUCING.md) for the full reproduction playbook.

## Systems benchmarked (13 total)

**Static strawmen:** `static_knn`, `static_linear` — zero learning, lower
bound on novel-class accuracy.

**Naive online:** `online_linear` — frozen encoder + per-correction SGD on
the classifier head.

**Continual-learning baselines:** `ewc` (Kirkpatrick 2017), `a_gem`
(Chaudhry 2019), `lwf` (Li & Hoiem 2017).

**Retrieval/parametric hybrids:** `knn_lm` (Khandelwal 2020).

**Online-ML libraries:** `river_logreg` (LogisticRegression).

**Parameter-efficient fine-tune:** `lora_deberta_v3_large` (LoRA rank 8 on
DeBERTa-v3-large query/value projections).

**Substrate:** `substrate` (unbounded), plus `bounded_reservoir_{1000, 5000}`
and `bounded_fifo_{1000, 5000}` storage-Pareto variants.

**Ablations** (not in main table): `substrate_k1`, `substrate_sumsim`,
`substrate_count_only`, `substrate_no_recency`. Vote-rule details barely
matter in the dense-substrate regime; margin-band gating is the only
load-bearing piece.

## Datasets

- **Banking77** (Casanueva et al. 2020) — 77 fine-grained banking intents,
  ~10 k train / ~3 k test. CC-BY-4.0.
- **CLINC150** (Larson et al. 2019) — 150-class cross-domain intents, ~15 k
  train / ~5 k test. CC-BY-3.0.

## Citation

```bibtex
@misc{grassi2026ocrr,
  title  = {OCRR: Online Correction Recovery Rate — A Benchmark for
            Classification Systems Under Distribution Shift},
  author = {Adrian Grassi},
  year   = {2026},
  note   = {arXiv preprint, NeurIPS Datasets & Benchmarks 2026 submission}
}
```

The arXiv ID will be inserted here once the submission is live.

## License

- **Code** (`ocrr_benchmark/`, `scripts/`): MIT. See [LICENSE](LICENSE).
- **Paper** (`paper/`): CC BY 4.0 — distributed via arXiv under that licence.
- **Data**: Banking77 (CC-BY-4.0) and CLINC150 (CC-BY-3.0) are upstream
  datasets distributed under their original licences.

## Status

- v0.1.0 — initial public release of paper draft + reproducibility package.
- See [open issues](https://github.com/adriangrassi/ocrr-benchmark/issues) for
  follow-up work (LLM-ICL with frontier-API spot check, cross-modal
  encoder-swap study, convergence theory).
