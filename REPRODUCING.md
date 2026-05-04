# Reproducing the OCRR results

This document walks through reproducing every numbered cell in the paper
from a clean checkout. If a step deviates from the published numbers, please
[open an issue](https://github.com/adriangrassi/ocrr-benchmark/issues) — we
treat reproducibility regressions as bugs.

## Environment

- Python ≥ 3.10
- ~16 GB RAM (substrate variants), ~24 GB GPU VRAM for LoRA-DeBERTa
- Datasets are downloaded automatically on first run from PolyAI
  (Banking77) and Hugging Face (CLINC150) — internet access required once.

## Install

```bash
git clone https://github.com/adriangrassi/ocrr-benchmark.git
cd ocrr-benchmark
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS
pip install -e .
pip install -e ".[river,plot]"    # for the river_logreg baseline + plotting
pip install -e ".[lora]"          # only if you want to reproduce LoRA-DeBERTa
```

Pinned versions are in `pyproject.toml`. Changing transitive versions of
torch / transformers may shift LoRA-DeBERTa numbers slightly because of
dtype handling — the substrate / kNN / linear baselines are version-stable.

## Reproduction scripts — what each one does

| Script | Cell in paper | Wall time |
|---|---|---|
| `scripts/run_ocrr.py` | v1 sanity: Banking77 / oracle / 1 seed / 4 systems | ~3 min |
| `scripts/run_ocrr_sweep.py` | v2 scope C: 2 datasets × 3 policies × 3 seeds × 4 systems | ~25 min |
| `scripts/run_ocrr_full_sweep.py` | **Headline table — 9 systems × 18 cells** | ~3.5 hr |
| `scripts/run_ocrr_storage_sweep.py` | Storage-vs-recovery Pareto (12 systems × 3 seeds) | ~45 min |
| `scripts/run_ocrr_ablation.py` | Vote-rule ablation (5 substrate variants × 3 seeds) | ~5 min |
| `scripts/run_ocrr_lora_deberta.py` | LoRA-DeBERTa-v3-large baseline (3 seeds, RTX 4090) | ~60 min |
| `scripts/run_substrate_scale_study.py` | Substrate scaling study: HNSW vs brute-force recall@k and accuracy at ledger scales 10 k / 100 k / 1 M (synthetic, validates the never-forget guarantee at scale) | ~5 min at 10 k, ~30 min at 100 k, ~6 hr at 1 M (CPU; brute-force prediction dominates) |

The published `results/` CSVs were produced on commit `c64e665` with the
versions pinned in `pyproject.toml`.

## Reproducing the headline number

```bash
python scripts/run_ocrr_full_sweep.py \
    --output results/_repro_full_sweep_results.csv \
    --summary results/_repro_full_sweep_summary.csv \
    --seeds 0,1,2
```

After ~3.5 hours, `results/_repro_full_sweep_summary.csv` should contain
the same per-cell mean ± std numbers as `results/ocrr_full_sweep_summary.csv`
to within numerical tolerance. The substrate row on Banking77/oracle should
read:

```
system=substrate, dataset=banking77, policy=oracle,
final_novel_mean=0.905, final_novel_std=0.027,
final_orig_mean=0.950, final_orig_std=0.007
```

## Reproducing the storage Pareto

```bash
python scripts/run_ocrr_storage_sweep.py \
    --output results/_repro_storage_sweep_results.csv \
    --summary results/_repro_storage_sweep_summary.csv
```

Then plot:

```bash
python -c "
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv('results/_repro_storage_sweep_summary.csv')
# (same plotting recipe used for results/ocrr_storage_sweep_pareto.png)
"
```

## Reproducing the LoRA-DeBERTa row (GPU required)

Requires `[lora]` extras and an ~8 GB+ GPU. RTX 4090 takes ~60 minutes
end-to-end across 3 seeds.

```bash
python scripts/run_ocrr_lora_deberta.py \
    --output results/_repro_lora_deberta_results.csv \
    --seeds 0,1,2
```

Expected: `final_novel ≈ 0.771 ± 0.086`, `final_orig ≈ 0.108 ± 0.008`.
The original-distribution drop is the headline finding here — LoRA on a
1.5 B-parameter encoder catastrophically forgets the seed distribution.

## Reproducing the vote-rule ablation

```bash
python scripts/run_ocrr_ablation.py \
    --output results/_repro_ablation_summary.csv
```

Expected (5 rows, banking77 / oracle, 3 seeds): in the dense-substrate
regime all margin-band variants converge to identical accuracy; only
`substrate_sumsim` (no margin band, just summed cosines) is consistently
worse, and `substrate_k1` (1-NN, no voting) loses ~1.2 pp on
original-distribution accuracy.

## Common gotchas

- **First run downloads ~1 GB of model weights** (`BAAI/bge-large-en-v1.5`,
  optionally `microsoft/deberta-v3-large` for the LoRA cell). They cache
  under `~/.cache/huggingface/`.
- **CLINC150 requires the `datasets` library**, which itself pulls
  `pyarrow` — already in our deps but worth knowing if `pip install` is
  surprising.
- **`river` is an optional extra** because it's a sizable dependency tree.
  Install it with `pip install -e ".[river]"` only if you want the
  `river_logreg` baseline.
- **Substrate without numeric stability**: cosine similarity floors are
  set in `ocrr_benchmark/eval/ocrr_systems.py:148`. Don't tune these — the
  values were validated against the published numbers.

## What is NOT reproducible from this repo

- **LLM-ICL row** — Phase 10.1d ran on local Ollama; the run was partial
  (CPU-bound at 30–60 s per inference). The frontier-API replication is
  open as Phase 10.1f. See `results/ocrr_llm_icl_partial.md` for the
  partial findings.

## If the numbers don't match

Please file an issue with:
1. Your Python / OS / GPU info
2. `pip freeze` output
3. The diff between your `_repro_*_summary.csv` and the published one
4. The first checkpoint where they diverge (from the per-checkpoint CSV)

Numerical drift > 1 % on the published mean is treated as a bug.
