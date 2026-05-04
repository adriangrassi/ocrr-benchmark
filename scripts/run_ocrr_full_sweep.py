"""OCRR — full sweep with strong external baselines.

Extends `run_ocrr_sweep.py` (which had 4 systems) to 9 systems:

  Strawman / sanity baselines (from ocrr_systems.py):
    substrate          ours
    static_knn         frozen vector index, no updates
    static_linear      frozen 67-output linear head
    online_linear      77-output linear head + per-correction SGD

  Strong algorithm-level baselines (from ocrr_baselines.py):
    ewc                Elastic Weight Consolidation (Kirkpatrick 2017)
    a_gem              Averaged Gradient Episodic Memory (Chaudhry 2019)
    lwf                Learning without Forgetting (Li & Hoiem 2017)
    knn_lm             retrieval/parametric mixture (Khandelwal 2020)
    river_logreg       online logistic regression from `river` library

The LLM in-context-learning baseline is run separately (see
`scripts/run_ocrr_llm_icl.py`) because it needs query text, not just
embeddings, and is too slow for the full 18-cell grid.

Total: 2 datasets x 3 policies x 3 seeds x 9 systems = 162 runs.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ocrr_benchmark.datasets import load_banking77
from ocrr_benchmark.datasets.clinc150 import load_clinc150
from ocrr_benchmark.eval.ocrr import (
    CorrectionPolicy,
    RunResult,
    corrections_to_accuracy,
    final_accuracies,
    policy_oracle,
    policy_random,
    run_ocrr,
)
from ocrr_benchmark.eval.ocrr_systems import (
    OnlineLinearSystem,
    StaticKNNSystem,
    StaticLinearSystem,
    SubstrateSystem,
)
from ocrr_benchmark.eval.ocrr_baselines import (
    AGEMSystem,
    EWCSystem,
    KNNLMSystem,
    LwFSystem,
    RiverLogRegSystem,
)


PRED_DIR = Path("data/predictions")
B77_TRAIN_EMB = PRED_DIR / "bge_large_train_emb.npy"
B77_TEST_EMB = PRED_DIR / "bge_large_test_emb.npy"
SEED_CACHE = Path("data/cache")  # populated by scripts/precompute_embeddings.py
CLINC_COMBINED_PT = SEED_CACHE / "clinc150_combined_BAAI_bge_large_en_v1.5.pt"
CLINC_TEST_PT = SEED_CACHE / "clinc150_test_BAAI_bge_large_en_v1.5.pt"
RESEARCH_DIR = Path("research")


def _norm(emb: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    return (emb / np.clip(n, 1e-9, None)).astype(np.float32)


def load_b77_data():
    train_emb = _norm(np.load(B77_TRAIN_EMB))
    test_emb = _norm(np.load(B77_TEST_EMB))
    train, test, label_names = load_banking77()
    return (
        train_emb, [ex.label for ex in train],
        test_emb, [ex.label for ex in test],
        list(label_names),
    )


def load_clinc_data():
    train, test, label_names = load_clinc150()
    combined = torch.load(CLINC_COMBINED_PT, map_location="cpu",
                          weights_only=False).numpy()
    test_emb = torch.load(CLINC_TEST_PT, map_location="cpu",
                          weights_only=False).numpy()
    train_emb = _norm(combined[: len(train)])
    test_emb = _norm(test_emb)
    return (
        train_emb, [ex.label for ex in train],
        test_emb, [ex.label for ex in test],
        list(label_names),
    )


def select_held_out(label_names, n_held, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(label_names), size=n_held, replace=False)
    return set(label_names[int(i)] for i in idx)


def split_known_held(emb, labels, held):
    known_mask = np.array([l not in held for l in labels])
    held_mask = ~known_mask
    return (
        emb[known_mask], [l for m, l in zip(known_mask, labels) if m],
        emb[held_mask],  [l for m, l in zip(held_mask, labels) if m],
    )


def run_one_config(dataset, policy_name, seed, n_held, *,
                   eval_cap=400, checkpoint_every=50, system_subset=None,
                   print_progress=False):
    """All 9 systems for one (dataset, policy, seed) cell."""
    if dataset == "banking77":
        tr_emb, tr_y, te_emb, te_y, labels = load_b77_data()
    else:
        tr_emb, tr_y, te_emb, te_y, labels = load_clinc_data()

    held = select_held_out(labels, n_held, seed=seed)
    known = sorted(c for c in labels if c not in held)

    seed_v, seed_y, stream_v_all, stream_y_all = split_known_held(tr_emb, tr_y, held)

    novel_mask = np.array([l in held for l in te_y])
    eval_novel_v = te_emb[novel_mask]
    eval_novel_y = [l for m, l in zip(novel_mask, te_y) if m]
    orig_v = te_emb[~novel_mask]
    orig_y = [l for m, l in zip(~novel_mask, te_y) if m]
    rng = np.random.default_rng(seed)
    if len(orig_v) > eval_cap:
        idx = rng.choice(len(orig_v), size=eval_cap, replace=False)
        orig_v = orig_v[idx]; orig_y = [orig_y[int(i)] for i in idx]
    if len(eval_novel_v) > eval_cap:
        idx = rng.choice(len(eval_novel_v), size=eval_cap, replace=False)
        eval_novel_v = eval_novel_v[idx]; eval_novel_y = [eval_novel_y[int(i)] for i in idx]

    perm = rng.permutation(len(stream_v_all))
    stream_v = stream_v_all[perm]
    stream_y = [stream_y_all[int(i)] for i in perm]

    eval_sets = {
        "novel":    (eval_novel_v, eval_novel_y),
        "original": (orig_v, orig_y),
    }

    systems_factory = {
        "substrate":     lambda: SubstrateSystem(seed_v, seed_y, k=5, margin=0.05),
        "static_knn":    lambda: StaticKNNSystem(seed_v, seed_y, k=5),
        "static_linear": lambda: StaticLinearSystem(seed_v, seed_y, known, seed=seed),
        "online_linear": lambda: OnlineLinearSystem(seed_v, seed_y, labels, init_seed=seed),
        "ewc":           lambda: EWCSystem(seed_v, seed_y, labels, init_seed=seed),
        "a_gem":         lambda: AGEMSystem(seed_v, seed_y, labels, init_seed=seed),
        "lwf":           lambda: LwFSystem(seed_v, seed_y, labels, init_seed=seed),
        "knn_lm":        lambda: KNNLMSystem(seed_v, seed_y, labels, init_seed=seed),
        "river_logreg":  lambda: RiverLogRegSystem(seed_v, seed_y, labels, init_seed=seed),
    }
    if system_subset is not None:
        systems_factory = {k: v for k, v in systems_factory.items() if k in system_subset}

    results = {}
    for name, factory in systems_factory.items():
        np.random.seed(seed)
        torch.manual_seed(seed)
        if policy_name == "oracle":
            p_use = policy_oracle
        elif policy_name == "random50":
            p_use = policy_random(0.50, seed=seed * 31 + hash(name) % 997)
        else:
            p_use = policy_random(0.10, seed=seed * 31 + hash(name) % 997)
        t0 = time.time()
        system = factory()
        init_secs = time.time() - t0
        result = run_ocrr(
            system, stream_v, stream_y, eval_sets,
            checkpoint_every=checkpoint_every,
            correction_policy=p_use,
            print_progress=print_progress,
        )
        result.system_name = name
        results[name] = (result, init_secs)
    return results


def aggregate_per_cell(all_results, targets):
    per_cell = defaultdict(list)
    for (ds, pol, _seed), sysmap in all_results.items():
        for sys_name, (result, _init_secs) in sysmap.items():
            per_cell[(ds, pol, sys_name)].append(result)
    rows = []
    for (ds, pol, sys_name), runs in sorted(per_cell.items()):
        finals_novel = [final_accuracies(r).get("novel", 0.0) for r in runs]
        finals_orig = [final_accuracies(r).get("original", 0.0) for r in runs]
        row = {
            "dataset": ds, "policy": pol, "system": sys_name,
            "n_seeds": len(runs),
            "final_novel_mean": float(np.mean(finals_novel)),
            "final_novel_std":  float(np.std(finals_novel)),
            "final_orig_mean":  float(np.mean(finals_orig)),
            "final_orig_std":   float(np.std(finals_orig)),
        }
        for t in targets:
            cs = [corrections_to_accuracy(r, "novel", t) for r in runs]
            cs_filt = [c for c in cs if c is not None]
            if not cs_filt:
                row[f"to_{int(t*100)}_mean"] = None
                row[f"to_{int(t*100)}_std"] = None
            else:
                row[f"to_{int(t*100)}_mean"] = float(np.mean(cs_filt))
                row[f"to_{int(t*100)}_std"] = float(np.std(cs_filt)) if len(cs_filt) > 1 else 0.0
            row[f"to_{int(t*100)}_n_reached"] = len(cs_filt)
        rows.append(row)
    return rows


def write_per_run_csv(all_results, csv_path):
    rows = []
    for (ds, pol, seed), sysmap in all_results.items():
        for sys_name, (result, _init) in sysmap.items():
            for cp in result.checkpoints:
                row = {
                    "dataset": ds, "policy": pol, "seed": seed,
                    "system": sys_name,
                    "step": cp.step,
                    "corrections": cp.corrections_so_far,
                    "pred_secs": round(cp.pred_secs, 4),
                    "correct_secs": round(cp.correct_secs, 4),
                    "acc_novel": round(cp.accuracies.get("novel", 0.0), 6),
                    "acc_original": round(cp.accuracies.get("original", 0.0), 6),
                }
                rows.append(row)
    fields = ["dataset", "policy", "seed", "system", "step", "corrections",
              "acc_novel", "acc_original", "pred_secs", "correct_secs"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_summary_csv(rows, csv_path, targets):
    fields = ["dataset", "policy", "system", "n_seeds",
              "final_novel_mean", "final_novel_std",
              "final_orig_mean", "final_orig_std"]
    for t in targets:
        fields += [f"to_{int(t*100)}_mean", f"to_{int(t*100)}_std",
                   f"to_{int(t*100)}_n_reached"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-held-out", type=int, default=10)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--datasets", type=str, nargs="+", default=["banking77", "clinc150"])
    ap.add_argument("--policies", type=str, nargs="+",
                    default=["oracle", "random50", "random10"])
    ap.add_argument("--systems", type=str, nargs="*", default=None,
                    help="optional subset of systems")
    args = ap.parse_args()
    targets = [0.10, 0.30, 0.50, 0.70, 0.90]

    print("=" * 78, flush=True)
    print("  OCRR full sweep — 9 systems incl. EWC / A-GEM / LwF / kNN-LM / river",
          flush=True)
    print(f"  datasets={args.datasets}", flush=True)
    print(f"  policies={args.policies}", flush=True)
    print(f"  seeds={args.seeds}", flush=True)
    n_systems = len(args.systems) if args.systems else 9
    print(f"  {n_systems} systems x {len(args.datasets)*len(args.policies)*len(args.seeds)} cells "
          f"= {n_systems*len(args.datasets)*len(args.policies)*len(args.seeds)} runs",
          flush=True)
    print("=" * 78, flush=True)

    all_results = {}
    cell_idx = 0
    n_cells = len(args.datasets) * len(args.policies) * len(args.seeds)
    grand_t0 = time.time()

    for ds in args.datasets:
        for pol in args.policies:
            for seed in args.seeds:
                cell_idx += 1
                t0 = time.time()
                print(f"\n[{cell_idx}/{n_cells}] {ds} / policy={pol} / seed={seed}",
                      flush=True)
                results = run_one_config(
                    ds, pol, seed, args.n_held_out,
                    checkpoint_every=args.checkpoint_every,
                    system_subset=set(args.systems) if args.systems else None,
                    print_progress=False,
                )
                for name, (r, init_s) in results.items():
                    finals = final_accuracies(r)
                    print(
                        f"    {name:>14}  novel={finals.get('novel', 0):.4f}  "
                        f"orig={finals.get('original', 0):.4f}  "
                        f"corr={r.checkpoints[-1].corrections_so_far}  "
                        f"(init {init_s:.1f}s)",
                        flush=True,
                    )
                all_results[(ds, pol, seed)] = results
                print(f"  cell took {time.time() - t0:.1f}s "
                      f"(elapsed {time.time() - grand_t0:.0f}s)", flush=True)

    print(f"\n[done] sweep took {time.time() - grand_t0:.0f}s", flush=True)

    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    per_run = RESEARCH_DIR / "ocrr_full_sweep_results.csv"
    write_per_run_csv(all_results, per_run)
    print(f"[csv] {per_run}", flush=True)

    summary = aggregate_per_cell(all_results, targets)
    summary_path = RESEARCH_DIR / "ocrr_full_sweep_summary.csv"
    write_summary_csv(summary, summary_path, targets)
    print(f"[csv] {summary_path}", flush=True)

    # Print compact summary table grouped by (dataset, policy)
    print()
    print("=" * 110)
    print("  Aggregated summary (mean +/- std over seeds, all 9 systems)")
    print("=" * 110)
    by_cell = defaultdict(list)
    for r in summary:
        by_cell[(r["dataset"], r["policy"])].append(r)
    for (ds, pol), rows in sorted(by_cell.items()):
        print(f"\n  --- {ds} / {pol} ---")
        print(f"  {'system':>14}  {'final_novel':>13}  {'final_orig':>13}  "
              f"{'->10%':>9}  {'->50%':>9}  {'->70%':>9}")
        # Sort: substrate first, then by final_novel desc
        def sort_key(r):
            return (0 if r["system"] == "substrate" else 1, -r["final_novel_mean"])
        for r in sorted(rows, key=sort_key):
            novel = f"{r['final_novel_mean']:.4f}+/-{r['final_novel_std']:.3f}"
            orig = f"{r['final_orig_mean']:.4f}+/-{r['final_orig_std']:.3f}"
            def fmt(t):
                m = r.get(f"to_{t}_mean")
                if m is None:
                    return "never"
                s = r.get(f"to_{t}_std", 0)
                return f"{int(m)}+/-{int(s)}"
            print(f"  {r['system']:>14}  {novel:>13}  {orig:>13}  "
                  f"{fmt(10):>9}  {fmt(50):>9}  {fmt(70):>9}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
