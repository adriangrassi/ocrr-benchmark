"""OCRR scope C — multi-dataset / multi-policy / multi-seed sweep.

Extends the v1 (scope B) harness:
  - 2 datasets: Banking77, CLINC150
  - 1 shift scenario: held-out classes (paraphrase deferred to v3)
  - 3 correction policies: oracle, random-50, random-10
  - 3 seeds per cell (error bars)
  - 4 systems: substrate, static_knn, static_linear, online_linear

Total: 2 datasets x 3 policies x 3 seeds x 4 systems = 72 runs.
Each run is fast (substrate sub-ms, linear heads sub-ms). Total ~30-60 min.

Outputs (research/):
  ocrr_sweep_results.csv     all per-checkpoint metrics, all 72 runs
  ocrr_sweep_summary.csv     per-cell mean +/- std at final / at thresholds
  ocrr_sweep_plot_*.png      per-(dataset, policy) recovery curves with ribbons
  ocrr_sweep_results.md      consolidated writeup
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict
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


PRED_DIR = Path("data/predictions")
B77_TRAIN_EMB = PRED_DIR / "bge_large_train_emb.npy"
B77_TEST_EMB = PRED_DIR / "bge_large_test_emb.npy"
SEED_CACHE = Path("C:/AI/DEV/Seed/data/cache")
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


def select_held_out(label_names: list[str], n_held: int, seed: int) -> set[str]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(label_names), size=n_held, replace=False)
    return set(label_names[int(i)] for i in idx)


def split_known_held(emb, labels, held: set[str]):
    known_mask = np.array([l not in held for l in labels])
    held_mask = ~known_mask
    return (
        emb[known_mask], [l for m, l in zip(known_mask, labels) if m],
        emb[held_mask],  [l for m, l in zip(held_mask, labels) if m],
    )


def run_one_config(
    dataset: str,
    policy_name: str,
    seed: int,
    n_held: int,
    *,
    eval_cap: int = 400,
    checkpoint_every: int = 50,
    print_progress: bool = False,
) -> dict[str, RunResult]:
    """Run all 4 systems for one (dataset, policy, seed) cell. Returns dict
    keyed by system name."""
    if dataset == "banking77":
        tr_emb, tr_y, te_emb, te_y, labels = load_b77_data()
    elif dataset == "clinc150":
        tr_emb, tr_y, te_emb, te_y, labels = load_clinc_data()
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    held = select_held_out(labels, n_held, seed=seed)
    known = sorted(c for c in labels if c not in held)

    seed_v, seed_y, stream_v_all, stream_y_all = split_known_held(tr_emb, tr_y, held)

    novel_mask = np.array([l in held for l in te_y])
    eval_novel_v = te_emb[novel_mask]
    eval_novel_y = [l for m, l in zip(novel_mask, te_y) if m]
    orig_v = te_emb[~novel_mask]
    orig_y = [l for m, l in zip(~novel_mask, te_y) if m]
    if len(orig_v) > eval_cap:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(orig_v), size=eval_cap, replace=False)
        orig_v = orig_v[idx]
        orig_y = [orig_y[int(i)] for i in idx]
    if len(eval_novel_v) > eval_cap:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(eval_novel_v), size=eval_cap, replace=False)
        eval_novel_v = eval_novel_v[idx]
        eval_novel_y = [eval_novel_y[int(i)] for i in idx]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(stream_v_all))
    stream_v = stream_v_all[perm]
    stream_y = [stream_y_all[int(i)] for i in perm]

    eval_sets = {
        "novel":    (eval_novel_v, eval_novel_y),
        "original": (orig_v, orig_y),
    }

    if policy_name == "oracle":
        policy: CorrectionPolicy = policy_oracle
    elif policy_name == "random50":
        policy = policy_random(0.50, seed=seed)
    elif policy_name == "random10":
        policy = policy_random(0.10, seed=seed)
    else:
        raise ValueError(f"unknown policy {policy_name!r}")

    systems = {
        "substrate":     SubstrateSystem(seed_v, seed_y, k=5, margin=0.05),
        "static_knn":    StaticKNNSystem(seed_v, seed_y, k=5),
        "static_linear": StaticLinearSystem(seed_v, seed_y, known, seed=seed),
        "online_linear": OnlineLinearSystem(seed_v, seed_y, labels, init_seed=seed),
    }

    results: dict[str, RunResult] = {}
    for name, system in systems.items():
        # Re-seed numpy / torch on each system so independent stochastic policies
        # don't drift across systems.
        np.random.seed(seed)
        torch.manual_seed(seed)
        if policy_name == "oracle":
            p_use: CorrectionPolicy = policy_oracle
        elif policy_name == "random50":
            p_use = policy_random(0.50, seed=seed * 31 + hash(name) % 997)
        else:
            p_use = policy_random(0.10, seed=seed * 31 + hash(name) % 997)
        result = run_ocrr(
            system, stream_v, stream_y, eval_sets,
            checkpoint_every=checkpoint_every,
            correction_policy=p_use,
            print_progress=print_progress,
        )
        results[name] = result
    return results


def aggregate_per_cell(
    all_results: dict[tuple[str, str, int], dict[str, RunResult]],
    targets: list[float],
) -> list[dict]:
    """For each (dataset, policy, system), aggregate over seeds.

    Returns one row per (dataset, policy, system) with mean + std across
    seeds for: final novel acc, final original acc, corrections-to-target."""
    per_cell: dict[tuple[str, str, str], list[RunResult]] = defaultdict(list)
    for (ds, pol, _seed), sysmap in all_results.items():
        for sys_name, result in sysmap.items():
            per_cell[(ds, pol, sys_name)].append(result)

    rows = []
    for (ds, pol, sys_name), runs in sorted(per_cell.items()):
        finals_novel = [final_accuracies(r).get("novel", 0.0) for r in runs]
        finals_orig = [final_accuracies(r).get("original", 0.0) for r in runs]
        row: dict = {
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
            n_reached = len(cs_filt)
            if n_reached == 0:
                row[f"to_{int(t*100)}_mean"] = None
                row[f"to_{int(t*100)}_std"] = None
            else:
                row[f"to_{int(t*100)}_mean"] = float(np.mean(cs_filt))
                row[f"to_{int(t*100)}_std"] = float(np.std(cs_filt)) if n_reached > 1 else 0.0
            row[f"to_{int(t*100)}_n_reached"] = n_reached
        rows.append(row)
    return rows


def write_per_run_csv(all_results, csv_path: Path) -> None:
    rows: list[dict] = []
    for (ds, pol, seed), sysmap in all_results.items():
        for sys_name, result in sysmap.items():
            for cp in result.checkpoints:
                row = {
                    "dataset": ds, "policy": pol, "seed": seed,
                    "system": sys_name,
                    "step": cp.step,
                    "corrections": cp.corrections_so_far,
                    "pred_secs": round(cp.pred_secs, 4),
                    "correct_secs": round(cp.correct_secs, 4),
                }
                for k, v in cp.accuracies.items():
                    row[f"acc_{k}"] = round(v, 6)
                rows.append(row)
    if not rows:
        return
    fields = ["dataset", "policy", "seed", "system", "step", "corrections",
              "acc_novel", "acc_original", "pred_secs", "correct_secs"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def write_summary_csv(rows: list[dict], csv_path: Path, targets: list[float]) -> None:
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


def plot_curves(all_results, dataset: str, policy: str, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [plot] skipped {dataset}/{policy}: {e}", flush=True)
        return

    # Collect curves per system, aligned by checkpoint step
    per_system: dict[str, list[list[tuple[int, float, float]]]] = defaultdict(list)
    for (ds, pol, seed), sysmap in all_results.items():
        if ds != dataset or pol != policy:
            continue
        for sys_name, result in sysmap.items():
            curve = [
                (cp.corrections_so_far,
                 cp.accuracies.get("novel", 0.0),
                 cp.accuracies.get("original", 0.0))
                for cp in result.checkpoints
            ]
            per_system[sys_name].append(curve)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=120)
    for ax, key, title in [
        (axes[0], 1, f"Novel-class accuracy ({dataset}, policy={policy})"),
        (axes[1], 2, f"Original-distribution accuracy ({dataset}, policy={policy})"),
    ]:
        for sys_name in ["substrate", "online_linear", "static_knn", "static_linear"]:
            curves = per_system.get(sys_name, [])
            if not curves:
                continue
            min_len = min(len(c) for c in curves)
            curves = [c[:min_len] for c in curves]
            xs = [c[0] for c in curves[0]]
            ys = np.array([[c[i][key] for i in range(min_len)] for c in curves])
            mean = ys.mean(axis=0)
            std = ys.std(axis=0)
            ax.plot(xs, mean, label=sys_name, linewidth=2, marker="o", markersize=3)
            ax.fill_between(xs, mean - std, mean + std, alpha=0.15)
        ax.set_xlabel("corrections applied")
        ax.set_ylabel("accuracy")
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    fig.suptitle(
        f"OCRR sweep — {dataset}, policy={policy} (held-out classes, mean ± std over seeds)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-held-out", type=int, default=10)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--datasets", type=str, nargs="+",
                    default=["banking77", "clinc150"])
    ap.add_argument("--policies", type=str, nargs="+",
                    default=["oracle", "random50", "random10"])
    args = ap.parse_args()

    targets = [0.10, 0.30, 0.50, 0.70, 0.90]

    print("=" * 78, flush=True)
    print(f"  OCRR scope C — sweep", flush=True)
    print(f"  datasets={args.datasets}", flush=True)
    print(f"  policies={args.policies}", flush=True)
    print(f"  seeds={args.seeds}", flush=True)
    print(f"  4 systems x {len(args.datasets)*len(args.policies)*len(args.seeds)} cells "
          f"= {4*len(args.datasets)*len(args.policies)*len(args.seeds)} runs", flush=True)
    print("=" * 78, flush=True)

    all_results: dict[tuple[str, str, int], dict[str, RunResult]] = {}

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
                    print_progress=False,
                )
                for name, r in results.items():
                    finals = final_accuracies(r)
                    print(f"    {name:>14}  novel={finals.get('novel', 0):.4f}  "
                          f"orig={finals.get('original', 0):.4f}  "
                          f"corr={r.checkpoints[-1].corrections_so_far}",
                          flush=True)
                all_results[(ds, pol, seed)] = results
                print(f"  cell took {time.time() - t0:.1f}s "
                      f"(elapsed {time.time() - grand_t0:.0f}s)", flush=True)

    print(f"\n[done] sweep took {time.time() - grand_t0:.0f}s", flush=True)

    # ---- write per-run CSV ----
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    per_run_path = RESEARCH_DIR / "ocrr_sweep_results.csv"
    write_per_run_csv(all_results, per_run_path)
    print(f"[csv] {per_run_path}", flush=True)

    # ---- aggregate per-cell ----
    summary_rows = aggregate_per_cell(all_results, targets)
    summary_path = RESEARCH_DIR / "ocrr_sweep_summary.csv"
    write_summary_csv(summary_rows, summary_path, targets)
    print(f"[csv] {summary_path}", flush=True)

    # ---- print summary table ----
    print()
    print("=" * 100)
    print("  Aggregated summary (mean +/- std over seeds)")
    print("=" * 100)
    print(f"  {'dataset':>10}  {'policy':>9}  {'system':>14}  "
          f"{'final_novel':>13}  {'final_orig':>13}  "
          f"{'->10%':>9}  {'->50%':>9}  {'->70%':>9}")
    print("  " + "-" * 96)
    for r in summary_rows:
        novel = f"{r['final_novel_mean']:.4f}+/-{r['final_novel_std']:.3f}"
        orig = f"{r['final_orig_mean']:.4f}+/-{r['final_orig_std']:.3f}"
        def fmt(t: int) -> str:
            m = r.get(f"to_{t}_mean")
            n = r.get(f"to_{t}_n_reached", 0)
            if m is None:
                return f"never({n})"
            s = r.get(f"to_{t}_std", 0)
            return f"{int(m)}+/-{int(s)}"
        print(f"  {r['dataset']:>10}  {r['policy']:>9}  {r['system']:>14}  "
              f"{novel:>13}  {orig:>13}  "
              f"{fmt(10):>9}  {fmt(50):>9}  {fmt(70):>9}")

    # ---- plots per (dataset, policy) ----
    print("\n[plot] generating...", flush=True)
    for ds in args.datasets:
        for pol in args.policies:
            out = RESEARCH_DIR / f"ocrr_sweep_plot_{ds}_{pol}.png"
            plot_curves(all_results, ds, pol, out)
            print(f"  {out}", flush=True)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
