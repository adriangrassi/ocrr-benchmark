"""OCRR storage-vs-recovery sweep — bounded substrate variants.

Tests whether the substrate's dominance is an artefact of unbounded
storage. Sweeps `BoundedSubstrateSystem` at several memory budgets,
under both reservoir-sampling and FIFO eviction policies.

Cell: banking77 / oracle / seed in {0, 1, 2}.

Systems compared:
  substrate                       — unbounded ledger (full retention)
  bounded_substrate_reservoir_*   — capped at {100, 500, 1000, 5000}
  bounded_substrate_fifo_*        — same caps with FIFO eviction
  online_linear                   — model parameters only (~80 KB), no buffer
  river_logreg                    — model parameters only, online ML library
  a_gem                           — 1000-example reservoir + linear head

Outputs:
  research/ocrr_storage_sweep_results.csv
  research/ocrr_storage_sweep_summary.csv
  research/ocrr_storage_sweep_plot.png    storage-vs-novel Pareto curve
  research/ocrr_storage_sweep.log
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
from ocrr_benchmark.eval.ocrr import (
    RunResult,
    corrections_to_accuracy,
    final_accuracies,
    policy_oracle,
    run_ocrr,
)
from ocrr_benchmark.eval.ocrr_systems import OnlineLinearSystem, SubstrateSystem
from ocrr_benchmark.eval.ocrr_baselines import (
    AGEMSystem,
    BoundedSubstrateSystem,
    RiverLogRegSystem,
)


PRED_DIR = Path("data/predictions")
B77_TRAIN_EMB = PRED_DIR / "bge_large_train_emb.npy"
B77_TEST_EMB = PRED_DIR / "bge_large_test_emb.npy"
RESEARCH_DIR = Path("research")


def _norm(emb):
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    return (emb / np.clip(n, 1e-9, None)).astype(np.float32)


def load_b77():
    train_emb = _norm(np.load(B77_TRAIN_EMB))
    test_emb = _norm(np.load(B77_TEST_EMB))
    train, test, label_names = load_banking77()
    return (
        train_emb, [ex.label for ex in train],
        test_emb, [ex.label for ex in test],
        list(label_names),
    )


def build_data(seed: int, n_held: int = 10):
    tr_emb, tr_y, te_emb, te_y, labels = load_b77()
    rng = np.random.default_rng(seed)
    held_idx = rng.choice(len(labels), size=n_held, replace=False)
    held = set(labels[int(i)] for i in held_idx)
    known = sorted(c for c in labels if c not in held)

    known_mask = np.array([l not in held for l in tr_y])
    seed_v = tr_emb[known_mask]
    seed_y = [l for m, l in zip(known_mask, tr_y) if m]
    stream_v = tr_emb[~known_mask]
    stream_y = [l for m, l in zip(~known_mask, tr_y) if m]

    novel_mask = np.array([l in held for l in te_y])
    novel_v = te_emb[novel_mask]; novel_y = [l for m, l in zip(novel_mask, te_y) if m]
    orig_v = te_emb[~novel_mask]; orig_y = [l for m, l in zip(~novel_mask, te_y) if m]

    eval_cap = 400
    if len(orig_v) > eval_cap:
        idx = rng.choice(len(orig_v), size=eval_cap, replace=False)
        orig_v = orig_v[idx]; orig_y = [orig_y[int(i)] for i in idx]
    if len(novel_v) > eval_cap:
        idx = rng.choice(len(novel_v), size=eval_cap, replace=False)
        novel_v = novel_v[idx]; novel_y = [novel_y[int(i)] for i in idx]

    perm = rng.permutation(len(stream_v))
    stream_v = stream_v[perm]
    stream_y = [stream_y[int(i)] for i in perm]

    return seed_v, seed_y, stream_v, stream_y, novel_v, novel_y, orig_v, orig_y, known, labels


def system_factories(seed_v, seed_y, known, all_labels, init_seed, budgets):
    factories = {
        "substrate": lambda: SubstrateSystem(seed_v, seed_y, k=5, margin=0.05),
        "online_linear": lambda: OnlineLinearSystem(seed_v, seed_y, all_labels, init_seed=init_seed),
        "a_gem": lambda: AGEMSystem(seed_v, seed_y, all_labels, init_seed=init_seed),
        "river_logreg": lambda: RiverLogRegSystem(seed_v, seed_y, all_labels, init_seed=init_seed),
    }
    for b in budgets:
        for evict in ("reservoir", "fifo"):
            name = f"bounded_substrate_{evict}_{b}"
            factories[name] = (
                lambda b=b, evict=evict:
                BoundedSubstrateSystem(
                    seed_v, seed_y, budget=b, eviction=evict, init_seed=init_seed
                )
            )
    return factories


def run_one_seed(seed: int, budgets: list[int], checkpoint_every: int):
    seed_v, seed_y, stream_v, stream_y, novel_v, novel_y, orig_v, orig_y, _known, labels = (
        build_data(seed)
    )
    eval_sets = {
        "novel":    (novel_v, novel_y),
        "original": (orig_v, orig_y),
    }
    factories = system_factories(seed_v, seed_y, None, labels, seed, budgets)

    results = {}
    for name, factory in factories.items():
        np.random.seed(seed)
        torch.manual_seed(seed)
        t0 = time.time()
        try:
            system = factory()
        except Exception as e:
            print(f"  [skip] {name} failed at init: {e}", flush=True)
            continue
        init_secs = time.time() - t0
        result = run_ocrr(
            system, stream_v, stream_y, eval_sets,
            checkpoint_every=checkpoint_every,
            correction_policy=policy_oracle,
            print_progress=False,
        )
        result.system_name = name
        # Track buffer size for bounded systems
        buf_size = None
        if hasattr(system, "buffer_size"):
            buf_size = system.buffer_size()
        results[name] = (result, init_secs, buf_size)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--budgets", type=int, nargs="+", default=[100, 500, 1000, 5000])
    ap.add_argument("--checkpoint-every", type=int, default=50)
    args = ap.parse_args()
    targets = [0.10, 0.30, 0.50, 0.70]

    print("=" * 78, flush=True)
    print("  OCRR storage-vs-recovery sweep — bounded substrate variants", flush=True)
    print(f"  budgets={args.budgets}  seeds={args.seeds}", flush=True)
    print(f"  cell: banking77 / oracle", flush=True)
    print("=" * 78, flush=True)

    all_results = {}  # (seed) -> dict[name] -> (result, init_secs, buf_size)
    grand_t0 = time.time()
    for seed in args.seeds:
        print(f"\n[seed={seed}]", flush=True)
        t0 = time.time()
        results = run_one_seed(seed, args.budgets, args.checkpoint_every)
        for name, (r, init_s, buf) in results.items():
            finals = final_accuracies(r)
            buf_str = f" buf={buf}" if buf is not None else ""
            print(
                f"  {name:>32}  novel={finals.get('novel', 0):.4f}  "
                f"orig={finals.get('original', 0):.4f}  "
                f"corr={r.checkpoints[-1].corrections_so_far}  init={init_s:.1f}s{buf_str}",
                flush=True,
            )
        all_results[seed] = results
        print(f"  seed took {time.time() - t0:.1f}s "
              f"(elapsed {time.time() - grand_t0:.0f}s)", flush=True)

    print(f"\n[done] sweep took {time.time() - grand_t0:.0f}s", flush=True)

    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

    # ----------- per-run CSV -----------
    per_run_path = RESEARCH_DIR / "ocrr_storage_sweep_results.csv"
    rows = []
    for seed, sysmap in all_results.items():
        for name, (r, init_s, buf) in sysmap.items():
            for cp in r.checkpoints:
                rows.append({
                    "seed": seed, "system": name,
                    "step": cp.step, "corrections": cp.corrections_so_far,
                    "buffer_size": buf if buf is not None else "",
                    "init_secs": round(init_s, 4),
                    "pred_secs": round(cp.pred_secs, 4),
                    "correct_secs": round(cp.correct_secs, 4),
                    "acc_novel": round(cp.accuracies.get("novel", 0.0), 6),
                    "acc_original": round(cp.accuracies.get("original", 0.0), 6),
                })
    fields = ["seed", "system", "step", "corrections", "buffer_size",
              "init_secs", "pred_secs", "correct_secs",
              "acc_novel", "acc_original"]
    with open(per_run_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[csv] {per_run_path}", flush=True)

    # ----------- aggregate -----------
    per_sys = defaultdict(list)
    per_sys_buf = defaultdict(list)
    for seed, sysmap in all_results.items():
        for name, (r, _i, buf) in sysmap.items():
            per_sys[name].append(r)
            if buf is not None:
                per_sys_buf[name].append(buf)
    summary_rows = []
    for name, runs in per_sys.items():
        finals_n = [final_accuracies(r).get("novel", 0.0) for r in runs]
        finals_o = [final_accuracies(r).get("original", 0.0) for r in runs]
        row = {
            "system": name, "n_seeds": len(runs),
            "final_novel_mean": float(np.mean(finals_n)),
            "final_novel_std":  float(np.std(finals_n)),
            "final_orig_mean":  float(np.mean(finals_o)),
            "final_orig_std":   float(np.std(finals_o)),
            "mean_buffer_size": (float(np.mean(per_sys_buf[name]))
                                 if per_sys_buf[name] else "(N/A)"),
        }
        for t in targets:
            cs = [corrections_to_accuracy(r, "novel", t) for r in runs]
            cs_filt = [c for c in cs if c is not None]
            row[f"to_{int(t*100)}_mean"] = (float(np.mean(cs_filt)) if cs_filt else None)
            row[f"to_{int(t*100)}_n_reached"] = len(cs_filt)
        summary_rows.append(row)
    summary_path = RESEARCH_DIR / "ocrr_storage_sweep_summary.csv"
    fields = ["system", "n_seeds", "mean_buffer_size",
              "final_novel_mean", "final_novel_std",
              "final_orig_mean", "final_orig_std"]
    for t in targets:
        fields += [f"to_{int(t*100)}_mean", f"to_{int(t*100)}_n_reached"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"[csv] {summary_path}", flush=True)

    # ----------- print summary -----------
    print()
    print("=" * 110)
    print("  Storage-vs-recovery summary (banking77/oracle, 3 seeds, mean ± std)")
    print("=" * 110)
    def order(r):
        # substrate first, then bounded reservoir by budget desc, then fifo desc, then others
        if r["system"] == "substrate":
            return (0, 0, 0)
        if r["system"].startswith("bounded_substrate_reservoir_"):
            b = int(r["system"].split("_")[-1])
            return (1, -b, 0)
        if r["system"].startswith("bounded_substrate_fifo_"):
            b = int(r["system"].split("_")[-1])
            return (2, -b, 0)
        return (3, 0, hash(r["system"]) & 0xFFFF)
    summary_rows.sort(key=order)

    print(f"  {'system':>32}  {'buf':>7}  {'final_novel':>13}  {'final_orig':>13}  "
          f"{'->10%':>6}  {'->50%':>6}  {'->70%':>6}")
    print("  " + "-" * 102)
    for r in summary_rows:
        novel = f"{r['final_novel_mean']:.4f}±{r['final_novel_std']:.3f}"
        orig = f"{r['final_orig_mean']:.4f}±{r['final_orig_std']:.3f}"
        buf = (str(int(r["mean_buffer_size"])) if isinstance(r["mean_buffer_size"], float)
               else r["mean_buffer_size"])
        def fmt(t):
            v = r.get(f"to_{t}_mean")
            return f"{int(v)}" if v is not None else "never"
        print(f"  {r['system']:>32}  {buf:>7}  {novel:>13}  {orig:>13}  "
              f"{fmt(10):>6}  {fmt(50):>6}  {fmt(70):>6}")

    # ----------- plot -----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=120)
        # Recovery curves: substrate + each bounded variant
        for ax, key, title, ylim in [
            (axes[0], "novel", "Novel-class accuracy vs corrections", (-0.02, 1.02)),
            (axes[1], "original", "Original-distribution accuracy vs corrections", (-0.02, 1.02)),
        ]:
            for name, runs in per_sys.items():
                if name not in (
                    "substrate",
                    "bounded_substrate_reservoir_5000",
                    "bounded_substrate_reservoir_1000",
                    "bounded_substrate_reservoir_500",
                    "bounded_substrate_reservoir_100",
                    "river_logreg",
                    "a_gem",
                ):
                    continue
                min_len = min(len(r.checkpoints) for r in runs)
                xs = [runs[0].checkpoints[i].corrections_so_far for i in range(min_len)]
                ys = np.array([
                    [r.checkpoints[i].accuracies.get(key, 0.0) for i in range(min_len)]
                    for r in runs
                ])
                mean = ys.mean(axis=0)
                std = ys.std(axis=0)
                ax.plot(xs, mean, label=name, linewidth=2, marker="o", markersize=3)
                ax.fill_between(xs, mean - std, mean + std, alpha=0.12)
            ax.set_xlabel("corrections applied")
            ax.set_ylabel("accuracy")
            ax.set_title(title)
            ax.set_ylim(ylim)
            ax.grid(alpha=0.3)
            ax.legend(loc="best", fontsize=8)
        fig.suptitle("OCRR storage sweep — bounded substrate variants (banking77, oracle)")
        fig.tight_layout()
        plot1 = RESEARCH_DIR / "ocrr_storage_sweep_curves.png"
        fig.savefig(plot1)
        plt.close(fig)
        print(f"[plot] {plot1}", flush=True)

        # Pareto plot: storage on x-axis, final novel on y-axis
        fig2, ax = plt.subplots(figsize=(8, 5), dpi=120)
        # Substrate (full): a single point at huge storage
        substrate_row = next(r for r in summary_rows if r["system"] == "substrate")
        # Find all bounded reservoir/fifo points
        for evict, marker, label_prefix in [("reservoir", "o", "reservoir"), ("fifo", "s", "fifo")]:
            xs = []; ys = []; errs = []
            labels_pts = []
            for r in summary_rows:
                if r["system"].startswith(f"bounded_substrate_{evict}_"):
                    b = int(r["system"].split("_")[-1])
                    xs.append(b)
                    ys.append(r["final_novel_mean"])
                    errs.append(r["final_novel_std"])
                    labels_pts.append(b)
            order_idx = np.argsort(xs)
            xs = np.array(xs)[order_idx]; ys = np.array(ys)[order_idx]; errs = np.array(errs)[order_idx]
            ax.errorbar(xs, ys, yerr=errs, fmt=marker + "-", label=f"bounded {label_prefix}",
                        capsize=3, markersize=8, linewidth=2)
        # Substrate as a horizontal asymptote
        ax.axhline(y=substrate_row["final_novel_mean"], color="black", linestyle="--",
                   alpha=0.6, label=f"unbounded substrate ({substrate_row['final_novel_mean']:.3f})")
        # river_logreg / online_linear / a_gem as horizontal references
        for ref_name, ref_color in [
            ("river_logreg", "tab:red"),
            ("online_linear", "tab:gray"),
            ("a_gem", "tab:purple"),
        ]:
            ref_row = next((r for r in summary_rows if r["system"] == ref_name), None)
            if ref_row:
                ax.axhline(y=ref_row["final_novel_mean"], color=ref_color, linestyle=":",
                           alpha=0.6, label=f"{ref_name} ({ref_row['final_novel_mean']:.3f})")
        ax.set_xscale("log")
        ax.set_xlabel("memory budget (entries)")
        ax.set_ylabel("final novel-class accuracy")
        ax.set_title("OCRR storage-vs-recovery Pareto (banking77/oracle, 3 seeds)")
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)
        fig2.tight_layout()
        plot2 = RESEARCH_DIR / "ocrr_storage_sweep_pareto.png"
        fig2.savefig(plot2)
        plt.close(fig2)
        print(f"[plot] {plot2}", flush=True)
    except Exception as e:
        print(f"[plot] skipped: {e}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
