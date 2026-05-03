"""OCRR v1 (scope B) — held-out-classes shift on Banking77.

Setup:
  - 10 of 77 Banking77 classes are held out from initial state.
  - Each system is "primed" with the 67 known classes' train data.
  - Stream: held-out classes' train queries, in seeded random order.
  - Eval sets:
        novel    = held-out classes' test queries
        original = known 67 classes' test queries (forgetting check)
  - Correction policy: oracle (every wrong prediction -> correct(true_label)).

Four systems benchmarked:
  substrate      — bge-large + ImmutableLedger + margin-band majority vote
  static_knn     — same encoder, frozen 67-class index, never updates
  static_linear  — frozen 67-output linear head over bge-large
  online_linear  — 77-output linear head, per-correction SGD on the held-out
                   outputs (the "fine-tune-on-correction" baseline)

Outputs:
  research/ocrr_v1_results.csv      per-checkpoint metrics for all systems
  research/ocrr_v1_plot.png         recovery curves
  research/ocrr_v1_results.md       writeup with corrections-to-X table
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from ocrr_benchmark.datasets import load_banking77
from ocrr_benchmark.eval.ocrr import (
    RunResult,
    corrections_to_accuracy,
    final_accuracies,
    run_ocrr,
    to_csv_rows,
)
from ocrr_benchmark.eval.ocrr_systems import (
    OnlineLinearSystem,
    StaticKNNSystem,
    StaticLinearSystem,
    SubstrateSystem,
)


PRED_DIR = Path("data/predictions")
TRAIN_EMB = PRED_DIR / "bge_large_train_emb.npy"
TEST_EMB = PRED_DIR / "bge_large_test_emb.npy"
RESEARCH_DIR = Path("research")


def _normalise(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return (emb / np.clip(norms, 1e-9, None)).astype(np.float32)


def load_b77():
    if not TRAIN_EMB.exists() or not TEST_EMB.exists():
        raise FileNotFoundError("bge-large embeddings not cached")
    train_emb = _normalise(np.load(TRAIN_EMB))
    test_emb = _normalise(np.load(TEST_EMB))
    train, test, label_names = load_banking77()
    train_labels = [ex.label for ex in train]
    test_labels = [ex.label for ex in test]
    return train_emb, train_labels, test_emb, test_labels, list(label_names)


def select_held_out(label_names: list[str], n_held_out: int = 10, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(label_names), size=n_held_out, replace=False)
    return sorted(label_names[int(i)] for i in idx)


def split_known_held_out(
    emb: np.ndarray,
    labels: list[str],
    held_out: set[str],
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    known_mask = np.array([lbl not in held_out for lbl in labels], dtype=bool)
    held_mask = ~known_mask
    return (
        emb[known_mask], [l for m, l in zip(known_mask, labels) if m],
        emb[held_mask],  [l for m, l in zip(held_mask, labels) if m],
    )


def shuffled_stream(vecs: np.ndarray, labels: list[str], seed: int = 0
                    ) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(vecs))
    return vecs[idx], [labels[int(i)] for i in idx]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-held-out", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--orig-eval-cap", type=int, default=400,
                    help="max examples sampled from original-distribution eval")
    args = ap.parse_args()

    print("=" * 78, flush=True)
    print("  OCRR v1 (scope B) — Banking77, held-out-classes shift", flush=True)
    print("=" * 78, flush=True)

    print("[load] Banking77 + cached bge-large embeddings...", flush=True)
    tr_emb, tr_y, te_emb, te_y, label_names = load_b77()
    print(f"  train={tr_emb.shape}  test={te_emb.shape}  classes={len(label_names)}",
          flush=True)

    held_out = set(select_held_out(label_names, args.n_held_out, args.seed))
    known = [c for c in label_names if c not in held_out]
    print(f"\n[split] held-out ({len(held_out)}): {sorted(held_out)}", flush=True)
    print(f"        known    ({len(known)}): kept", flush=True)

    seed_vecs, seed_labels, stream_vecs_all, stream_labels_all = split_known_held_out(
        tr_emb, tr_y, held_out
    )
    eval_novel_v, eval_novel_y, eval_orig_v_all, eval_orig_y_all = (
        np.empty((0, te_emb.shape[1])), [], te_emb, te_y,
    )
    novel_mask = np.array([l in held_out for l in te_y])
    eval_novel_v = te_emb[novel_mask]
    eval_novel_y = [l for m, l in zip(novel_mask, te_y) if m]
    orig_mask = ~novel_mask
    orig_v = te_emb[orig_mask]
    orig_y = [l for m, l in zip(orig_mask, te_y) if m]
    # Cap the original-distribution eval to keep checkpoints fast.
    if len(orig_v) > args.orig_eval_cap:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(orig_v), size=args.orig_eval_cap, replace=False)
        orig_v = orig_v[idx]
        orig_y = [orig_y[int(i)] for i in idx]

    print(f"\n[seed]   {seed_vecs.shape[0]:>5} train entries (67 known classes)",
          flush=True)
    print(f"[stream] {stream_vecs_all.shape[0]:>5} train entries (10 held-out)",
          flush=True)
    print(f"[eval-novel]    {eval_novel_v.shape[0]:>5} test entries (held-out classes)",
          flush=True)
    print(f"[eval-original] {orig_v.shape[0]:>5} test entries (sampled, forgetting check)",
          flush=True)

    stream_vecs, stream_labels = shuffled_stream(
        stream_vecs_all, stream_labels_all, args.seed
    )

    eval_sets = {
        "novel":    (eval_novel_v, eval_novel_y),
        "original": (orig_v, orig_y),
    }

    # ----------------------------------------------------------------
    # Build the 4 systems
    # ----------------------------------------------------------------
    print("\n[build] systems...", flush=True)

    t0 = time.time()
    sys_substrate = SubstrateSystem(seed_vecs, seed_labels, k=5, margin=0.05)
    print(f"  substrate     ledger={len(sys_substrate.ledger)}  ({time.time() - t0:.1f}s)",
          flush=True)

    t0 = time.time()
    sys_knn = StaticKNNSystem(seed_vecs, seed_labels, k=5)
    print(f"  static_knn    seed={seed_vecs.shape[0]}  ({time.time() - t0:.1f}s)",
          flush=True)

    t0 = time.time()
    sys_static = StaticLinearSystem(seed_vecs, seed_labels, known, seed=args.seed)
    print(f"  static_linear out=67  ({time.time() - t0:.1f}s)", flush=True)

    t0 = time.time()
    sys_online = OnlineLinearSystem(
        seed_vecs, seed_labels, label_names, init_seed=args.seed,
    )
    print(f"  online_linear out=77  ({time.time() - t0:.1f}s)", flush=True)

    systems = [sys_substrate, sys_knn, sys_static, sys_online]

    # ----------------------------------------------------------------
    # Run OCRR for each system
    # ----------------------------------------------------------------
    results: list[RunResult] = []
    for system in systems:
        print(f"\n[run] {system.name}", flush=True)
        t0 = time.time()
        result = run_ocrr(
            system,
            stream_vecs, stream_labels,
            eval_sets,
            checkpoint_every=args.checkpoint_every,
            print_progress=True,
        )
        print(f"  done in {time.time() - t0:.1f}s", flush=True)
        results.append(result)

    # ----------------------------------------------------------------
    # Dump CSV
    # ----------------------------------------------------------------
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESEARCH_DIR / "ocrr_v1_results.csv"
    all_rows: list[dict] = []
    for result in results:
        all_rows.extend(to_csv_rows(result))
    if all_rows:
        fieldnames = sorted({k for row in all_rows for k in row.keys()})
        # Reorder so essentials come first
        priority = ["system", "step", "corrections", "acc_novel", "acc_original",
                    "pred_secs", "correct_secs"]
        ordered = [f for f in priority if f in fieldnames]
        ordered += [f for f in fieldnames if f not in ordered]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ordered)
            w.writeheader()
            for row in all_rows:
                w.writerow(row)
        print(f"\n[csv] {csv_path}", flush=True)

    # ----------------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------------
    print()
    print("=" * 78)
    print("  OCRR v1 results — Banking77, held-out-classes shift")
    print("=" * 78)
    targets = [0.10, 0.30, 0.50, 0.70, 0.90]
    header = (
        f"  {'system':>14}  {'final_novel':>11}  {'final_orig':>10}  "
        + "  ".join(f"{'->' + str(int(t * 100)) + '%':>8}" for t in targets)
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for result in results:
        finals = final_accuracies(result)
        cells = [
            corrections_to_accuracy(result, "novel", t) for t in targets
        ]
        cell_strs = [f"{c:>8d}" if c is not None else f"{'never':>8}" for c in cells]
        print(
            f"  {result.system_name:>14}  {finals.get('novel', 0):>11.4f}  "
            f"{finals.get('original', 0):>10.4f}  " + "  ".join(cell_strs)
        )

    # ----------------------------------------------------------------
    # Plot
    # ----------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=120)
        for ax, eval_name, title in [
            (axes[0], "novel", "Novel-class accuracy (10 held-out classes)"),
            (axes[1], "original", "Original-distribution accuracy (forgetting check)"),
        ]:
            for result in results:
                xs = [cp.corrections_so_far for cp in result.checkpoints]
                ys = [cp.accuracies.get(eval_name, 0.0) for cp in result.checkpoints]
                ax.plot(xs, ys, label=result.system_name, linewidth=2, marker="o", markersize=3)
            ax.set_xlabel("corrections applied")
            ax.set_ylabel("accuracy")
            ax.set_title(title)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)
            ax.legend(loc="best", fontsize=9)
        fig.suptitle(
            "OCRR v1 — Banking77, 10 held-out classes, oracle correction policy",
            fontsize=12,
        )
        fig.tight_layout()
        plot_path = RESEARCH_DIR / "ocrr_v1_plot.png"
        fig.savefig(plot_path)
        print(f"[plot] {plot_path}", flush=True)
    except Exception as e:
        print(f"[plot] skipped: {e}", flush=True)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
