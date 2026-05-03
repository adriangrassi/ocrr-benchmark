"""OCRR vote-rule ablation on the substrate.

Tests whether the substrate's full vote rule (margin-band majority count
+ max-similarity tiebreak + recency tiebreak) is overdetermined. Five
variants:

  substrate_k1            k=1 nearest-neighbour, no voting
  substrate_sumsim        sum-of-similarities (no margin, no recency)
  substrate_count_only    margin-band count + insertion-order tiebreak
  substrate_no_recency    margin-band count + max_sim tiebreak (no recency)
  substrate (full)        margin-band count + max_sim + recency

Cell: banking77 / oracle / 3 seeds. Quick sweep.

Reports each variant's final novel/original + corrections-to-N% so the
paper's Section 6.x ablation table writes itself.
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
    RunResult, corrections_to_accuracy, final_accuracies,
    policy_oracle, run_ocrr,
)
from ocrr_benchmark.eval.ocrr_systems import SubstrateSystem
from ocrr_benchmark.eval.ocrr_ablations import (
    SubstrateCountOnlySystem,
    SubstrateK1System,
    SubstrateNoRecencySystem,
    SubstrateSumSimSystem,
)


PRED_DIR = Path("data/predictions")
B77_TRAIN_EMB = PRED_DIR / "bge_large_train_emb.npy"
B77_TEST_EMB = PRED_DIR / "bge_large_test_emb.npy"
RESEARCH_DIR = Path("research")


def _norm(emb):
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    return (emb / np.clip(n, 1e-9, None)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-held-out", type=int, default=10)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    args = ap.parse_args()
    targets = [0.10, 0.30, 0.50, 0.70, 0.90]

    print("=" * 78, flush=True)
    print("  OCRR vote-rule ablation — substrate variants", flush=True)
    print(f"  cell: banking77 / oracle / seeds={args.seeds}", flush=True)
    print("=" * 78, flush=True)

    train_emb = _norm(np.load(B77_TRAIN_EMB))
    test_emb = _norm(np.load(B77_TEST_EMB))
    train, test, labels = load_banking77()
    train_y = [ex.label for ex in train]
    test_y = [ex.label for ex in test]
    labels = list(labels)

    all_results = defaultdict(list)
    grand_t0 = time.time()
    for seed in args.seeds:
        print(f"\n[seed={seed}]", flush=True)
        rng = np.random.default_rng(seed)
        held = set(labels[int(i)] for i in
                   rng.choice(len(labels), size=args.n_held_out, replace=False))
        known_mask = np.array([l not in held for l in train_y])
        seed_v = train_emb[known_mask]
        seed_y = [l for m, l in zip(known_mask, train_y) if m]
        stream_v = train_emb[~known_mask]
        stream_y = [l for m, l in zip(~known_mask, train_y) if m]
        novel_mask = np.array([l in held for l in test_y])
        eval_novel_v = test_emb[novel_mask]
        eval_novel_y = [l for m, l in zip(novel_mask, test_y) if m]
        orig_v = test_emb[~novel_mask]
        orig_y = [l for m, l in zip(~novel_mask, test_y) if m]
        for vs, ys, name in [
            (orig_v, orig_y, "orig"),
            (eval_novel_v, eval_novel_y, "novel"),
        ]:
            if len(vs) > 400:
                idx = rng.choice(len(vs), size=400, replace=False)
                if name == "orig":
                    orig_v = vs[idx]; orig_y = [ys[int(i)] for i in idx]
                else:
                    eval_novel_v = vs[idx]; eval_novel_y = [ys[int(i)] for i in idx]
        perm = rng.permutation(len(stream_v))
        stream_v = stream_v[perm]
        stream_y = [stream_y[int(i)] for i in perm]
        eval_sets = {
            "novel":    (eval_novel_v, eval_novel_y),
            "original": (orig_v, orig_y),
        }

        factories = [
            ("substrate_k1",         lambda: SubstrateK1System(seed_v, seed_y)),
            ("substrate_sumsim",     lambda: SubstrateSumSimSystem(seed_v, seed_y)),
            ("substrate_count_only", lambda: SubstrateCountOnlySystem(seed_v, seed_y)),
            ("substrate_no_recency", lambda: SubstrateNoRecencySystem(seed_v, seed_y)),
            ("substrate (full)",     lambda: SubstrateSystem(seed_v, seed_y, k=5, margin=0.05)),
        ]
        for name, factory in factories:
            np.random.seed(seed)
            torch.manual_seed(seed)
            t0 = time.time()
            system = factory()
            init_s = time.time() - t0
            r = run_ocrr(
                system, stream_v, stream_y, eval_sets,
                checkpoint_every=args.checkpoint_every,
                correction_policy=policy_oracle,
                print_progress=False,
            )
            r.system_name = name
            finals = final_accuracies(r)
            print(f"  {name:>22}  novel={finals.get('novel', 0):.4f}  "
                  f"orig={finals.get('original', 0):.4f}  "
                  f"corr={r.checkpoints[-1].corrections_so_far}  init={init_s:.1f}s",
                  flush=True)
            all_results[name].append(r)

    print(f"\n[done] {time.time() - grand_t0:.0f}s", flush=True)

    # Aggregate
    summary = []
    for name, runs in all_results.items():
        finals_n = [final_accuracies(r).get("novel", 0.0) for r in runs]
        finals_o = [final_accuracies(r).get("original", 0.0) for r in runs]
        row = {
            "system": name, "n_seeds": len(runs),
            "final_novel_mean": float(np.mean(finals_n)),
            "final_novel_std":  float(np.std(finals_n)),
            "final_orig_mean":  float(np.mean(finals_o)),
            "final_orig_std":   float(np.std(finals_o)),
        }
        for t in targets:
            cs = [corrections_to_accuracy(r, "novel", t) for r in runs]
            cs_filt = [c for c in cs if c is not None]
            row[f"to_{int(t*100)}_mean"] = (float(np.mean(cs_filt)) if cs_filt else None)
            row[f"to_{int(t*100)}_n_reached"] = len(cs_filt)
        summary.append(row)

    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESEARCH_DIR / "ocrr_ablation_summary.csv"
    fields = ["system", "n_seeds",
              "final_novel_mean", "final_novel_std",
              "final_orig_mean", "final_orig_std"]
    for t in targets:
        fields += [f"to_{int(t*100)}_mean", f"to_{int(t*100)}_n_reached"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary:
            w.writerow({k: r.get(k) for k in fields})
    print(f"[csv] {summary_path}", flush=True)

    print()
    print("=" * 90)
    print("  Vote-rule ablation summary (banking77 / oracle, 3 seeds)")
    print("=" * 90)
    print(f"  {'system':>22}  {'final_novel':>13}  {'final_orig':>13}  "
          f"{'->10%':>7}  {'->50%':>7}  {'->70%':>7}  {'->90%':>7}")
    print("  " + "-" * 86)
    for r in sorted(summary, key=lambda r: -r["final_novel_mean"]):
        novel = f"{r['final_novel_mean']:.4f}±{r['final_novel_std']:.3f}"
        orig = f"{r['final_orig_mean']:.4f}±{r['final_orig_std']:.3f}"
        def fmt(t):
            v = r.get(f"to_{t}_mean")
            return f"{int(v)}" if v is not None else "never"
        print(f"  {r['system']:>22}  {novel:>13}  {orig:>13}  "
              f"{fmt(10):>7}  {fmt(50):>7}  {fmt(70):>7}  {fmt(90):>7}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
