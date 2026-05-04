"""Substrate scaling study: validate the never-forget property at scale.

Claims the OCRR paper makes today were validated up to ~10k stored entries.
At larger scales the substrate's no-forgetting guarantee depends on
HNSW retrieval recall, which is approximate and could in principle miss
the true nearest neighbor of a query and thus "forget" a rare-class
entry that physically lives in the ledger but isn't surfaced.

This script tests substrate retention at synthetic class-incremental
scales (10k, 100k, optionally 1M) to characterise:

  1. ``recall@k``: HNSW top-k vs brute-force top-k overlap
  2. ``hnsw_acc``: prediction accuracy with default (HNSW-when-available) path
  3. ``brute_acc``: prediction accuracy with force_brute=True (recall ceiling)
  4. ``agreement_rate``: how often HNSW and brute-force vote the same label
  5. ``pred_latency_ms``: per-query wall-time, both backends

A widening gap between hnsw_acc and brute_acc as scale grows is the
direct empirical signal that approximate retrieval is starting to cause
forgetting. A flat gap means the architecture's never-forget claim
holds at the tested scale.

Usage:
    python scripts/run_substrate_scale_study.py \\
        --scales 10000,100000 \\
        --classes 100 \\
        --dim 384 \\
        --queries 200 \\
        --seed 0 \\
        --output results/substrate_scale_study.csv

Synthetic data: each class is a Gaussian centroid in `dim`-D space.
Examples are centroid + Gaussian noise. This guarantees a clear correct
answer per query (the true class's centroid neighbourhood) so accuracy
gaps between HNSW and brute-force are unambiguously attributable to
retrieval rather than ambiguous labels.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from ocrr_benchmark.eval.ocrr_systems import SubstrateSystem


# ---------------------------------------------------------------- data helpers

def make_synthetic_corpus(
    n_examples: int, n_classes: int, dim: int, *, seed: int,
    noise_scale: float = 0.3,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Generate `n_examples` (vec, label) pairs and a class-centroid table.

    Returns (train_vecs, train_labels, centroids).
    `centroids[i]` is the prototype for class `i`. Training examples are
    generated as `centroid + N(0, noise_scale * I)`, which keeps each
    example unambiguously closest to its own centroid.
    """
    rng = np.random.default_rng(seed)
    centroids = rng.standard_normal((n_classes, dim)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True).clip(1e-12)
    examples_per_class = max(1, n_examples // n_classes)
    train_vecs = np.zeros((examples_per_class * n_classes, dim), dtype=np.float32)
    train_labels: list[str] = []
    for c in range(n_classes):
        start = c * examples_per_class
        train_vecs[start : start + examples_per_class] = (
            centroids[c]
            + rng.standard_normal((examples_per_class, dim)).astype(np.float32) * noise_scale
        )
        train_labels.extend([f"class_{c}"] * examples_per_class)
    return train_vecs, train_labels, centroids


def make_test_queries(
    centroids: np.ndarray, queries_per_class: int, *, seed: int,
    noise_scale: float = 0.3,
) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed + 1)
    n_classes, dim = centroids.shape
    test_vecs = np.zeros((queries_per_class * n_classes, dim), dtype=np.float32)
    test_labels: list[str] = []
    for c in range(n_classes):
        start = c * queries_per_class
        test_vecs[start : start + queries_per_class] = (
            centroids[c]
            + rng.standard_normal((queries_per_class, dim)).astype(np.float32) * noise_scale
        )
        test_labels.extend([f"class_{c}"] * queries_per_class)
    return test_vecs, test_labels


# ---------------------------------------------------------------- evaluation

def evaluate_substrate(
    substrate: SubstrateSystem, test_vecs: np.ndarray, test_labels: list[str],
) -> tuple[float, float, list[str | None]]:
    """Run the substrate over the test set, single-process.

    Returns (accuracy, total_pred_secs, predictions).
    """
    t0 = time.time()
    preds = [substrate.predict(v) for v in test_vecs]
    elapsed = time.time() - t0
    correct = sum(1 for p, lbl in zip(preds, test_labels) if p == lbl)
    return correct / len(test_vecs), elapsed, preds


# ---------------------------------------------------------------- driver

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scales", default="10000,100000",
        help="Comma-separated ledger sizes to test (e.g. '10000,100000,1000000')",
    )
    parser.add_argument("--classes", type=int, default=100)
    parser.add_argument("--dim", type=int, default=384, help="Embedding dim (bge-small=384, bge-large=1024)")
    parser.add_argument("--queries", type=int, default=200, help="Test queries (sampled across all classes)")
    parser.add_argument("--noise", type=float, default=0.3, help="Per-example noise scale around centroid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/substrate_scale_study.csv"))
    parser.add_argument(
        "--recall-sample-size", type=int, default=200,
        help="How many queries to sample for the HNSW recall@k metric",
    )
    args = parser.parse_args()

    scales = sorted(int(s) for s in args.scales.split(","))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[scale-study] scales={scales} classes={args.classes} dim={args.dim} "
          f"queries={args.queries}", flush=True)
    print(f"[scale-study] output -> {args.output}", flush=True)

    rows: list[dict] = []

    for scale in scales:
        scale_t0 = time.time()
        print(f"\n[scale-study] === scale={scale} ===", flush=True)

        # Generate corpus + test set
        train_vecs, train_labels, centroids = make_synthetic_corpus(
            n_examples=scale, n_classes=args.classes, dim=args.dim,
            seed=args.seed, noise_scale=args.noise,
        )
        actual_n = len(train_vecs)
        queries_per_class = max(1, args.queries // args.classes)
        test_vecs, test_labels = make_test_queries(
            centroids, queries_per_class=queries_per_class,
            seed=args.seed, noise_scale=args.noise,
        )
        print(f"[scale-study]   corpus={actual_n} entries, test={len(test_vecs)} queries",
              flush=True)

        # Build substrate (default backend: HNSW above ImmutableLedger threshold)
        print("[scale-study]   building HNSW substrate...", flush=True)
        t0 = time.time()
        substrate_hnsw = SubstrateSystem(train_vecs, train_labels, force_brute=False)
        build_secs_hnsw = time.time() - t0
        print(f"[scale-study]   HNSW substrate built in {build_secs_hnsw:.1f}s "
              f"(backend={substrate_hnsw.ledger.backend})", flush=True)

        # Build substrate (force brute-force - the recall ceiling)
        print("[scale-study]   building brute substrate...", flush=True)
        t0 = time.time()
        substrate_brute = SubstrateSystem(train_vecs, train_labels, force_brute=True)
        build_secs_brute = time.time() - t0
        print(f"[scale-study]   brute substrate built in {build_secs_brute:.1f}s",
              flush=True)

        # Recall@k between HNSW and brute, sampled
        recall_sample = test_vecs[: args.recall_sample_size]
        print(f"[scale-study]   measuring HNSW recall@k on {len(recall_sample)} queries...",
              flush=True)
        t0 = time.time()
        recall_metrics = substrate_hnsw.ledger.verify_hnsw_recall(
            recall_sample.astype(np.float32), k=substrate_hnsw._k,
        )
        recall_secs = time.time() - t0
        print(f"[scale-study]   recall@{recall_metrics['k']}={recall_metrics['recall_at_k']:.4f} "
              f"(backend={recall_metrics['backend']}, {recall_secs:.1f}s)", flush=True)

        # Prediction accuracy: HNSW path
        print("[scale-study]   evaluating HNSW substrate accuracy...", flush=True)
        hnsw_acc, hnsw_secs, hnsw_preds = evaluate_substrate(
            substrate_hnsw, test_vecs, test_labels,
        )
        print(f"[scale-study]   hnsw_acc={hnsw_acc:.4f} "
              f"({hnsw_secs:.1f}s total, {hnsw_secs / len(test_vecs) * 1000:.1f}ms/query)",
              flush=True)

        # Prediction accuracy: brute path (the recall ceiling)
        print("[scale-study]   evaluating brute substrate accuracy "
              "(this is the slow one)...", flush=True)
        brute_acc, brute_secs, brute_preds = evaluate_substrate(
            substrate_brute, test_vecs, test_labels,
        )
        print(f"[scale-study]   brute_acc={brute_acc:.4f} "
              f"({brute_secs:.1f}s total, {brute_secs / len(test_vecs) * 1000:.1f}ms/query)",
              flush=True)

        # Agreement and forgetting gap
        agreement = sum(1 for h, b in zip(hnsw_preds, brute_preds) if h == b) / len(hnsw_preds)
        forgetting_gap = brute_acc - hnsw_acc

        scale_secs = time.time() - scale_t0
        print(f"[scale-study]   agreement={agreement:.4f}  forgetting_gap="
              f"{forgetting_gap:+.4f}  scale_total={scale_secs:.1f}s", flush=True)

        rows.append({
            "scale": actual_n,
            "n_classes": args.classes,
            "dim": args.dim,
            "n_test_queries": len(test_vecs),
            "ledger_backend": substrate_hnsw.ledger.backend,
            "build_secs_hnsw": round(build_secs_hnsw, 2),
            "build_secs_brute": round(build_secs_brute, 2),
            "recall_at_k": round(recall_metrics["recall_at_k"], 4),
            "recall_sample_size": recall_metrics["num_queries"],
            "hnsw_acc": round(hnsw_acc, 4),
            "brute_acc": round(brute_acc, 4),
            "forgetting_gap": round(forgetting_gap, 4),
            "agreement_rate": round(agreement, 4),
            "hnsw_pred_ms_per_query": round(hnsw_secs / len(test_vecs) * 1000, 2),
            "brute_pred_ms_per_query": round(brute_secs / len(test_vecs) * 1000, 2),
            "scale_total_secs": round(scale_secs, 1),
        })

        # Free large matrices before next scale to bound RAM
        del substrate_hnsw, substrate_brute, train_vecs

    # Write CSV
    fieldnames = list(rows[0].keys()) if rows else []
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[scale-study] DONE - wrote {len(rows)} scale rows to {args.output}",
          flush=True)
    print("\n[scale-study] Summary:", flush=True)
    print(f"  {'scale':>10} {'recall@k':>9} {'hnsw_acc':>9} {'brute_acc':>10} "
          f"{'gap':>7} {'agree':>7}", flush=True)
    for r in rows:
        print(f"  {r['scale']:>10} {r['recall_at_k']:>9.4f} {r['hnsw_acc']:>9.4f} "
              f"{r['brute_acc']:>10.4f} {r['forgetting_gap']:>+7.4f} "
              f"{r['agreement_rate']:>7.4f}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
