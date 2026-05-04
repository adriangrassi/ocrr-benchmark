"""Precompute bge-large-en-v1.5 embeddings for Banking77 and CLINC150.

Creates the cached embedding files that all run_ocrr_*.py scripts depend
on. Run this once after `pip install -e .` and before any OCRR sweep.

Outputs:
  data/predictions/bge_large_train_emb.npy   (banking77 train)
  data/predictions/bge_large_test_emb.npy    (banking77 test)
  data/cache/clinc150_combined_BAAI_bge_large_en_v1.5.pt
  data/cache/clinc150_test_BAAI_bge_large_en_v1.5.pt

Wall time:
  ~10 min on RTX 4090 (cuda)
  ~15 min on Apple Silicon (mps)
  ~30-60 min on CPU

Usage:
  python scripts/precompute_embeddings.py                 # auto-pick device
  python scripts/precompute_embeddings.py --device cuda
  python scripts/precompute_embeddings.py --device mps
  python scripts/precompute_embeddings.py --device cpu
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ocrr_benchmark.datasets import load_banking77
from ocrr_benchmark.datasets.clinc150 import load_clinc150


PRED_DIR = Path("data/predictions")
CACHE_DIR = Path("data/cache")


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _maybe_skip(path: Path, label: str) -> bool:
    if path.exists():
        print(f"[precompute] {label} already exists at {path} ({path.stat().st_size / 1024**2:.1f} MB), skipping",
              flush=True)
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"],
                    help="Inference device (default: auto-detect)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--force", action="store_true",
                    help="Re-encode even if output files already exist")
    args = ap.parse_args()

    device = args.device or auto_device()
    print(f"[precompute] device={device}", flush=True)
    print(f"[precompute] model={args.model}", flush=True)
    print(f"[precompute] batch_size={args.batch_size}", flush=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        "b77_train": PRED_DIR / "bge_large_train_emb.npy",
        "b77_test": PRED_DIR / "bge_large_test_emb.npy",
        "clinc_combined": CACHE_DIR / "clinc150_combined_BAAI_bge_large_en_v1.5.pt",
        "clinc_test": CACHE_DIR / "clinc150_test_BAAI_bge_large_en_v1.5.pt",
    }
    if not args.force:
        all_present = all(p.exists() for p in paths.values())
        if all_present:
            print("[precompute] all 4 cache files already present; pass --force to re-encode",
                  flush=True)
            return 0

    # Lazy import — sentence_transformers is heavy
    print("[precompute] importing sentence_transformers (~few seconds)...", flush=True)
    from sentence_transformers import SentenceTransformer

    print("[precompute] loading encoder (downloads ~1.3 GB on first use)...", flush=True)
    t0 = time.time()
    encoder = SentenceTransformer(args.model, device=device)
    print(f"[precompute] encoder loaded in {time.time() - t0:.1f}s", flush=True)

    def encode(texts, label):
        print(f"[precompute] encoding {label} ({len(texts)} texts)...", flush=True)
        t0 = time.time()
        emb = encoder.encode(
            texts, batch_size=args.batch_size,
            show_progress_bar=True, convert_to_numpy=True,
        ).astype(np.float32)
        print(f"[precompute]   done in {time.time() - t0:.1f}s, shape={emb.shape}",
              flush=True)
        return emb

    # ---- Banking77 ---------------------------------------------------------
    if args.force or not (paths["b77_train"].exists() and paths["b77_test"].exists()):
        print("[precompute] loading banking77 dataset (downloads on first use)...", flush=True)
        b77_train, b77_test, _ = load_banking77()
        if args.force or not paths["b77_train"].exists():
            b77_train_emb = encode([ex.text for ex in b77_train], "banking77 train")
            np.save(paths["b77_train"], b77_train_emb)
            print(f"[precompute]   saved {paths['b77_train']}", flush=True)
        else:
            _maybe_skip(paths["b77_train"], "banking77 train")
        if args.force or not paths["b77_test"].exists():
            b77_test_emb = encode([ex.text for ex in b77_test], "banking77 test")
            np.save(paths["b77_test"], b77_test_emb)
            print(f"[precompute]   saved {paths['b77_test']}", flush=True)
        else:
            _maybe_skip(paths["b77_test"], "banking77 test")
    else:
        _maybe_skip(paths["b77_train"], "banking77 train")
        _maybe_skip(paths["b77_test"], "banking77 test")

    # ---- CLINC150 ----------------------------------------------------------
    need_clinc = (
        args.force
        or not paths["clinc_combined"].exists()
        or not paths["clinc_test"].exists()
    )
    if need_clinc:
        print("[precompute] loading clinc150 dataset (downloads via HuggingFace on first use)...",
              flush=True)
        clinc_train, clinc_test, _ = load_clinc150()
        clinc_train_emb = encode([ex.text for ex in clinc_train], "clinc150 train")
        clinc_test_emb = encode([ex.text for ex in clinc_test], "clinc150 test")
        # The original sweep scripts read CLINC150 as a "combined" tensor
        # (train concatenated with test, train portion = combined[: len(train)]).
        combined = torch.from_numpy(np.concatenate([clinc_train_emb, clinc_test_emb], axis=0))
        torch.save(combined, paths["clinc_combined"])
        print(f"[precompute]   saved {paths['clinc_combined']} shape={tuple(combined.shape)}",
              flush=True)
        torch.save(torch.from_numpy(clinc_test_emb), paths["clinc_test"])
        print(f"[precompute]   saved {paths['clinc_test']} shape={clinc_test_emb.shape}",
              flush=True)
    else:
        _maybe_skip(paths["clinc_combined"], "clinc150 combined")
        _maybe_skip(paths["clinc_test"], "clinc150 test")

    print("[precompute] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
