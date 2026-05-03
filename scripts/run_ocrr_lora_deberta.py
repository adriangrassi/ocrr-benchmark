"""OCRR: LoRA-on-DeBERTa-v3-large baseline.

Strongest credible parametric fine-tune-on-correction baseline:
  - DeBERTa-v3-large encoder (frozen base, LoRA rank-8 adapters on
    query/value projections of every transformer block).
  - 77-output classification head on top of [CLS] (also trainable).
  - Per-correction: forward + backward through DeBERTa+LoRA, single
    AdamW step on adapter+head parameters.

Runs outside the standard OCRR harness because:
  - Needs raw text (the harness passes embedding vectors).
  - GPU-accelerated; per-step cost is dominated by DeBERTa forward.

Cell: banking77 / oracle / 3 seeds. Same shift scenario as the rest of
OCRR (10 held-out classes). Cached eval = 400 novel + 400 original
(the harness default). On RTX 4090, expected runtime: ~15-20 min/seed.

Substrate is run on the same shift slice (seeded identically) so the
comparison is apples-to-apples.

Outputs:
  research/ocrr_lora_deberta_results.csv  per-checkpoint metrics
  research/ocrr_lora_deberta_curves.png   recovery curves
  research/ocrr_lora_deberta.log          full stdout
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ocrr_benchmark.datasets import load_banking77
from ocrr_benchmark.eval.ocrr_systems import SubstrateSystem


PRED_DIR = Path("data/predictions")
B77_TRAIN_EMB = PRED_DIR / "bge_large_train_emb.npy"
B77_TEST_EMB = PRED_DIR / "bge_large_test_emb.npy"
RESEARCH_DIR = Path("research")


def _norm(emb):
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    return (emb / np.clip(n, 1e-9, None)).astype(np.float32)


# ============================================================================
# DeBERTa-v3-large + LoRA system
# ============================================================================

class DeBERTaLoRASystem:
    """DeBERTa-v3-large with LoRA adapters + a fresh classification head.

    On `correct(text, label)` we do one forward + backward + AdamW step on
    the adapter parameters (and the head). The base DeBERTa weights are
    frozen.
    """

    name = "lora_deberta"

    def __init__(self, seed_texts, seed_labels, all_classes, *,
                 init_seed=0, seed_epochs=2, sgd_lr=5e-4, lora_r=8,
                 lora_alpha=16, batch_size=16, device=None):
        try:
            from transformers import (
                AutoTokenizer, AutoModel, get_linear_schedule_with_warmup,
            )
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError as e:
            raise ImportError(
                "DeBERTaLoRASystem requires `transformers` + `peft`.\n"
                "    pip install transformers peft"
            ) from e

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._all_classes = list(all_classes)
        self._lbl_to_idx = {c: i for i, c in enumerate(all_classes)}
        self._idx_to_lbl = {i: c for c, i in self._lbl_to_idx.items()}
        self._sgd_lr = sgd_lr
        self._batch_size = batch_size
        self._init_seed = init_seed
        self._seed_epochs = seed_epochs
        self._lora_r = lora_r
        self._lora_alpha = lora_alpha
        self._seed_texts = list(seed_texts)
        self._seed_labels = list(seed_labels)

        # Backbone — load fresh each .reset() so seed=N runs are independent.
        self._model_name = "microsoft/deberta-v3-large"
        self._tok = AutoTokenizer.from_pretrained(self._model_name)
        self._lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["query_proj", "value_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self._AutoModel = AutoModel
        self._get_peft_model = get_peft_model
        self.reset()

    def reset(self):
        torch.manual_seed(self._init_seed)
        np.random.seed(self._init_seed)
        # Load fresh base model + LoRA wrapper
        base = self._AutoModel.from_pretrained(self._model_name)
        for p in base.parameters():
            p.requires_grad_(False)
        self._enc = self._get_peft_model(base, self._lora_cfg).to(self._device)
        # Classification head on top of [CLS]
        hidden = base.config.hidden_size
        self._head = nn.Linear(hidden, len(self._all_classes)).to(self._device)
        # Train head + adapters on the seed corpus for a few epochs.
        self._enc.train(); self._head.train()
        opt = torch.optim.AdamW(
            [p for p in self._enc.parameters() if p.requires_grad]
            + list(self._head.parameters()),
            lr=2e-4, weight_decay=1e-4,
        )
        n = len(self._seed_texts)
        rng = np.random.default_rng(self._init_seed)
        bs = self._batch_size
        for epoch in range(self._seed_epochs):
            perm = rng.permutation(n)
            for s in range(0, n, bs):
                idx = perm[s: s + bs]
                texts = [self._seed_texts[int(i)] for i in idx]
                labels = torch.tensor(
                    [self._lbl_to_idx[self._seed_labels[int(i)]] for i in idx],
                    dtype=torch.long, device=self._device,
                )
                enc_in = self._tok(texts, padding=True, truncation=True,
                                    max_length=64, return_tensors="pt")
                enc_in = {k: v.to(self._device) for k, v in enc_in.items()}
                opt.zero_grad()
                out = self._enc(**enc_in)
                cls_h = out.last_hidden_state[:, 0].float()
                logits = self._head(cls_h)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                opt.step()
        self._enc.eval(); self._head.eval()
        # Per-correction optimiser uses plain SGD, single example
        self._sgd = torch.optim.SGD(
            [p for p in self._enc.parameters() if p.requires_grad]
            + list(self._head.parameters()),
            lr=self._sgd_lr,
        )

    @torch.no_grad()
    def predict(self, text):
        self._enc.eval(); self._head.eval()
        enc_in = self._tok([text], padding=True, truncation=True,
                            max_length=64, return_tensors="pt")
        enc_in = {k: v.to(self._device) for k, v in enc_in.items()}
        out = self._enc(**enc_in)
        logits = self._head(out.last_hidden_state[:, 0].float())
        return self._idx_to_lbl.get(int(logits.argmax(dim=-1).item()))

    def correct(self, text, true_label):
        if true_label not in self._lbl_to_idx:
            return
        self._enc.train(); self._head.train()
        target = torch.tensor([self._lbl_to_idx[true_label]],
                               dtype=torch.long, device=self._device)
        enc_in = self._tok([text], padding=True, truncation=True,
                            max_length=64, return_tensors="pt")
        enc_in = {k: v.to(self._device) for k, v in enc_in.items()}
        self._sgd.zero_grad()
        out = self._enc(**enc_in)
        logits = self._head(out.last_hidden_state[:, 0].float())
        loss = F.cross_entropy(logits, target)
        loss.backward()
        self._sgd.step()
        self._enc.eval(); self._head.eval()


# ============================================================================
# Driver
# ============================================================================

def run_one_seed(seed, *, n_held_out=10, eval_cap=400, checkpoint_every=50,
                 lora_seed_epochs=2):
    train_emb = _norm(np.load(B77_TRAIN_EMB))
    test_emb = _norm(np.load(B77_TEST_EMB))
    train, test, label_names = load_banking77()
    train_text = [ex.text for ex in train]
    train_y = [ex.label for ex in train]
    test_text = [ex.text for ex in test]
    test_y = [ex.label for ex in test]
    label_names = list(label_names)

    rng = np.random.default_rng(seed)
    held = set(label_names[int(i)] for i in
               rng.choice(len(label_names), size=n_held_out, replace=False))
    known = sorted(c for c in label_names if c not in held)
    print(f"  held-out classes: {sorted(held)}", flush=True)

    known_mask = np.array([l not in held for l in train_y])
    seed_v = train_emb[known_mask]; seed_y = [l for m, l in zip(known_mask, train_y) if m]
    seed_t = [t for m, t in zip(known_mask, train_text) if m]
    stream_v = train_emb[~known_mask]; stream_y = [l for m, l in zip(~known_mask, train_y) if m]
    stream_t = [t for m, t in zip(~known_mask, train_text) if m]
    novel_mask = np.array([l in held for l in test_y])
    eval_novel_v = test_emb[novel_mask]
    eval_novel_y = [l for m, l in zip(novel_mask, test_y) if m]
    eval_novel_t = [t for m, t in zip(novel_mask, test_text) if m]
    orig_v = test_emb[~novel_mask]
    orig_y = [l for m, l in zip(~novel_mask, test_y) if m]
    orig_t = [t for m, t in zip(~novel_mask, test_text) if m]
    if len(orig_v) > eval_cap:
        idx = rng.choice(len(orig_v), size=eval_cap, replace=False)
        orig_v = orig_v[idx]; orig_y = [orig_y[int(i)] for i in idx]
        orig_t = [orig_t[int(i)] for i in idx]
    if len(eval_novel_v) > eval_cap:
        idx = rng.choice(len(eval_novel_v), size=eval_cap, replace=False)
        eval_novel_v = eval_novel_v[idx]; eval_novel_y = [eval_novel_y[int(i)] for i in idx]
        eval_novel_t = [eval_novel_t[int(i)] for i in idx]
    perm = rng.permutation(len(stream_v))
    stream_v = stream_v[perm]
    stream_y = [stream_y[int(i)] for i in perm]
    stream_t = [stream_t[int(i)] for i in perm]

    # Build LoRA system
    print(f"  [build] DeBERTa-v3-large + LoRA r=8 (seed_epochs={lora_seed_epochs})", flush=True)
    t0 = time.time()
    lora = DeBERTaLoRASystem(
        seed_t, seed_y, label_names,
        init_seed=seed, seed_epochs=lora_seed_epochs,
    )
    print(f"    init done in {time.time() - t0:.1f}s", flush=True)

    # Build substrate (reference)
    sub = SubstrateSystem(seed_v, seed_y, k=5, margin=0.05)

    def text_eval_lora(texts, ys):
        n_correct = 0
        for t, y in zip(texts, ys):
            if lora.predict(t) == y:
                n_correct += 1
        return n_correct / max(1, len(texts))

    def vec_eval_sub(vecs, ys):
        n_correct = 0
        for v, y in zip(vecs, ys):
            if sub.predict(v) == y:
                n_correct += 1
        return n_correct / max(1, len(vecs))

    rows = []  # (system, step, corr, novel, orig)

    print("\n  [run] lora_deberta", flush=True)
    corrections = 0
    t0 = time.time()
    n0 = text_eval_lora(eval_novel_t, eval_novel_y)
    o0 = text_eval_lora(orig_t, orig_y)
    rows.append(("lora_deberta", 0, 0, n0, o0))
    print(f"    [step    0  corr     0]  novel={n0:.4f}  orig={o0:.4f}  ({time.time() - t0:.1f}s)",
          flush=True)
    for i, (txt, lbl) in enumerate(zip(stream_t, stream_y)):
        pred = lora.predict(txt)
        if pred != lbl:
            lora.correct(txt, lbl)
            corrections += 1
        if (i + 1) % checkpoint_every == 0 or (i + 1) == len(stream_t):
            n_acc = text_eval_lora(eval_novel_t, eval_novel_y)
            o_acc = text_eval_lora(orig_t, orig_y)
            rows.append(("lora_deberta", i + 1, corrections, n_acc, o_acc))
            print(f"    [step {i + 1:>4}  corr {corrections:>4}]  novel={n_acc:.4f}  orig={o_acc:.4f}  "
                  f"(elapsed {time.time() - t0:.0f}s)", flush=True)

    print("\n  [run] substrate (reference, same shift slice)", flush=True)
    corrections = 0
    n0 = vec_eval_sub(eval_novel_v, eval_novel_y)
    o0 = vec_eval_sub(orig_v, orig_y)
    rows.append(("substrate", 0, 0, n0, o0))
    print(f"    [step    0  corr     0]  novel={n0:.4f}  orig={o0:.4f}", flush=True)
    for i, (vec, lbl) in enumerate(zip(stream_v, stream_y)):
        pred = sub.predict(vec)
        if pred != lbl:
            sub.correct(vec, lbl)
            corrections += 1
        if (i + 1) % checkpoint_every == 0 or (i + 1) == len(stream_v):
            n_acc = vec_eval_sub(eval_novel_v, eval_novel_y)
            o_acc = vec_eval_sub(orig_v, orig_y)
            rows.append(("substrate", i + 1, corrections, n_acc, o_acc))
            print(f"    [step {i + 1:>4}  corr {corrections:>4}]  novel={n_acc:.4f}  orig={o_acc:.4f}",
                  flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-held-out", type=int, default=10)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--eval-cap", type=int, default=400)
    ap.add_argument("--lora-seed-epochs", type=int, default=2)
    args = ap.parse_args()

    print("=" * 78, flush=True)
    print("  OCRR LoRA-DeBERTa baseline (banking77 / oracle)", flush=True)
    print(f"  seeds={args.seeds}  device={'cuda' if torch.cuda.is_available() else 'cpu'}",
          flush=True)
    print("=" * 78, flush=True)

    all_rows = []
    grand_t0 = time.time()
    for seed in args.seeds:
        print(f"\n[seed={seed}]", flush=True)
        t0 = time.time()
        rows = run_one_seed(
            seed,
            n_held_out=args.n_held_out,
            eval_cap=args.eval_cap,
            checkpoint_every=args.checkpoint_every,
            lora_seed_epochs=args.lora_seed_epochs,
        )
        print(f"  seed took {time.time() - t0:.0f}s "
              f"(elapsed {time.time() - grand_t0:.0f}s)", flush=True)
        for r in rows:
            all_rows.append((seed,) + r)

    print(f"\n[done] {time.time() - grand_t0:.0f}s", flush=True)

    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESEARCH_DIR / "ocrr_lora_deberta_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "system", "step", "corrections", "acc_novel", "acc_original"])
        for r in all_rows:
            w.writerow([r[0], r[1], r[2], r[3], f"{r[4]:.6f}", f"{r[5]:.6f}"])
    print(f"[csv] {csv_path}", flush=True)

    # Aggregate per (system, step)
    from collections import defaultdict
    by_sys = defaultdict(list)
    for seed, sysname, step, corr, n_acc, o_acc in all_rows:
        if step == 0 or sysname not in by_sys:
            pass
        by_sys[sysname].append((seed, step, corr, n_acc, o_acc))

    print()
    print("=" * 86)
    print("  Final accuracies (mean ± std over seeds)")
    print("=" * 86)
    final = {}
    for sysname, runs in by_sys.items():
        # final = last step per seed
        per_seed_final = {}
        for seed, step, corr, n_acc, o_acc in runs:
            cur = per_seed_final.get(seed)
            if cur is None or step > cur[0]:
                per_seed_final[seed] = (step, corr, n_acc, o_acc)
        n_arr = np.array([v[2] for v in per_seed_final.values()])
        o_arr = np.array([v[3] for v in per_seed_final.values()])
        final[sysname] = (n_arr.mean(), n_arr.std(), o_arr.mean(), o_arr.std())
        print(f"  {sysname:>14}  novel={n_arr.mean():.4f}±{n_arr.std():.3f}  "
              f"orig={o_arr.mean():.4f}±{o_arr.std():.3f}  n_seeds={len(per_seed_final)}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=120)
        for ax, key_idx, title in [
            (axes[0], 3, "Novel-class accuracy"),
            (axes[1], 4, "Original-distribution accuracy"),
        ]:
            for sysname in ["substrate", "lora_deberta"]:
                runs = [r for r in all_rows if r[1] == sysname]
                # Group by seed; align by checkpoint index
                by_seed = defaultdict(list)
                for r in runs:
                    by_seed[r[0]].append(r)
                for s in by_seed:
                    by_seed[s].sort(key=lambda r: r[2])
                min_len = min(len(v) for v in by_seed.values())
                xs = [by_seed[next(iter(by_seed))][i][3] for i in range(min_len)]
                ys = np.array([
                    [by_seed[s][i][key_idx + 1] for i in range(min_len)]
                    for s in by_seed
                ])
                mean = ys.mean(axis=0)
                std = ys.std(axis=0)
                ax.plot(xs, mean, label=sysname, linewidth=2, marker="o", markersize=3)
                ax.fill_between(xs, mean - std, mean + std, alpha=0.15)
            ax.set_xlabel("corrections applied")
            ax.set_ylabel("accuracy")
            ax.set_title(title)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)
            ax.legend(loc="best", fontsize=9)
        fig.suptitle("OCRR — LoRA-on-DeBERTa-v3-large vs substrate (banking77, oracle)")
        fig.tight_layout()
        plot_path = RESEARCH_DIR / "ocrr_lora_deberta_curves.png"
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"[plot] {plot_path}", flush=True)
    except Exception as e:
        print(f"[plot] skipped: {e}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
