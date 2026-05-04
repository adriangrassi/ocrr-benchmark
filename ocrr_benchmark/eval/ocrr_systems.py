"""OCRR system implementations — substrate, static k-NN, static linear,
online linear (fine-tune-on-correction).

All operate on pre-computed embeddings (encoder-agnostic). The harness
in ``ocrr.py`` calls ``predict(vec)`` / ``correct(vec, label)``.

The four systems were chosen to make the recovery-rate comparison sharp:

    substrate           — what we built. Online-correctable by design.
    static k-NN         — same encoder, same retrieval, but never updates.
                          Floor for "non-learning system."
    static linear       — linear classifier head trained once on the 67-class
                          subset. Cannot emit the 10 held-out labels by
                          construction (output layer is 67-dim).
    online linear       — same linear head architecture but with 77 outputs
                          (10 zero-init), updated by per-correction SGD.
                          The honest "fine-tune-on-correction" baseline that
                          sceptics will demand.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ocrr_benchmark.memory.episodic import ImmutableLedger
from ocrr_benchmark.eval.ocrr import OCRRSystem


# ============================================================================
# Substrate: bge-large + ImmutableLedger + margin-band majority + max_sim
# tiebreak. Same vote rule as the live demo.
# ============================================================================

class SubstrateSystem(OCRRSystem):
    name = "substrate"

    def __init__(
        self,
        seed_vecs: np.ndarray,
        seed_labels: list[str],
        *,
        k: int = 5,
        margin: float = 0.05,
        force_brute: bool = False,
    ) -> None:
        """Substrate classifier over an append-only ledger.

        ``force_brute=True`` makes every retrieval use brute-force cosine
        similarity over the full ledger instead of HNSW. This guarantees
        100% recall (no "forgetting via approximate retrieval") at the
        cost of O(N) latency per query. Recommended for compliance-bound
        deployments and for the never-forget guarantee at scale; the
        scaling-study script uses this mode to establish a recall ceiling
        the HNSW path is benchmarked against.
        """
        self._seed_vecs = seed_vecs
        self._seed_labels = seed_labels
        self._k = k
        self._margin = margin
        self._force_brute = force_brute
        self.reset()

    def reset(self) -> None:
        self.ledger = ImmutableLedger()
        for v, lbl in zip(self._seed_vecs, self._seed_labels):
            self.ledger.write(v.astype(np.float32), text="", tags=(lbl,))

    def predict(self, vec: np.ndarray) -> str | None:
        v = vec.astype(np.float32)
        hits = self.ledger.nearest(v, k=self._k, force_brute=self._force_brute)
        if not hits:
            return None
        top_sim = max(float(s) for _, s in hits)
        band = [(e, float(s)) for e, s in hits if float(s) >= top_sim - self._margin]
        voters = band if band else hits
        counts: dict[str, int] = {}
        max_sim: dict[str, float] = {}
        latest_id: dict[str, int] = {}
        for entry, score in voters:
            if not entry.tags:
                continue
            label = entry.tags[0]
            counts[label] = counts.get(label, 0) + 1
            max_sim[label] = max(max_sim.get(label, -1.0), float(score))
            latest_id[label] = max(latest_id.get(label, -1), entry.id)
        if not counts:
            return None
        return max(
            counts.keys(),
            key=lambda lbl: (counts[lbl], max_sim[lbl], latest_id[lbl]),
        )

    def correct(self, vec: np.ndarray, true_label: str) -> None:
        self.ledger.write(vec.astype(np.float32), text="", tags=(true_label,))


# ============================================================================
# Static k-NN: same encoder, same retrieval, never updates.
# ============================================================================

class StaticKNNSystem(OCRRSystem):
    name = "static_knn"

    def __init__(
        self,
        seed_vecs: np.ndarray,
        seed_labels: list[str],
        *,
        k: int = 5,
    ) -> None:
        self._seed_vecs = seed_vecs
        self._seed_labels = seed_labels
        self._k = k
        self.reset()

    def reset(self) -> None:
        # Frozen index — exactly the seed corpus, never grows.
        self._idx_vecs = self._seed_vecs.astype(np.float32)
        # Pre-normalize for cosine via dot.
        norms = np.linalg.norm(self._idx_vecs, axis=1, keepdims=True)
        self._idx_unit = self._idx_vecs / np.clip(norms, 1e-9, None)
        self._idx_labels = list(self._seed_labels)

    def predict(self, vec: np.ndarray) -> str | None:
        v = vec.astype(np.float32)
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            return None
        v_unit = v / n
        sims = self._idx_unit @ v_unit
        k = min(self._k, len(self._idx_unit))
        top = np.argpartition(-sims, k - 1)[:k]
        # Margin-band majority + max_sim tiebreak (same as substrate)
        top_sim = float(sims[top].max())
        band_mask = sims[top] >= top_sim - 0.05
        band = top[band_mask]
        counts: dict[str, int] = {}
        max_sim: dict[str, float] = {}
        for j in band:
            lbl = self._idx_labels[int(j)]
            counts[lbl] = counts.get(lbl, 0) + 1
            max_sim[lbl] = max(max_sim.get(lbl, -1.0), float(sims[int(j)]))
        if not counts:
            return None
        return max(counts.keys(), key=lambda lbl: (counts[lbl], max_sim[lbl]))

    def correct(self, vec: np.ndarray, true_label: str) -> None:
        # No-op by design. This is the "no online learning" floor.
        return


# ============================================================================
# Static linear head — frozen 67-class softmax classifier over bge-large
# embeddings. Cannot emit the 10 held-out labels (its output dim is 67).
# ============================================================================

def _train_linear_head(
    vecs: np.ndarray,
    labels: list[str],
    classes: list[str],
    *,
    epochs: int = 30,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[nn.Linear, dict[str, int]]:
    """Train a logistic-regression head on (vec -> class). Returns the
    fitted Linear layer and the label->index map used during training."""
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.asarray([label_to_idx[lbl] for lbl in labels], dtype=np.int64)
    X = torch.from_numpy(vecs.astype(np.float32))
    yt = torch.from_numpy(y).long()
    torch.manual_seed(seed)
    head = nn.Linear(X.shape[1], len(classes), bias=True).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    n = X.shape[0]
    g = torch.Generator().manual_seed(seed)
    batch_size = 256
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for s in range(0, n, batch_size):
            idx = perm[s: s + batch_size]
            opt.zero_grad()
            logits = head(X[idx])
            loss = F.cross_entropy(logits, yt[idx])
            loss.backward()
            opt.step()
    head.eval()
    return head, label_to_idx


class StaticLinearSystem(OCRRSystem):
    name = "static_linear"

    def __init__(
        self,
        seed_vecs: np.ndarray,
        seed_labels: list[str],
        known_classes: list[str],
        *,
        seed: int = 0,
    ) -> None:
        # Train once at construction time (frozen forever).
        head, lbl_to_idx = _train_linear_head(
            seed_vecs, seed_labels, known_classes, seed=seed,
        )
        self._head = head
        self._idx_to_lbl = {i: c for c, i in lbl_to_idx.items()}

    def reset(self) -> None:
        # Frozen head — reset is a no-op (it was already fitted at __init__).
        return

    def predict(self, vec: np.ndarray) -> str | None:
        v = torch.from_numpy(vec.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            logits = self._head(v)
        idx = int(logits.argmax(dim=-1).item())
        return self._idx_to_lbl.get(idx)

    def correct(self, vec: np.ndarray, true_label: str) -> None:
        return  # no-op


# ============================================================================
# Online linear head — 77-output classifier (10 zero-init for held-out
# classes). Per-correction SGD on the (vec, label) pair.
# ============================================================================

class OnlineLinearSystem(OCRRSystem):
    name = "online_linear"

    def __init__(
        self,
        seed_vecs: np.ndarray,
        seed_labels: list[str],
        all_classes: list[str],
        *,
        init_seed: int = 0,
        sgd_lr: float = 0.05,
        seed_epochs: int = 30,
    ) -> None:
        self._seed_vecs = seed_vecs
        self._seed_labels = seed_labels
        self._all_classes = list(all_classes)
        self._lbl_to_idx = {c: i for i, c in enumerate(all_classes)}
        self._idx_to_lbl = {i: c for i, c in enumerate(all_classes)}
        self._init_seed = init_seed
        self._sgd_lr = sgd_lr
        self._seed_epochs = seed_epochs
        self.reset()

    def reset(self) -> None:
        # Build a 77-output head; train only on the seed (67-class) data.
        # The 10 held-out outputs are randomly initialized but never see
        # gradient until correct() is called.
        torch.manual_seed(self._init_seed)
        dim = self._seed_vecs.shape[1]
        head = nn.Linear(dim, len(self._all_classes), bias=True)
        # Train head end-to-end on seed data. Outputs corresponding to
        # held-out classes get ~zero gradient (no examples), so they
        # remain near random init.
        X = torch.from_numpy(self._seed_vecs.astype(np.float32))
        y = torch.tensor(
            [self._lbl_to_idx[lbl] for lbl in self._seed_labels], dtype=torch.long
        )
        opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-4)
        n = X.shape[0]
        g = torch.Generator().manual_seed(self._init_seed)
        bs = 256
        head.train()
        for _ in range(self._seed_epochs):
            perm = torch.randperm(n, generator=g)
            for s in range(0, n, bs):
                idx = perm[s: s + bs]
                opt.zero_grad()
                logits = head(X[idx])
                loss = F.cross_entropy(logits, y[idx])
                loss.backward()
                opt.step()
        head.eval()
        self._head = head
        # Hold the per-correction SGD optimizer separately (plain SGD,
        # no momentum, single-example updates).
        self._sgd = torch.optim.SGD(self._head.parameters(), lr=self._sgd_lr)

    def predict(self, vec: np.ndarray) -> str | None:
        v = torch.from_numpy(vec.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            logits = self._head(v)
        idx = int(logits.argmax(dim=-1).item())
        return self._idx_to_lbl.get(idx)

    def correct(self, vec: np.ndarray, true_label: str) -> None:
        if true_label not in self._lbl_to_idx:
            return  # unknown class — out of label space
        target = torch.tensor([self._lbl_to_idx[true_label]], dtype=torch.long)
        v = torch.from_numpy(vec.astype(np.float32)).unsqueeze(0)
        self._head.train()
        self._sgd.zero_grad()
        logits = self._head(v)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        self._sgd.step()
        self._head.eval()
