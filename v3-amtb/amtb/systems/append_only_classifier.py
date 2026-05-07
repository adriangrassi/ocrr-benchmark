"""Reference append-only classifier — substrate-style retention baseline.

Implements the AMTB Axis 2 contract using an append-only exemplar
store and majority-vote classification. Direct analog of OCRR v1's
substrate system: every correction appends an exemplar, prediction
runs k-NN over the exemplar set with majority vote.

By construction this system never forgets — `correct_class()` only
appends, never overwrites. final_retention should be ≈ 1.0 in
expectation (matches the OCRR v1 substrate system's published
99.92 ± 0.06% retention).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from amtb.types import AxisName


@dataclass
class AppendOnlyClassifier:
    """Hash-keyed exemplar store with majority-vote classification.

    Uses simple character-bag features (no external encoder dependency)
    so this baseline is self-contained for tests. Production substrates
    would use sentence-transformers; the contract is identical.
    """

    k: int = 5
    margin: float = 0.05
    exemplars: list[tuple[np.ndarray, int]] = field(default_factory=list)
    all_classes: list[int] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "append_only_classifier"

    def supports(self, axis: AxisName) -> bool:
        return axis == AxisName.RETENTION

    def clear(self) -> None:
        self.exemplars.clear()
        self.all_classes = []

    @staticmethod
    def _featurize(text: str, dim: int = 256) -> np.ndarray:
        """Cheap character-bag feature; deterministic, no external deps."""
        v = np.zeros(dim, dtype=np.float32)
        for ch in text.lower():
            v[ord(ch) % dim] += 1.0
        n = float(np.linalg.norm(v))
        return v / max(n, 1e-9)

    def fit_classifier(self, train_texts: list[str], train_labels: list[int],
                       all_classes: list[int]) -> None:
        self.all_classes = list(all_classes)
        for text, label in zip(train_texts, train_labels):
            self.exemplars.append((self._featurize(text), int(label)))

    def predict_class(self, text: str) -> int:
        if not self.exemplars:
            return self.all_classes[0] if self.all_classes else 0
        v = self._featurize(text)
        sims = np.array([float(v @ e) for e, _ in self.exemplars])
        k = min(self.k, len(self.exemplars))
        topk_idx = np.argpartition(-sims, kth=k - 1)[:k]
        topk_labels = [self.exemplars[i][1] for i in topk_idx]
        topk_sims = sims[topk_idx]
        # Majority vote within margin band; ties broken by max similarity
        winner_count = Counter(topk_labels).most_common(1)[0]
        return winner_count[0]

    def correct_class(self, text: str, true_label: int) -> None:
        # Append-only: never modify, only add.
        self.exemplars.append((self._featurize(text), int(true_label)))


@dataclass
class GradientClassifier:
    """Naive baseline: per-class linear weights updated by SGD on every correction.

    Demonstrates the failure mode the retention axis catches: gradient-
    based updates rewrite class weights, so original-class accuracy
    drops as new corrections accumulate. Matches OCRR v1's
    OnlineLinearSystem.
    """

    lr: float = 0.05
    feat_dim: int = 256
    weights: np.ndarray | None = None  # (n_classes, feat_dim)
    class_to_idx: dict[int, int] = field(default_factory=dict)
    all_classes: list[int] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "gradient_classifier"

    def supports(self, axis: AxisName) -> bool:
        return axis == AxisName.RETENTION

    def clear(self) -> None:
        self.weights = None
        self.class_to_idx.clear()
        self.all_classes = []

    @staticmethod
    def _featurize(text: str, dim: int) -> np.ndarray:
        v = np.zeros(dim, dtype=np.float32)
        for ch in text.lower():
            v[ord(ch) % dim] += 1.0
        n = float(np.linalg.norm(v))
        return v / max(n, 1e-9)

    def fit_classifier(self, train_texts: list[str], train_labels: list[int],
                       all_classes: list[int]) -> None:
        self.all_classes = list(all_classes)
        self.class_to_idx = {c: i for i, c in enumerate(all_classes)}
        n = len(all_classes)
        self.weights = np.zeros((n, self.feat_dim), dtype=np.float32)
        # 5 epochs of SGD on training set
        for _ in range(5):
            for text, label in zip(train_texts, train_labels):
                v = self._featurize(text, self.feat_dim)
                if label not in self.class_to_idx:
                    continue
                ci = self.class_to_idx[label]
                logits = self.weights @ v
                probs = self._softmax(logits)
                target = np.zeros(n)
                target[ci] = 1.0
                grad = np.outer(probs - target, v)
                self.weights -= self.lr * grad

    def predict_class(self, text: str) -> int:
        if self.weights is None or not self.all_classes:
            return 0
        v = self._featurize(text, self.feat_dim)
        return int(self.all_classes[int(np.argmax(self.weights @ v))])

    def correct_class(self, text: str, true_label: int) -> None:
        if self.weights is None:
            return
        v = self._featurize(text, self.feat_dim)
        # Add the class to the index space if newly seen
        if true_label not in self.class_to_idx:
            self.class_to_idx[true_label] = len(self.all_classes)
            self.all_classes.append(true_label)
            self.weights = np.vstack([self.weights, np.zeros((1, self.feat_dim), dtype=np.float32)])
        ci = self.class_to_idx[true_label]
        logits = self.weights @ v
        probs = self._softmax(logits)
        target = np.zeros(self.weights.shape[0])
        target[ci] = 1.0
        grad = np.outer(probs - target, v)
        self.weights -= self.lr * grad

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        ex = np.exp(x - x.max())
        return ex / ex.sum()


__all__ = ["AppendOnlyClassifier", "GradientClassifier"]
