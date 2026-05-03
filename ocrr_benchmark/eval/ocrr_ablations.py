"""Substrate vote-rule ablations for OCRR.

Tests whether the substrate's full vote rule (margin-band majority count +
max-similarity tiebreak + recency tiebreak) is overdetermined. Each
variant ablates one design choice:

  SubstrateK1System          k=1 only — no voting at all, nearest-neighbour
                             label wins.
  SubstrateSumSimSystem      sum-of-similarities vote (no margin band,
                             no recency). Re-introduces the bug we fixed
                             in the demo where 4 mediocre matches outvote
                             1 strong match.
  SubstrateCountOnlySystem   margin-band count + insertion-order tiebreak
                             (no max-sim, no recency). Tests whether
                             max-sim tiebreak is load-bearing.

The reference system is the unmodified `SubstrateSystem`:
  margin-band count → max_sim → latest_id

If the full vote rule lifts ≥1 pp on novel over the simplest variant
(k=1) or the count-only variant, then it's load-bearing.
"""

from __future__ import annotations

import numpy as np

from ocrr_benchmark.eval.ocrr import OCRRSystem
from ocrr_benchmark.memory.episodic import ImmutableLedger


# ============================================================================
# k=1 only
# ============================================================================

class SubstrateK1System(OCRRSystem):
    name = "substrate_k1"

    def __init__(self, seed_vecs, seed_labels):
        self._seed_vecs = seed_vecs
        self._seed_labels = seed_labels
        self.reset()

    def reset(self):
        self.ledger = ImmutableLedger()
        for v, lbl in zip(self._seed_vecs, self._seed_labels):
            self.ledger.write(v.astype(np.float32), text="", tags=(lbl,))

    def predict(self, vec):
        hits = self.ledger.nearest(vec.astype(np.float32), k=1)
        if not hits:
            return None
        entry, _ = hits[0]
        return entry.tags[0] if entry.tags else None

    def correct(self, vec, true_label):
        self.ledger.write(vec.astype(np.float32), text="", tags=(true_label,))


# ============================================================================
# sum-of-similarities (the original bug we fixed)
# ============================================================================

class SubstrateSumSimSystem(OCRRSystem):
    name = "substrate_sumsim"

    def __init__(self, seed_vecs, seed_labels, *, k=5):
        self._seed_vecs = seed_vecs
        self._seed_labels = seed_labels
        self._k = k
        self.reset()

    def reset(self):
        self.ledger = ImmutableLedger()
        for v, lbl in zip(self._seed_vecs, self._seed_labels):
            self.ledger.write(v.astype(np.float32), text="", tags=(lbl,))

    def predict(self, vec):
        hits = self.ledger.nearest(vec.astype(np.float32), k=self._k)
        if not hits:
            return None
        # Pure sum-of-similarities; no band, no recency. The variant we had
        # in the demo before fixing it — 4 mediocre 0.82 hits outvote 1
        # fresh 0.98 correction.
        votes: dict[str, float] = {}
        for entry, score in hits:
            if not entry.tags:
                continue
            label = entry.tags[0]
            votes[label] = votes.get(label, 0.0) + float(score)
        if not votes:
            return None
        return max(votes.items(), key=lambda kv: kv[1])[0]

    def correct(self, vec, true_label):
        self.ledger.write(vec.astype(np.float32), text="", tags=(true_label,))


# ============================================================================
# margin-band count + insertion-order tiebreak (no max-sim, no recency)
# ============================================================================

class SubstrateCountOnlySystem(OCRRSystem):
    name = "substrate_count_only"

    def __init__(self, seed_vecs, seed_labels, *, k=5, margin=0.05):
        self._seed_vecs = seed_vecs
        self._seed_labels = seed_labels
        self._k = k
        self._margin = margin
        self.reset()

    def reset(self):
        self.ledger = ImmutableLedger()
        for v, lbl in zip(self._seed_vecs, self._seed_labels):
            self.ledger.write(v.astype(np.float32), text="", tags=(lbl,))

    def predict(self, vec):
        hits = self.ledger.nearest(vec.astype(np.float32), k=self._k)
        if not hits:
            return None
        top_sim = max(float(s) for _, s in hits)
        band = [(e, float(s)) for e, s in hits if float(s) >= top_sim - self._margin]
        voters = band if band else hits
        counts: dict[str, int] = {}
        # No max_sim, no recency tracking — pure count, ties broken by Python's
        # max() which uses insertion order (effectively "first label seen wins").
        for entry, _score in voters:
            if not entry.tags:
                continue
            label = entry.tags[0]
            counts[label] = counts.get(label, 0) + 1
        if not counts:
            return None
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def correct(self, vec, true_label):
        self.ledger.write(vec.astype(np.float32), text="", tags=(true_label,))


# ============================================================================
# count + max-sim tiebreak (no recency) — substrate without the recency bit
# ============================================================================

class SubstrateNoRecencySystem(OCRRSystem):
    name = "substrate_no_recency"

    def __init__(self, seed_vecs, seed_labels, *, k=5, margin=0.05):
        self._seed_vecs = seed_vecs
        self._seed_labels = seed_labels
        self._k = k
        self._margin = margin
        self.reset()

    def reset(self):
        self.ledger = ImmutableLedger()
        for v, lbl in zip(self._seed_vecs, self._seed_labels):
            self.ledger.write(v.astype(np.float32), text="", tags=(lbl,))

    def predict(self, vec):
        hits = self.ledger.nearest(vec.astype(np.float32), k=self._k)
        if not hits:
            return None
        top_sim = max(float(s) for _, s in hits)
        band = [(e, float(s)) for e, s in hits if float(s) >= top_sim - self._margin]
        voters = band if band else hits
        counts: dict[str, int] = {}
        max_sim: dict[str, float] = {}
        for entry, score in voters:
            if not entry.tags:
                continue
            label = entry.tags[0]
            counts[label] = counts.get(label, 0) + 1
            max_sim[label] = max(max_sim.get(label, -1.0), float(score))
        if not counts:
            return None
        return max(counts.keys(), key=lambda lbl: (counts[lbl], max_sim[lbl]))

    def correct(self, vec, true_label):
        self.ledger.write(vec.astype(np.float32), text="", tags=(true_label,))
