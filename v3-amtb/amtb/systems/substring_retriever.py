"""Reference scale-axis retriever — substring-match baseline.

Minimal text retriever that can answer Axis 5 (scale) and a subset of
Axis 1 (recall@k) at zero compute cost. Indexes entries by their text
words and returns top-k by overlap count. Suitable as a non-trivial
baseline that has nothing to do with vector embeddings.

A better baseline would use bge-small or another sentence encoder; this
class is mainly for tests and as a sanity floor.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from amtb.types import AxisName


@dataclass
class SubstringRetriever:
    """Inverted-word-index retriever. Top-k by token overlap with query."""

    storage: dict[str, str] = field(default_factory=dict)
    inverted: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    @property
    def name(self) -> str:
        return "substring_retriever"

    def supports(self, axis: AxisName) -> bool:
        # Append-only by construction (never deletes from storage), so
        # it also qualifies for adversarial-revision retrieval evaluation.
        return axis in {
            AxisName.SCALE, AxisName.RECALL, AxisName.ADVERSARIAL_REVISION,
        }

    def clear(self) -> None:
        self.storage.clear()
        self.inverted.clear()

    def ingest(self, entry_id: str, text: str) -> None:
        self.storage[entry_id] = text
        for tok in self._tokens(text):
            self.inverted[tok].add(entry_id)

    def query_topk(self, query: str, k: int = 10) -> list[str]:
        toks = self._tokens(query)
        if not toks:
            return []
        # Score each candidate by how many query tokens it matches.
        scores: dict[str, int] = defaultdict(int)
        for t in toks:
            for eid in self.inverted.get(t, ()):
                scores[eid] += 1
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [eid for eid, _ in ranked[:k]]

    @staticmethod
    def _tokens(text: str) -> Iterable[str]:
        # Lowercase + simple alphanumeric split.
        out = []
        cur = []
        for ch in text.lower():
            if ch.isalnum():
                cur.append(ch)
            elif cur:
                out.append("".join(cur))
                cur = []
        if cur:
            out.append("".join(cur))
        return out


__all__ = ["SubstringRetriever"]
