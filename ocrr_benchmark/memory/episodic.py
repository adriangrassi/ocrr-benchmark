"""Immutable append-only episodic memory.

Adapted from Cortex's ``ImmutableMemoryLedger`` — see
``research/snapshots/cortex-2026-04-27/memory/ledger.py`` for the upstream
version with the full v3 retrieval surface (hybrid BM25+vector, conflict
detection, memory graph, time-travel queries).

Invariants:
    * No delete, no update, no truncate. Once written, an entry exists forever.
    * Each entry carries a SHA-256 hash of its content. "Corrections" become
      new entries that link to the originals via ``contradicts``.
    * Retention policy is implemented as indexing, not deletion.

Phase 1.1: brute-force cosine + auto-switch to HNSW at ``HNSW_AUTO_THRESHOLD``
entries. Subsequent phases pull in consolidation, slow-cortex replay, and the
v3 retrieval extensions.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ocrr_benchmark.memory._hnsw import HNSWCosineIndex


_GENESIS_PREV_HASH = "0" * 64


@dataclass
class MemoryEntry:
    """A single committed memory episode.

    The ``content_hash`` covers all entry payload fields *plus* ``prev_hash``,
    forming a Merkle-style chain across the ledger: entry N's hash depends on
    entry N-1's hash. Tampering with, deleting, or reordering any past entry
    breaks every subsequent entry's hash and is caught by
    ``ImmutableLedger.verify_integrity()``.
    """

    id: int
    t_created: float
    embedding: np.ndarray
    text: str = ""
    tags: tuple[str, ...] = ()
    contradicts: tuple[int, ...] = ()
    prev_hash: str = _GENESIS_PREV_HASH
    content_hash: str = ""

    def compute_hash(self) -> str:
        payload = {
            "id": self.id,
            "t_created": self.t_created,
            "embedding": self.embedding.tolist(),
            "text": self.text,
            "tags": list(self.tags),
            "contradicts": list(self.contradicts),
            "prev_hash": self.prev_hash,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    # Compat shim: cortex's MemoryEntry exposes the embedding as `.ec_code`
    # (entorhinal-cortex code, biology-naming). Lifted cortex modules
    # (slow_cortex, semantic_store, consolidation) read `.ec_code`. We expose
    # the same surface here so they work unchanged.
    @property
    def ec_code(self) -> list[float]:
        return self.embedding.tolist()


class ImmutableLedger:
    """Append-only ledger with auto-switching brute-force / HNSW retrieval.

    Below ``HNSW_AUTO_THRESHOLD`` entries: cached numpy matmul (fastest at
    small scale, zero extra dependencies). At and above the threshold: lazy-
    initialized HNSW for sub-millisecond retrieval at 100k+ entries.

    The HNSW index is a rebuildable cache — if it is unavailable (hnswlib
    not installed) or goes out of sync, the ledger gracefully falls back to
    brute force.
    """

    HNSW_AUTO_THRESHOLD = 10_000
    INITIAL_BUFFER_CAPACITY = 256

    def __init__(
        self,
        *,
        hnsw_threshold: int | None = None,
        hnsw_max_elements: int = 1_000_000,
        initial_capacity: int | None = None,
    ) -> None:
        self._entries: list[MemoryEntry] = []
        # Doubling-buffer arena for the brute-force cosine matrix. Append is
        # amortized O(1) instead of the O(N) reallocate-and-copy that
        # `np.stack` / `np.concatenate` would incur on every write.
        # `_buffer[: len(self._entries)]` is always the live data.
        self._buffer: np.ndarray | None = None
        self._unit_buffer: np.ndarray | None = None  # incremental L2-normalized cache
        self._capacity = (
            initial_capacity if initial_capacity is not None
            else self.INITIAL_BUFFER_CAPACITY
        )
        self._hnsw: HNSWCosineIndex | None = None
        self._hnsw_threshold = (
            hnsw_threshold if hnsw_threshold is not None else self.HNSW_AUTO_THRESHOLD
        )
        self._hnsw_max_elements = hnsw_max_elements
        self._dim: int | None = None

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def backend(self) -> str:
        """Which retrieval backend the next ``nearest()`` call will use."""
        if self._hnsw is not None and self._hnsw.is_available and len(self._hnsw) > 0:
            return "hnsw"
        return "brute"

    def write(
        self,
        embedding: np.ndarray,
        *,
        text: str = "",
        tags: Sequence[str] = (),
        contradicts: Sequence[int] = (),
    ) -> MemoryEntry:
        if embedding.ndim != 1:
            raise ValueError(f"embedding must be 1-D, got shape {embedding.shape}")
        if self._dim is None:
            self._dim = int(embedding.shape[0])
        elif embedding.shape[0] != self._dim:
            raise ValueError(
                f"embedding dim mismatch: ledger is {self._dim}-D, got {embedding.shape[0]}-D"
            )

        emb = embedding.astype(np.float32, copy=True)
        prev_hash = (
            self._entries[-1].content_hash if self._entries else _GENESIS_PREV_HASH
        )
        entry = MemoryEntry(
            id=len(self._entries),
            t_created=time.time(),
            embedding=emb,
            text=text,
            tags=tuple(tags),
            contradicts=tuple(contradicts),
            prev_hash=prev_hash,
        )
        entry.content_hash = entry.compute_hash()
        self._entries.append(entry)

        # Append to the doubling-buffer arena. O(1) amortized: copy a single
        # row most of the time, with a single (capacity, D) reallocation
        # roughly every log2(N) writes. No full-cache rebuild on read.
        self._append_to_buffer(emb)

        # Maybe promote to HNSW
        if self._hnsw is None and len(self._entries) >= self._hnsw_threshold:
            self._initialize_hnsw()
        elif self._hnsw is not None:
            self._hnsw.add(emb[None, :], ids=[entry.id])

        return entry

    def get(self, entry_id: int) -> MemoryEntry:
        return self._entries[entry_id]

    def all_entries(self) -> tuple[MemoryEntry, ...]:
        return tuple(self._entries)

    # Compat shim for cortex code that calls ledger.all()
    def all(self) -> list[MemoryEntry]:
        return list(self._entries)

    def nearest(
        self, query: np.ndarray, k: int = 5, *, force_brute: bool = False,
    ) -> list[tuple[MemoryEntry, float]]:
        """Return up to k entries by cosine similarity, descending.

        ``force_brute=True`` bypasses the HNSW backend even when available
        and forces a brute-force scan over every entry. Useful when the
        caller needs a 100%-recall guarantee at the cost of latency, e.g.
        for the substrate's "never forget" claim at scale where HNSW
        approximate-recall could otherwise miss a rare-class entry.
        """
        if not self._entries:
            return []

        use_hnsw = (
            not force_brute
            and self._hnsw is not None
            and self._hnsw.is_available
            and len(self._hnsw) > 0
        )
        if use_hnsw:
            ids, sims = self._hnsw.knn(query.astype(np.float32, copy=False), k=k)
            return [(self._entries[int(i)], float(s)) for i, s in zip(ids, sims)]

        # Brute-force path: live slice of the unit-normalized buffer.
        n = len(self._entries)
        m_unit = self._unit_buffer[:n] if self._unit_buffer is not None else None
        if m_unit is None:
            return []
        q = query.astype(np.float32, copy=False)
        q_norm = q / max(float(np.linalg.norm(q)), 1e-12)
        scores = m_unit @ q_norm
        top_idx = np.argsort(-scores)[: min(k, n)]
        return [(self._entries[int(i)], float(scores[int(i)])) for i in top_idx]

    def verify_hnsw_recall(
        self, queries: np.ndarray, k: int = 5,
    ) -> dict[str, float]:
        """Compare HNSW top-k against ground-truth brute-force top-k.

        Runs both backends on the same queries and reports recall@k:
        the fraction of brute-force-correct neighbors that HNSW also
        returned. recall@k = 1.0 means HNSW found every true top-k
        neighbor on every query in the sample. recall@k < 1.0 means
        HNSW missed entries the ledger physically stores, which is the
        actual mechanism by which the substrate could "forget" at scale.

        Returns a dict with ``recall_at_k``, ``num_queries``, ``k``,
        and ``backend`` ("hnsw" or "brute_only" if HNSW unavailable).
        """
        if not self._entries:
            return {"recall_at_k": 1.0, "num_queries": 0, "k": k, "backend": "empty"}
        if self._hnsw is None or not self._hnsw.is_available or len(self._hnsw) == 0:
            return {
                "recall_at_k": 1.0,
                "num_queries": int(queries.shape[0] if queries.ndim > 1 else 1),
                "k": k,
                "backend": "brute_only",
            }
        if queries.ndim == 1:
            queries = queries[None, :]

        total_recovered = 0
        total_target = 0
        for q in queries:
            hnsw_results = self.nearest(q, k=k, force_brute=False)
            brute_results = self.nearest(q, k=k, force_brute=True)
            hnsw_ids = {e.id for e, _ in hnsw_results}
            brute_ids = {e.id for e, _ in brute_results}
            total_recovered += len(hnsw_ids & brute_ids)
            total_target += len(brute_ids)
        recall = (total_recovered / total_target) if total_target > 0 else 1.0
        return {
            "recall_at_k": float(recall),
            "num_queries": int(queries.shape[0]),
            "k": k,
            "backend": "hnsw",
        }

    def nearest_batch(
        self, queries: np.ndarray, k: int = 5
    ) -> list[list[tuple[MemoryEntry, float]]]:
        """Batched nearest — one matmul (HNSW or brute) for many queries.

        Returns a list of B per-query result lists, each up to length k.
        """
        if not self._entries:
            batch_size = queries.shape[0] if queries.ndim > 1 else 1
            return [[] for _ in range(batch_size)]
        if queries.ndim == 1:
            queries = queries[None, :]

        if self._hnsw is not None and self._hnsw.is_available and len(self._hnsw) > 0:
            id_lists, sim_lists = self._hnsw.knn_batch(
                queries.astype(np.float32, copy=False), k=k
            )
            return [
                [(self._entries[int(i)], float(s)) for i, s in zip(ids, sims)]
                for ids, sims in zip(id_lists, sim_lists)
            ]

        # Brute-force batched on the live slice
        n = len(self._entries)
        m_unit = self._unit_buffer[:n] if self._unit_buffer is not None else None
        if m_unit is None:
            return [[] for _ in range(queries.shape[0])]
        q = queries.astype(np.float32, copy=False)
        q_norms = np.linalg.norm(q, axis=1, keepdims=True)
        q_unit = q / np.maximum(q_norms, 1e-12)
        scores = q_unit @ m_unit.T  # (B, N)
        top_idx = np.argsort(-scores, axis=1)[:, : min(k, n)]
        out: list[list[tuple[MemoryEntry, float]]] = []
        for i, idxs in enumerate(top_idx):
            out.append(
                [(self._entries[int(j)], float(scores[i, int(j)])) for j in idxs]
            )
        return out

    def verify_integrity(self) -> bool:
        """Verify the hash chain end-to-end.

        Returns True iff every entry satisfies BOTH:
            (a) ``content_hash == compute_hash()`` — entry payload hasn't been
                modified after commit;
            (b) ``prev_hash`` equals the previous entry's ``content_hash``
                (or the genesis sentinel for entry 0) — no entries have been
                inserted, deleted, or reordered.

        Any single past-entry mutation breaks (a) for that entry; any
        deletion/reorder breaks (b) at the gap. Either failure mode trips
        this check.
        """
        expected_prev = _GENESIS_PREV_HASH
        for entry in self._entries:
            if entry.prev_hash != expected_prev:
                return False
            if entry.content_hash != entry.compute_hash():
                return False
            expected_prev = entry.content_hash
        return True

    def integrity_break(self) -> tuple[int, str] | None:
        """Locate the first chain break, if any.

        Returns ``(index, reason)`` of the first broken entry, or ``None``
        if the chain is intact. Useful for diagnostics. Reason is one of
        ``"prev_hash_mismatch"``, ``"content_hash_mismatch"``.
        """
        expected_prev = _GENESIS_PREV_HASH
        for i, entry in enumerate(self._entries):
            if entry.prev_hash != expected_prev:
                return (i, "prev_hash_mismatch")
            if entry.content_hash != entry.compute_hash():
                return (i, "content_hash_mismatch")
            expected_prev = entry.content_hash
        return None

    def _initialize_hnsw(self) -> None:
        """Lazy-init HNSW and bulk-load all existing entries."""
        if self._dim is None or self._buffer is None:
            return
        index = HNSWCosineIndex(dim=self._dim, max_elements=self._hnsw_max_elements)
        if not index.is_available:
            # hnswlib not installed; stay on brute force
            return
        n = len(self._entries)
        vectors = self._buffer[:n]  # live slice, zero-copy
        ids = [e.id for e in self._entries]
        index.add(vectors, ids=ids)
        self._hnsw = index

    def _append_to_buffer(self, emb: np.ndarray) -> None:
        """Append one embedding to the doubling-buffer arena.

        Lazily allocates the buffer on first write (when ``self._dim`` is
        known). When the buffer fills up, allocates a new one with double
        the capacity and copies the existing rows once. The unit-normalized
        cache mirrors the same growth, normalized incrementally so
        retrieval never recomputes norms over the whole cache.
        """
        assert self._dim is not None
        if self._buffer is None:
            self._buffer = np.empty((self._capacity, self._dim), dtype=np.float32)
            self._unit_buffer = np.empty((self._capacity, self._dim), dtype=np.float32)

        n = len(self._entries) - 1  # row to write into (entry was just appended)
        if n >= self._capacity:
            new_capacity = self._capacity * 2
            new_buffer = np.empty((new_capacity, self._dim), dtype=np.float32)
            new_unit = np.empty((new_capacity, self._dim), dtype=np.float32)
            new_buffer[: self._capacity] = self._buffer
            new_unit[: self._capacity] = self._unit_buffer  # type: ignore[index]
            self._buffer = new_buffer
            self._unit_buffer = new_unit
            self._capacity = new_capacity

        self._buffer[n] = emb
        norm = float(np.linalg.norm(emb))
        if norm > 1e-12:
            self._unit_buffer[n] = emb / norm  # type: ignore[index]
        else:
            self._unit_buffer[n] = 0.0  # type: ignore[index]
