"""HNSW backend for the immutable ledger.

Adapted from ``research/snapshots/cortex-2026-04-27/memory/hnsw_index.py``.

Differences from the upstream snapshot:
    * Numpy throughout instead of torch (Horizon's memory layer is np-native;
      torch is reserved for ``growth/`` where actual learning happens).
    * Slightly tighter type signatures.
    * Same algorithmic behavior: cosine space, lazy init, graceful no-op when
      hnswlib is unavailable.

The ledger remains source of truth — this index is a rebuildable cache.
At 10k+ entries HNSW gives 10–100× speedup with ~99% recall@10. Below
that threshold the brute-force matmul on the cached matrix is faster.
"""

from __future__ import annotations

import numpy as np


class HNSWCosineIndex:
    """Cosine-similarity HNSW index wrapping ``hnswlib``.

    Graceful no-op when ``hnswlib`` is not installed — ``is_available``
    returns False and ``add`` / ``knn`` do nothing / return empties. The
    caller is expected to fall back to brute force in that case.
    """

    def __init__(
        self,
        dim: int,
        max_elements: int = 1_000_000,
        ef_construction: int = 200,
        M: int = 16,
        ef_search: int = 64,
    ) -> None:
        self.dim = dim
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.M = M
        self.ef_search = ef_search
        self._index = None
        self._n_items = 0
        self._init_if_possible()

    def _init_if_possible(self) -> None:
        try:
            import hnswlib  # type: ignore[import-not-found]
        except ImportError:
            self._index = None
            return
        self._index = hnswlib.Index(space="cosine", dim=self.dim)
        self._index.init_index(
            max_elements=self.max_elements,
            ef_construction=self.ef_construction,
            M=self.M,
        )
        self._index.set_ef(self.ef_search)

    @property
    def is_available(self) -> bool:
        return self._index is not None

    def __len__(self) -> int:
        return self._n_items

    def add(self, vectors: np.ndarray, ids: list[int] | None = None) -> None:
        if self._index is None or vectors.shape[0] == 0:
            return
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        data = vectors.astype(np.float32, copy=False)
        if ids is None:
            ids = list(range(self._n_items, self._n_items + data.shape[0]))
        self._index.add_items(data, ids)
        self._n_items += data.shape[0]

    def knn(self, query: np.ndarray, k: int = 1) -> tuple[list[int], list[float]]:
        if self._index is None or self._n_items == 0:
            return [], []
        q = query.astype(np.float32, copy=False)
        if q.ndim == 1:
            q = q[None, :]
        k_eff = min(k, self._n_items)
        labels, distances = self._index.knn_query(q, k=k_eff)
        # cosine distance = 1 - cosine similarity
        sims = (1.0 - distances[0]).tolist()
        return labels[0].tolist(), sims

    def knn_batch(
        self, queries: np.ndarray, k: int = 1
    ) -> tuple[list[list[int]], list[list[float]]]:
        if self._index is None or self._n_items == 0:
            batch_size = queries.shape[0] if queries.ndim > 1 else 1
            return [[] for _ in range(batch_size)], [[] for _ in range(batch_size)]
        q = queries.astype(np.float32, copy=False)
        if q.ndim == 1:
            q = q[None, :]
        k_eff = min(k, self._n_items)
        labels, distances = self._index.knn_query(q, k=k_eff)
        sims = (1.0 - distances).tolist()
        return labels.tolist(), sims
