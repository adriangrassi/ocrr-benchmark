"""Reference hash-chained ledger.

Minimal append-only memory system with cryptographic provenance per
entry. Each entry includes the hash of the previous entry, forming a
chain that any tampering breaks. Verification re-hashes the entire
chain.

This is a reference implementation for AMTB Axis 3 (auditability).
Production memory systems with hash-chained storage (e.g. Horizon's
Cortex `ImmutableMemoryLedger`) inherit from the same pattern but
add retrieval/encoding layers. For audit-axis purposes only the
append-and-verify contract matters.

Per the pre-registration: Axis 3 deliberately rewards systems that
treat memory as cryptographically verifiable. Systems that don't
(SQL-backed, naive flat-dict) score 0.0. We make that architectural
gap visible.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Iterable


@dataclass(frozen=True)
class ChainedEntry:
    """One entry in a hash-chained ledger."""
    entry_id: str
    text: str
    prev_hash: str
    entry_hash: str

    @staticmethod
    def compute_hash(entry_id: str, text: str, prev_hash: str) -> str:
        h = hashlib.sha256()
        h.update(entry_id.encode("utf-8")); h.update(b"\x00")
        h.update(text.encode("utf-8")); h.update(b"\x00")
        h.update(prev_hash.encode("utf-8"))
        return h.hexdigest()


GENESIS_HASH = "0" * 64


@dataclass
class HashChainedLedger:
    """Append-only ledger with sha256 chain of entries.

    Tampering with any entry's text, id, or prev_hash invalidates the
    chain at and after that entry. `verify_chain()` returns False if
    any entry's stored hash doesn't match what its content would produce.
    """

    entries: list[ChainedEntry] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "hash_chained_ledger"

    def supports(self, axis) -> bool:
        # Reference impl claims auditability only.
        try:
            return axis.value == "auditability"
        except AttributeError:
            return str(axis) == "auditability"

    def ingest(self, entry_id: str, text: str) -> None:
        prev_hash = self.entries[-1].entry_hash if self.entries else GENESIS_HASH
        entry_hash = ChainedEntry.compute_hash(entry_id, text, prev_hash)
        self.entries.append(ChainedEntry(
            entry_id=entry_id, text=text,
            prev_hash=prev_hash, entry_hash=entry_hash,
        ))

    def verify_chain(self) -> bool:
        """Re-hash the entire chain. Returns True iff all entries verify
        AND prev_hash links are intact."""
        prev = GENESIS_HASH
        for e in self.entries:
            if e.prev_hash != prev:
                return False
            expected = ChainedEntry.compute_hash(e.entry_id, e.text, e.prev_hash)
            if e.entry_hash != expected:
                return False
            prev = e.entry_hash
        return True

    def read_all(self) -> Iterable[ChainedEntry]:
        return list(self.entries)

    # --- Tampering primitives used by Axis 3 evaluator (NOT a normal API) ---
    def _force_overwrite(self, idx: int, new_text: str) -> None:
        """Direct storage tamper: replace text without recomputing hash."""
        e = self.entries[idx]
        self.entries[idx] = replace(e, text=new_text)

    def _force_swap(self, i: int, j: int) -> None:
        """Direct storage tamper: swap two entries."""
        self.entries[i], self.entries[j] = self.entries[j], self.entries[i]

    def _force_delete(self, idx: int) -> None:
        """Direct storage tamper: remove entry without rotating chain."""
        del self.entries[idx]

    def _force_inject(self, idx: int, fake_id: str, fake_text: str) -> None:
        """Direct storage tamper: insert new entry with valid-looking but
        wrong hash (forgery attempt)."""
        prev_hash = self.entries[idx - 1].entry_hash if idx > 0 else GENESIS_HASH
        # Forge: write the entry but with a hash that doesn't match the chain
        # downstream (we don't update subsequent prev_hash refs).
        forged_hash = ChainedEntry.compute_hash(fake_id, fake_text, prev_hash)
        self.entries.insert(idx, ChainedEntry(
            entry_id=fake_id, text=fake_text,
            prev_hash=prev_hash, entry_hash=forged_hash,
        ))


@dataclass
class FlatDictLedger:
    """Naive baseline: dict-based store with no provenance.

    Implements `ingest` and `read_all`. `verify_chain()` always returns
    True because it has no chain to verify — but this is the very
    failure mode Axis 3 measures: it cannot DETECT tampering, even if
    no actual tampering happened, the system has no way to prove it.
    """
    storage: dict[str, str] = field(default_factory=dict)
    insertion_order: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "flat_dict_ledger"

    def supports(self, axis) -> bool:
        # Naive baseline does NOT claim auditability — but we still
        # evaluate it so its 0.0 appears in the matrix per the
        # pre-registration's transparent-failure-reporting commitment.
        return False

    def ingest(self, entry_id: str, text: str) -> None:
        self.storage[entry_id] = text
        self.insertion_order.append(entry_id)

    def verify_chain(self) -> bool:
        # Naive system has no concept of chain — claims "verified" by
        # default. The audit axis catches this: when tampering happens,
        # this still returns True (false negative).
        return True

    def read_all(self):
        return [(eid, self.storage[eid]) for eid in self.insertion_order]

    def _force_overwrite(self, idx: int, new_text: str) -> None:
        eid = self.insertion_order[idx]
        self.storage[eid] = new_text

    def _force_swap(self, i: int, j: int) -> None:
        self.insertion_order[i], self.insertion_order[j] = (
            self.insertion_order[j], self.insertion_order[i],
        )

    def _force_delete(self, idx: int) -> None:
        eid = self.insertion_order.pop(idx)
        self.storage.pop(eid, None)

    def _force_inject(self, idx: int, fake_id: str, fake_text: str) -> None:
        self.storage[fake_id] = fake_text
        self.insertion_order.insert(idx, fake_id)


__all__ = [
    "ChainedEntry",
    "FlatDictLedger",
    "GENESIS_HASH",
    "HashChainedLedger",
]
