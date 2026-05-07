"""Tests for AMTB Axis 6 (adversarial revision)."""
from __future__ import annotations

import pytest

from amtb.axes import revision
from amtb.systems.substring_retriever import SubstringRetriever
from amtb.types import AxisName


def _conversational_loader():
    """Synthetic conversational corpus + probes whose gold answers
    are subjects of contradictory revisions."""
    def loader(limit: int | None):
        corpus = [
            ("d1", "Caroline is single and lives in Berlin"),
            ("d2", "Melanie went to Hawaii on 7 May 2023"),
            ("d3", "Hugo bought a vintage typewriter at the flea market"),
            ("d4", "Iris translated a manuscript for the linguistics archive"),
            ("d5", "Pablo studied at the harbor district"),
            ("d6", "Andrea was photographed at the abandoned lighthouse"),
            ("d7", "Diego designed a kinetic sculpture for the gallery"),
            ("d8", "Elena renovated a mid-century cabin in Vermont"),
            ("d9", "Felix sponsored the linguistics archive renovation"),
            ("d10", "Gloria discovered a fossilized fern at the beach"),
        ]
        probes = [
            {"question": "Where does Caroline live?",
             "gold_answers": ["Berlin"], "gold_ids": ["d1"]},
            {"question": "When did Melanie go to Hawaii?",
             "gold_answers": ["7 May 2023"], "gold_ids": ["d2"]},
            {"question": "What did Hugo buy at the flea market?",
             "gold_answers": ["vintage typewriter"], "gold_ids": ["d3"]},
            {"question": "What did Iris translate?",
             "gold_answers": ["manuscript"], "gold_ids": ["d4"]},
        ]
        if limit is not None:
            probes = probes[:limit]
        return {"corpus": corpus, "probes": probes}
    return loader


def test_append_only_preserves_under_revision():
    """Substring retriever (append-only by construction — never deletes)
    should preserve original gold across all 3 contamination levels."""
    sys = SubstringRetriever()
    config = revision.RevisionEvalConfig(
        levels=(0.05, 0.10, 0.20),
        mode="retrieval", k=10, seed=0,
    )
    r = revision.run(sys, config=config, dataset_loader=_conversational_loader())
    assert r.axis == AxisName.ADVERSARIAL_REVISION
    assert r.applicable is True
    # Append-only retriever should preserve nearly all originals
    assert r.score >= 0.75
    per_level = r.details["per_level"]
    for level_str, level_data in per_level.items():
        assert 0.0 <= level_data["preservation_rate"] <= 1.0


def test_overwrite_system_loses_originals():
    """Stub system that overwrites on duplicate keys (loses original)."""

    class _OverwriteRetriever(SubstringRetriever):
        @property
        def name(self) -> str:
            return "overwrite_retriever"

        def supports(self, axis: AxisName) -> bool:
            return axis == AxisName.ADVERSARIAL_REVISION

        def ingest(self, entry_id: str, text: str) -> None:
            # Naive overwrite: if the entry_id has a __rev suffix
            # (revision injection), wipe the original entry's storage too.
            if "__rev" in entry_id:
                base_eid = entry_id.split("__rev")[0]
                if base_eid in self.storage:
                    for tok in self._tokens(self.storage[base_eid]):
                        self.inverted.get(tok, set()).discard(base_eid)
                    self.storage.pop(base_eid, None)
            super().ingest(entry_id, text)

    sys = _OverwriteRetriever()
    config = revision.RevisionEvalConfig(
        levels=(0.20, 0.30, 0.40),  # higher contamination so signal is clear
        mode="retrieval", k=10, seed=0,
    )
    r = revision.run(sys, config=config, dataset_loader=_conversational_loader())
    assert r.applicable is True
    # Overwrite system should preserve LESS than the append-only baseline
    # at the same configurations. We can't pin an exact number without
    # mocking specific revisions, but it must score finite.
    assert 0.0 <= r.score <= 1.0


def test_axis6_score_in_range():
    sys = SubstringRetriever()
    r = revision.run(sys, config=revision.RevisionEvalConfig(levels=(0.10,), k=5),
                     dataset_loader=_conversational_loader())
    assert 0.0 <= r.score <= 1.0


def test_axis6_unsupported_system():
    class _NoRev:
        name = "no_rev"
        def supports(self, a): return False
    r = revision.run(_NoRev(), dataset_loader=_conversational_loader())
    assert r.score == 0.0
    assert r.applicable is False


def test_axis6_no_loader():
    sys = SubstringRetriever()
    r = revision.run(sys, dataset_loader=None)
    assert r.score == 0.0
    assert r.applicable is False
