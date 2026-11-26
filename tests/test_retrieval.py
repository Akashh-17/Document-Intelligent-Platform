"""Hybrid retrieval: RRF maths, tokenizer, BM25 freshness, and the full pipeline."""

from __future__ import annotations

import pytest

from rag_agent.config import settings
from rag_agent.ingest import ensure_seeded, ingest_files
from rag_agent.retrieval import hybrid_search
from rag_agent.retrieval.bm25 import get_bm25_index, tokenize
from rag_agent.retrieval.fusion import reciprocal_rank_fusion
from rag_agent.vectorstore import delete_by_source

# --- RRF ------------------------------------------------------------------------


def test_rrf_matches_formula():
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)


def test_rrf_rewards_agreement_over_a_single_first_place():
    # "b" is 2nd in both lists; "a" is 1st in one list only -> b wins
    fused = reciprocal_rank_fusion([["a", "b"], ["c", "b"]], k=60)
    assert fused[0][0] == "b"


def test_rrf_ties_are_deterministic_and_empty_input_ok():
    assert [d for d, _ in reciprocal_rank_fusion([["x"], ["y"]])] == ["x", "y"]
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_rejects_negative_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=-1)


# --- tokenizer -----------------------------------------------------------------


def test_tokenizer_keeps_identifiers_and_their_parts():
    toks = tokenize("What does AUR-2001 mean for query.timeout_ms?")
    assert {"aur-2001", "aur", "2001", "query.timeout_ms", "timeout"} <= set(toks)
    assert "what" not in toks and "does" not in toks  # stopwords dropped


# --- pipeline ------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _seed():
    ensure_seeded()


def test_exact_error_code_is_found_first():
    hits = hybrid_search("AUR-2005")
    assert hits and hits[0].source == "aurora_error_codes.md"
    assert "CROSS_TENANT_JOIN" in hits[0].text
    assert hits[0].bm25_rank == 1  # the keyword retriever is what found it


def test_paraphrase_is_found():
    hits = hybrid_search("Who gets paged if the main on-call engineer ignores the alert?")
    assert any("secondary on-call" in h.text for h in hits)


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid", "hybrid_rerank"])
def test_offtopic_question_returns_nothing(mode):
    assert hybrid_search("Give me a recipe for chocolate cake", mode=mode) == []


def test_every_mode_respects_k():
    for mode in ("dense", "bm25", "hybrid", "hybrid_rerank"):
        assert len(hybrid_search("Aurora retention", k=2, mode=mode)) <= 2


def test_hits_explain_themselves():
    hit = hybrid_search("AUR-1007 backpressure")[0]
    signals = hit.signals()
    assert {"dense_rank", "bm25_rank", "rerank_score"} <= signals.keys()
    assert 0.0 <= hit.relevance <= 1.0


def test_bm25_index_sees_new_uploads_immediately():
    marker = "Zephyrion quasar flux capacitor"
    assert not get_bm25_index().search(marker, k=5)
    ingest_files([("zephyr.md", f"Internal note: the {marker} lives in rack 7.".encode())])
    try:
        found = get_bm25_index().search(marker, k=5)
        assert found and found[0][0].metadata["source"] == "zephyr.md"
    finally:
        delete_by_source("zephyr.md")
    assert not get_bm25_index().search(marker, k=5)  # and forgets deleted ones


def test_reranker_failure_degrades_to_hybrid(monkeypatch):
    import rag_agent.retrieval.hybrid as hybrid

    monkeypatch.setattr(hybrid, "get_reranker", lambda: None)
    hits = hybrid_search("AUR-2005", mode="hybrid_rerank")
    assert hits and hits[0].rerank_score is None and hits[0].rrf_score is not None


def test_settings_defaults_are_sane():
    assert settings.retrieval_mode == "hybrid_rerank"
    assert settings.rerank_top_n >= settings.retriever_k
