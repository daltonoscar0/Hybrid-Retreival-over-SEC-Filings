"""Dense retriever: the FAISS round trip, chunk_id mapping, k clamping, the
empty index, and the one thing that fails silently rather than loudly -- the
query prefix policy drifting apart from the policy the index was built under.

Almost everything here injects a fake encoder and never touches the network or
the model cache. The fake is a bag-of-words embedding over a small fixed
vocabulary, L2-normalized, so cosine search over it behaves like cosine search
over real embeddings and the assertions are about ranking rather than about
whatever numbers a stub happened to return. None of the instruction prefix's
words are in that vocabulary, which is what lets one fake serve both the
"prefix is applied" tests and the "ranking is correct" tests.

The integration test at the bottom is the only one that loads
`bge-base-en-v1.5`, and it skips when the snapshot is not already in the
Hugging Face cache rather than downloading 440 MB inside a test run.

Index builds go through `scripts.index_dense.build_index` against a small
on-disk DuckDB under `tmp_path`, never against `data/ticker.duckdb` and never
against anything in `data/qrels` or `data/spotcheck`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
import pytest

from ticker import db
from ticker.records import Chunk, Filing, Section
from ticker.retrieval import dense
from ticker.retrieval.dense import (
    EMBEDDING_DIM,
    MANIFEST_FILENAME,
    QUERY_INSTRUCTION,
    DenseEncoder,
    DenseRetriever,
    load_retriever,
)

# scripts/ is not a package; import scripts/index_dense.py by file path, the
# same way tests/test_bm25.py imports scripts/index.py.
_SPEC = importlib.util.spec_from_file_location(
    "ticker_scripts_index_dense",
    Path(__file__).resolve().parents[1] / "scripts" / "index_dense.py",
)
index_dense_script = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = index_dense_script
_SPEC.loader.exec_module(index_dense_script)

FILED_AT = datetime(2024, 1, 1, tzinfo=timezone.utc)
FAKE_REVISION = "0" * 40

_VOCAB = (
    "goodwill", "impairment", "charge", "quarter", "customer", "concentration",
    "risk", "semiconductor", "foreign", "exchange", "revenue", "litigation",
)


# --- fake model -------------------------------------------------------------


class RecordingModel:
    """Stands in for `SentenceTransformer`. Records every string handed to it
    so the prefix policy is directly observable, and returns a deterministic
    bag-of-words embedding so ranking assertions mean something."""

    def __init__(self) -> None:
        self.seen: list[str] = []
        self.max_seq_length = 8

    def encode(
        self,
        sentences,
        *,
        batch_size,
        normalize_embeddings,
        convert_to_numpy,
        show_progress_bar,
    ):
        self.seen.extend(sentences)
        vectors = np.zeros((len(sentences), EMBEDDING_DIM), dtype=np.float32)
        for row, text in enumerate(sentences):
            words = text.lower().replace(".", " ").split()
            for col, term in enumerate(_VOCAB):
                vectors[row, col] = words.count(term)
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.where(norms == 0, 1.0, norms)
        return vectors

    def tokenizer(self, batch, add_special_tokens, truncation):
        return {"input_ids": [text.split() for text in batch]}


def _fake_encoder() -> DenseEncoder:
    return DenseEncoder(RecordingModel())


# --- corpus and index fixtures ---------------------------------------------

_CHUNKS = [
    ("z-chunk", "goodwill impairment charge recognized in the current quarter"),
    ("a-chunk", "customer concentration risk in the semiconductor segment"),
    ("m-chunk", "foreign exchange impact on reported revenue for the period"),
]


def _seed_corpus(db_path: Path, chunks: list[tuple[str, str]]) -> None:
    con = db.connect(str(db_path))
    try:
        db.create_schema(con)
        db.insert_filing(
            con,
            Filing(
                accession="acc-1",
                cik=1,
                ticker="TEST",
                sector="test_sector",
                form="10-K",
                filed_at=FILED_AT,
                period_end=None,
                url="https://example.com/acc-1",
            ),
        )
        db.insert_section(
            con,
            Section(
                section_id="acc-1#1A",
                accession="acc-1",
                item="1A",
                text="body",
                char_start=0,
                char_end=4,
            ),
        )
        for chunk_id, text in chunks:
            db.insert_chunk(
                con, Chunk(chunk_id=chunk_id, section_id="acc-1#1A", sentence_ids=(), text=text)
            )
    finally:
        con.close()


@pytest.fixture
def built_index(tmp_path, monkeypatch):
    """A real FAISS index and manifest built by the real build path, with only
    the model faked out. `resolve_revision` is stubbed so the build does not
    consult the Hugging Face cache or the network."""
    monkeypatch.setattr(index_dense_script, "resolve_revision", lambda *_: FAKE_REVISION)
    db_path = tmp_path / "corpus.duckdb"
    _seed_corpus(db_path, _CHUNKS)
    out_dir = tmp_path / "index"
    model = RecordingModel()
    manifest = index_dense_script.build_index(
        db_path, out_dir, model=model, device="cpu", show_progress=False
    )
    return out_dir, manifest, model


# --- prefix policy ----------------------------------------------------------


def test_queries_get_the_instruction_and_passages_do_not():
    model = RecordingModel()
    encoder = DenseEncoder(model)

    encoder.encode_passages(["goodwill impairment charge"])
    assert model.seen == ["goodwill impairment charge"]

    model.seen.clear()
    encoder.encode_queries(["goodwill impairment charge"])
    assert model.seen == [QUERY_INSTRUCTION + "goodwill impairment charge"]


def test_build_records_the_prefix_policy_in_the_manifest(built_index):
    out_dir, manifest, model = built_index
    assert manifest["query_instruction"] == QUERY_INSTRUCTION
    assert manifest["passage_instruction"] == ""
    assert manifest["uses_query_instruction"] is True

    # every string the build handed the model is a bare chunk text
    assert sorted(model.seen) == sorted(text for _, text in _CHUNKS)
    assert not any(s.startswith(QUERY_INSTRUCTION) for s in model.seen)


def test_retriever_uses_the_manifest_prefix_not_the_module_constant(built_index, monkeypatch):
    """The drift guard. If `load_retriever` read `QUERY_INSTRUCTION` off this
    module instead of off the manifest, editing the constant would silently
    change how every already-built index is queried and no test or error would
    fire. Rewriting the manifest to a different instruction and watching the
    retriever follow it is what pins that down."""
    out_dir, _, _ = built_index
    manifest_path = out_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["query_instruction"] = "SOME OTHER INSTRUCTION: "
    manifest_path.write_text(json.dumps(manifest))

    query_model = RecordingModel()
    monkeypatch.setattr(dense, "load_model", lambda *a, **kw: query_model)

    retriever = load_retriever(out_dir)
    retriever.search("customer concentration risk", k=1)

    assert query_model.seen == ["SOME OTHER INSTRUCTION: customer concentration risk"]


def test_retriever_pins_the_model_name_and_revision_from_the_manifest(built_index, monkeypatch):
    out_dir, manifest, _ = built_index
    assert manifest["model_revision"] == FAKE_REVISION
    assert manifest["model_revision"] != "main"

    captured: dict = {}

    def _capture(model_name, *, revision=None, device=None):
        captured["model_name"] = model_name
        captured["revision"] = revision
        return RecordingModel()

    monkeypatch.setattr(dense, "load_model", _capture)
    load_retriever(out_dir)

    assert captured["model_name"] == manifest["model_name"]
    assert captured["revision"] == FAKE_REVISION


# --- FAISS round trip -------------------------------------------------------


def test_search_returns_ranked_chunks_by_cosine(built_index):
    out_dir, manifest, _ = built_index
    assert manifest["index_type"] == "IndexFlatIP"
    assert manifest["normalize"] is True
    assert manifest["embedding_dim"] == EMBEDDING_DIM

    retriever = load_retriever(out_dir, encoder=_fake_encoder())
    results = retriever.search("customer concentration risk", k=3)

    assert results
    assert results[0].chunk_id == "a-chunk"
    assert all(a.score >= b.score for a, b in zip(results, results[1:]))
    # unit vectors, so every inner product is a cosine in [-1, 1]
    assert all(-1.0001 <= rc.score <= 1.0001 for rc in results)


def test_chunk_ids_round_trip_from_index_position_to_database(built_index):
    """FAISS knows integer positions and nothing else. The chunk_ids.json
    sidecar is the only link from a search result back to a corpus row, and
    an off-by-one in it would return real chunk_ids for the wrong vectors,
    which no score check would catch."""
    out_dir, _, _ = built_index
    retriever = load_retriever(out_dir, encoder=_fake_encoder())

    known_ids = {chunk_id for chunk_id, _ in _CHUNKS}
    assert set(retriever._chunk_ids) == known_ids
    # build order is `ORDER BY chunk_id`, independent of insertion order
    assert retriever._chunk_ids == sorted(known_ids)

    text_by_id = dict(_CHUNKS)
    for query_terms, expected in [
        ("goodwill impairment", "z-chunk"),
        ("customer concentration", "a-chunk"),
        ("foreign exchange revenue", "m-chunk"),
    ]:
        top = retriever.search(query_terms, k=3)[0]
        assert top.chunk_id == expected
        # the winning chunk really does contain the query terms
        assert all(term in text_by_id[top.chunk_id] for term in query_terms.split())


def test_k_is_capped_to_corpus_size(built_index):
    out_dir, _, _ = built_index
    retriever = load_retriever(out_dir, encoder=_fake_encoder())
    assert len(retriever) == 3
    assert len(retriever.search("goodwill impairment", k=1000)) == 3


def test_non_positive_k_returns_nothing(built_index):
    out_dir, _, _ = built_index
    retriever = load_retriever(out_dir, encoder=_fake_encoder())
    assert retriever.search("goodwill", k=0) == []
    assert retriever.search("goodwill", k=-1) == []


def test_empty_index_returns_nothing_rather_than_faiss_sentinels():
    """FAISS pads a short result set with position -1. On an empty index every
    position comes back -1, and -1 indexes a python list from the end, so the
    unguarded version of this returns the last chunk_id in the sidecar with a
    garbage score instead of returning nothing."""
    retriever = DenseRetriever(
        index=faiss.IndexFlatIP(EMBEDDING_DIM), chunk_ids=[], encoder=_fake_encoder()
    )
    assert len(retriever) == 0
    assert retriever.search("goodwill impairment", k=10) == []


def test_index_and_sidecar_length_mismatch_raises(built_index):
    out_dir, _, _ = built_index
    chunk_ids_path = out_dir / "chunk_ids.json"
    chunk_ids_path.write_text(json.dumps(json.loads(chunk_ids_path.read_text())[:2]))

    with pytest.raises(RuntimeError, match="different builds"):
        load_retriever(out_dir, encoder=_fake_encoder())


def test_missing_index_raises_with_a_pointer_to_the_build_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="index_dense.py"):
        load_retriever(tmp_path / "nope", encoder=_fake_encoder())


# --- build metadata ---------------------------------------------------------


def test_manifest_carries_the_corpus_fingerprint_and_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(index_dense_script, "resolve_revision", lambda *_: FAKE_REVISION)
    db_path = tmp_path / "corpus.duckdb"
    _seed_corpus(db_path, _CHUNKS)

    full = index_dense_script.build_index(
        db_path, tmp_path / "full", model=RecordingModel(), device="cpu", show_progress=False
    )
    subset = index_dense_script.build_index(
        db_path,
        tmp_path / "subset",
        limit=2,
        model=RecordingModel(),
        device="cpu",
        show_progress=False,
    )

    assert full["n_chunks"] == 3 and full["limit"] is None
    assert subset["n_chunks"] == 2 and subset["limit"] == 2
    # a subset index must not claim the fingerprint of the full corpus
    assert full["corpus_sha256"] != subset["corpus_sha256"]


def test_truncation_count_is_reported(tmp_path, monkeypatch):
    """A silently truncated chunk loses its tail from the embedding. The count
    belongs in the manifest, not in nobody's head."""
    monkeypatch.setattr(index_dense_script, "resolve_revision", lambda *_: FAKE_REVISION)
    db_path = tmp_path / "corpus.duckdb"
    # RecordingModel.max_seq_length is 8 whitespace tokens
    _seed_corpus(
        db_path,
        [("short", "goodwill impairment"), ("long", " ".join(["revenue"] * 40))],
    )
    manifest = index_dense_script.build_index(
        db_path, tmp_path / "index", model=RecordingModel(), device="cpu", show_progress=False
    )
    assert manifest["n_chunks_truncated"] == 1


def test_empty_corpus_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(index_dense_script, "resolve_revision", lambda *_: FAKE_REVISION)
    db_path = tmp_path / "corpus.duckdb"
    _seed_corpus(db_path, [])
    with pytest.raises(RuntimeError, match="no chunks found"):
        index_dense_script.build_index(
            db_path, tmp_path / "index", model=RecordingModel(), device="cpu", show_progress=False
        )


# --- integration ------------------------------------------------------------


def _model_is_cached() -> bool:
    try:
        return dense.cached_snapshot_revision() is not None
    except Exception:
        return False


@pytest.mark.skipif(
    not _model_is_cached(),
    reason=f"{dense.MODEL_NAME} is not in the Hugging Face cache; "
    "run scripts/index_dense.py --limit 2000 once to fetch it",
)
def test_real_model_matches_a_paraphrase_bm25_would_miss(tmp_path):
    """The whole reason for a second arm. The query shares no content word
    with the chunk it should rank first, so a lexical ranker scores that chunk
    zero and the dense arm has to carry it on meaning alone."""
    db_path = tmp_path / "corpus.duckdb"
    _seed_corpus(
        db_path,
        [
            ("a-writedown", "The Company recorded a non-cash write-down of intangible assets."),
            ("b-buyback", "The board authorized an additional share repurchase program."),
            ("c-weather", "Severe weather closed two branch offices during the period."),
        ],
    )
    out_dir = tmp_path / "index"
    manifest = index_dense_script.build_index(db_path, out_dir, show_progress=False)

    assert manifest["model_name"] == dense.MODEL_NAME
    assert len(manifest["model_revision"]) == 40
    assert manifest["model_revision"] != "main"
    assert manifest["embedding_dim"] == EMBEDDING_DIM

    retriever = load_retriever(out_dir)
    results = retriever.search("goodwill impairment charge", k=3)

    assert [rc.chunk_id for rc in results][0] == "a-writedown"
    assert all(-1.0001 <= rc.score <= 1.0001 for rc in results)
    assert all(a.score >= b.score for a, b in zip(results, results[1:]))
