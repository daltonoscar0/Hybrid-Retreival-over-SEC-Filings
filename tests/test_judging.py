"""Judging-session core logic, driven by a scripted key sequence instead of
a real keyboard. Everything writes to tmp_path, never to data/qrels/.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from ticker import db
from ticker.judging import (
    fetch_chunk_displays,
    rejudge_path,
    run_judging_session,
    sample_for_rejudge,
)
from ticker.qrels import Judgment, append_judgment, load_judgments, load_qrels_dict
from ticker.records import Chunk, Filing, Section

FILED_AT = datetime(2024, 3, 1, tzinfo=timezone.utc)


@pytest.fixture
def con():
    connection = db.connect(":memory:")
    db.create_schema(connection)
    db.insert_filing(
        connection,
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
        connection,
        Section(section_id="acc-1#1A", accession="acc-1", item="1A", text="body", char_start=0, char_end=4),
    )
    for i in range(4):
        db.insert_chunk(
            connection,
            Chunk(chunk_id=f"c{i}", section_id="acc-1#1A", sentence_ids=(), text=f"chunk text {i}"),
        )
    return connection


def _keys(sequence):
    it = iter(sequence)
    return lambda: next(it)


def test_grading_writes_one_record_per_chunk(con, tmp_path: Path):
    out = tmp_path / "standard.jsonl"
    queries = [("q1", "some query")]
    pool = {"q1": ["c0", "c1", "c2"]}
    summary = run_judging_session(
        con, queries, pool, instruction="standard", out_path=out,
        session_id="s1", next_key=_keys(["3", "0", "2"]),
    )
    assert summary.judged_this_session == 3
    assert not summary.quit_early
    records = load_judgments(out)
    assert [(r.chunk_id, r.grade) for r in records] == [("c0", 3), ("c1", 0), ("c2", 2)]
    assert all(r.query_id == "q1" and r.session_id == "s1" for r in records)


def test_skip_writes_grade_none(con, tmp_path: Path):
    out = tmp_path / "standard.jsonl"
    run_judging_session(
        con, [("q1", "q")], {"q1": ["c0"]}, instruction="standard",
        out_path=out, session_id="s1", next_key=_keys(["s"]),
    )
    records = load_judgments(out)
    assert records[0].grade is None


def test_quit_stops_before_remaining_items(con, tmp_path: Path):
    out = tmp_path / "standard.jsonl"
    summary = run_judging_session(
        con, [("q1", "q")], {"q1": ["c0", "c1", "c2"]}, instruction="standard",
        out_path=out, session_id="s1", next_key=_keys(["2", "q"]),
    )
    assert summary.judged_this_session == 1
    assert summary.quit_early
    assert len(load_judgments(out)) == 1


def test_invalid_key_reprompts_without_writing(con, tmp_path: Path):
    out = tmp_path / "standard.jsonl"
    run_judging_session(
        con, [("q1", "q")], {"q1": ["c0"]}, instruction="standard",
        out_path=out, session_id="s1", next_key=_keys(["x", "9", "1"]),
    )
    records = load_judgments(out)
    assert len(records) == 1
    assert records[0].grade == 1


def test_resumed_session_skips_already_judged_pairs(con, tmp_path: Path):
    out = tmp_path / "standard.jsonl"
    append_judgment(out, Judgment("q1", "c0", 2, "2024-01-01T00:00:00+00:00", "earlier-session"))

    summary = run_judging_session(
        con, [("q1", "q")], {"q1": ["c0", "c1"]}, instruction="standard",
        out_path=out, session_id="s2", next_key=_keys(["3"]),
    )
    assert summary.judged_this_session == 1
    records = load_judgments(out)
    assert [(r.chunk_id, r.session_id) for r in records] == [
        ("c0", "earlier-session"),
        ("c1", "s2"),
    ]


def test_redo_reshows_already_judged_pairs(con, tmp_path: Path):
    out = tmp_path / "standard.jsonl"
    append_judgment(out, Judgment("q1", "c0", 0, "2024-01-01T00:00:00+00:00", "earlier-session"))

    run_judging_session(
        con, [("q1", "q")], {"q1": ["c0"]}, instruction="standard",
        out_path=out, session_id="s2", next_key=_keys(["3"]), redo=True,
    )
    qrels = load_qrels_dict(out)
    assert qrels == {"q1": {"c0": 3}}  # latest line wins


def test_novelty_instruction_writes_to_its_own_out_path(con, tmp_path: Path):
    standard_out = tmp_path / "standard.jsonl"
    novelty_out = tmp_path / "novelty.jsonl"
    run_judging_session(
        con, [("q1", "q")], {"q1": ["c0"]}, instruction="novelty",
        out_path=novelty_out, session_id="s1", next_key=_keys(["2"]),
    )
    assert load_judgments(novelty_out)
    assert load_judgments(standard_out) == []


def test_unknown_instruction_raises(con, tmp_path: Path):
    with pytest.raises(ValueError):
        run_judging_session(
            con, [("q1", "q")], {"q1": ["c0"]}, instruction="bogus",
            out_path=tmp_path / "x.jsonl", session_id="s1", next_key=_keys(["2"]),
        )


def test_fetch_chunk_displays_carries_firm_period_item(con):
    displays = fetch_chunk_displays(con, ["c0"])
    d = displays["c0"]
    assert d.ticker == "TEST"
    assert d.form == "10-K"
    assert d.item == "1A"
    assert d.filed_at == FILED_AT


def test_fetch_chunk_displays_empty_input_returns_empty(con):
    assert fetch_chunk_displays(con, []) == {}


def test_rejudge_path_is_a_sibling_file():
    assert rejudge_path(Path("data/qrels/standard.jsonl")) == Path("data/qrels/standard.rejudge.jsonl")
    assert rejudge_path(Path("data/qrels/novelty.jsonl")) == Path("data/qrels/novelty.rejudge.jsonl")


def test_sample_for_rejudge_excludes_skips_and_respects_seed(tmp_path: Path):
    path = tmp_path / "standard.jsonl"
    for i in range(5):
        append_judgment(path, Judgment("q1", f"c{i}", i % 4, "t", "s1"))
    append_judgment(path, Judgment("q1", "c-skip", None, "t", "s1"))

    sample_a = sample_for_rejudge(path, 3, seed=42)
    sample_b = sample_for_rejudge(path, 3, seed=42)
    assert sample_a == sample_b
    assert len(sample_a) == 3
    assert "c-skip" not in {chunk_id for _, chunk_id in sample_a}


def test_sample_for_rejudge_caps_at_available_graded_pairs(tmp_path: Path):
    path = tmp_path / "standard.jsonl"
    append_judgment(path, Judgment("q1", "c0", 1, "t", "s1"))
    assert len(sample_for_rejudge(path, 100, seed=1)) == 1
