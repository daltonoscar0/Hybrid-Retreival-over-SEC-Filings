"""JSONL judgment I/O: append, resume, last-write-wins, skip handling.

Runs entirely against tmp_path -- never against data/qrels/, which this
module's own docstring calls human-only and which a repo hook blocks
agent writes to regardless.
"""

from __future__ import annotations

from pathlib import Path

from ticker.qrels import (
    Judgment,
    append_judgment,
    judged_pairs,
    latest_by_pair,
    load_judgments,
    load_qrels_dict,
    now_iso,
)


def _judgment(query_id="q1", chunk_id="c1", grade=2, session_id="s1") -> Judgment:
    return Judgment(
        query_id=query_id, chunk_id=chunk_id, grade=grade, judged_at=now_iso(), session_id=session_id
    )


def test_append_then_load_round_trips(tmp_path: Path):
    path = tmp_path / "standard.jsonl"
    append_judgment(path, _judgment())
    loaded = load_judgments(path)
    assert len(loaded) == 1
    assert loaded[0].query_id == "q1"
    assert loaded[0].chunk_id == "c1"
    assert loaded[0].grade == 2


def test_load_judgments_on_missing_file_returns_empty(tmp_path: Path):
    assert load_judgments(tmp_path / "does-not-exist.jsonl") == []


def test_append_creates_parent_directories(tmp_path: Path):
    path = tmp_path / "nested" / "standard.jsonl"
    append_judgment(path, _judgment())
    assert path.exists()


def test_grade_none_round_trips_for_skip(tmp_path: Path):
    path = tmp_path / "standard.jsonl"
    append_judgment(path, _judgment(grade=None))
    loaded = load_judgments(path)
    assert loaded[0].grade is None


def test_latest_by_pair_last_line_wins(tmp_path: Path):
    path = tmp_path / "standard.jsonl"
    append_judgment(path, _judgment(grade=1))
    append_judgment(path, _judgment(grade=3))  # a --redo correction
    latest = latest_by_pair(load_judgments(path))
    assert latest[("q1", "c1")].grade == 3
    assert len(latest) == 1


def test_load_qrels_dict_shape(tmp_path: Path):
    path = tmp_path / "standard.jsonl"
    append_judgment(path, _judgment(query_id="q1", chunk_id="c1", grade=3))
    append_judgment(path, _judgment(query_id="q1", chunk_id="c2", grade=0))
    append_judgment(path, _judgment(query_id="q2", chunk_id="c3", grade=2))
    qrels = load_qrels_dict(path)
    assert qrels == {"q1": {"c1": 3, "c2": 0}, "q2": {"c3": 2}}


def test_load_qrels_dict_excludes_skips(tmp_path: Path):
    path = tmp_path / "standard.jsonl"
    append_judgment(path, _judgment(query_id="q1", chunk_id="c1", grade=None))
    append_judgment(path, _judgment(query_id="q1", chunk_id="c2", grade=1))
    qrels = load_qrels_dict(path)
    assert qrels == {"q1": {"c2": 1}}


def test_load_qrels_dict_uses_latest_grade_on_redo(tmp_path: Path):
    path = tmp_path / "standard.jsonl"
    append_judgment(path, _judgment(query_id="q1", chunk_id="c1", grade=0))
    append_judgment(path, _judgment(query_id="q1", chunk_id="c1", grade=2))
    assert load_qrels_dict(path) == {"q1": {"c1": 2}}


def test_judged_pairs_includes_skips(tmp_path: Path):
    path = tmp_path / "standard.jsonl"
    append_judgment(path, _judgment(query_id="q1", chunk_id="c1", grade=None))
    append_judgment(path, _judgment(query_id="q1", chunk_id="c2", grade=3))
    assert judged_pairs(path) == {("q1", "c1"), ("q1", "c2")}


def test_judged_pairs_on_missing_file_is_empty(tmp_path: Path):
    assert judged_pairs(tmp_path / "missing.jsonl") == set()
