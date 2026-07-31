"""Blind novelty spot-check: the score-file contract, the decile
stratification, the blindness of the screen, and the session driven by a
scripted key sequence instead of a real keyboard.

Everything writes to tmp_path. Nothing here touches `data/spotcheck/`, which
is human-only, and nothing here writes a label file the project would treat
as real.

The blindness tests lean on a synthetic fixture built so the claim is
checkable: sentence text and sentence ids contain no digits, and every
novelty value is made only of the digits 2-9. Any digit in 2-9 reaching the
screen is therefore a digit of a score, and any digit at all outside the
progress line is something that should not be rendered.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ticker import db
from ticker.records import Filing, Section, Sentence

# scripts/ is not a package; import scripts/spotcheck.py by file path.
_SPEC = importlib.util.spec_from_file_location(
    "ticker_scripts_spotcheck",
    Path(__file__).resolve().parents[1] / "scripts" / "spotcheck.py",
)
spotcheck = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = spotcheck
_SPEC.loader.exec_module(spotcheck)

FILED_AT = datetime(2024, 3, 1, tzinfo=timezone.utc)
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _tag(i: int) -> str:
    return LETTERS[i // 26] + LETTERS[i % 26]


def _digit_free_text(i: int) -> str:
    return (
        f"the registrant maintains reserves for identified exposures marked {_tag(i)} "
        "and reviews them at the close of each reporting period"
    )


def _digit_restricted_scores(n: int) -> list[float]:
    """`n` distinct novelty values whose decimal form uses only the digits 2-9."""
    combos = ["".join(c) for c in itertools.product("23456789", repeat=3)]
    return [float(f"{a}.{b}{c}") for a, b, c in combos[:n]]


def _blind_fixture(n: int = 100):
    """Score rows and matching items, all digit-free except the scores."""
    scores = _digit_restricted_scores(n)
    scored = [
        spotcheck.ScoredSentence(sentence_id=f"sent-{_tag(i)}", novelty=score)
        for i, score in enumerate(scores)
    ]
    deciles = spotcheck.assign_deciles(scored)
    items = [
        spotcheck.SpotcheckItem(
            sentence_id=s.sentence_id,
            text=_digit_free_text(i),
            decile=deciles[s.sentence_id],
        )
        for i, s in enumerate(scored)
    ]
    return scored, items


def _scores_file(path: Path, scored) -> Path:
    with path.open("w") as f:
        for s in scored:
            f.write(json.dumps({"sentence_id": s.sentence_id, "novelty": s.novelty}) + "\n")
    return path


def _keys(sequence):
    it = iter(sequence)
    return lambda: next(it)


def _pearson(xs, ys) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy)


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
        Section(
            section_id="acc-1#1A",
            accession="acc-1",
            item="1A",
            text="body",
            char_start=0,
            char_end=4,
        ),
    )
    for i in range(100):
        db.insert_sentence(
            connection,
            Sentence(
                sentence_id=f"sent-{_tag(i)}",
                section_id="acc-1#1A",
                ordinal=i,
                text=_digit_free_text(i),
                filed_at=FILED_AT,
            ),
        )
    return connection


# --- score-file contract ---------------------------------------------------


def test_load_scores_reads_the_contract_and_ignores_extra_keys(tmp_path: Path):
    path = tmp_path / "scores.jsonl"
    path.write_text(
        json.dumps({"sentence_id": "a", "novelty": 1.5, "surprisal_firm": 9.0}) + "\n"
        "\n"  # blank lines are skipped
        + json.dumps({"sentence_id": "b", "novelty": -0.25}) + "\n"
    )
    scored = spotcheck.load_scores(path)
    assert [(s.sentence_id, s.novelty) for s in scored] == [("a", 1.5), ("b", -0.25)]


def test_load_scores_rejects_a_duplicate_sentence_id(tmp_path: Path):
    path = tmp_path / "scores.jsonl"
    path.write_text(
        json.dumps({"sentence_id": "a", "novelty": 1.0}) + "\n"
        + json.dumps({"sentence_id": "a", "novelty": 2.0}) + "\n"
    )
    with pytest.raises(ValueError, match="duplicate sentence_id"):
        spotcheck.load_scores(path)


def test_load_scores_rejects_non_finite_novelty(tmp_path: Path):
    path = tmp_path / "scores.jsonl"
    path.write_text(json.dumps({"sentence_id": "a", "novelty": "NaN"}) + "\n")
    with pytest.raises(ValueError, match="non-finite"):
        spotcheck.load_scores(path)


def test_load_scores_rejects_a_missing_key(tmp_path: Path):
    path = tmp_path / "scores.jsonl"
    path.write_text(json.dumps({"sentence_id": "a"}) + "\n")
    with pytest.raises(ValueError, match="sentence_id.*novelty"):
        spotcheck.load_scores(path)


# --- stratification --------------------------------------------------------


def test_deciles_are_rank_based_and_equally_sized(tmp_path: Path):
    scored, _ = _blind_fixture(100)
    deciles = spotcheck.assign_deciles(scored)
    counts = {d: sum(1 for v in deciles.values() if v == d) for d in range(1, 11)}
    assert counts == {d: 10 for d in range(1, 11)}

    by_novelty = sorted(scored, key=lambda s: s.novelty)
    sequence = [deciles[s.sentence_id] for s in by_novelty]
    assert sequence == sorted(sequence)  # decile rises with novelty
    assert sequence[0] == 1 and sequence[-1] == 10


def test_deciles_on_an_empty_score_file_are_empty():
    assert spotcheck.assign_deciles([]) == {}


def test_sample_takes_ten_from_every_decile(tmp_path: Path):
    scored, _ = _blind_fixture(500)
    chosen = spotcheck.stratified_sample(scored, per_decile=10, seed=0)
    assert len(chosen) == 100
    counts = {d: sum(1 for _, decile in chosen if decile == d) for d in range(1, 11)}
    assert counts == {d: 10 for d in range(1, 11)}
    assert len({sentence_id for sentence_id, _ in chosen}) == 100


def test_sample_is_reproducible_for_a_seed_and_differs_across_seeds():
    scored, _ = _blind_fixture(500)
    a = spotcheck.stratified_sample(scored, per_decile=10, seed=7)
    b = spotcheck.stratified_sample(scored, per_decile=10, seed=7)
    c = spotcheck.stratified_sample(scored, per_decile=10, seed=8)
    assert a == b
    assert a != c


def test_presentation_order_carries_no_decile_signal():
    scored, _ = _blind_fixture(500)
    chosen = spotcheck.stratified_sample(scored, per_decile=10, seed=0)
    deciles = [decile for _, decile in chosen]
    assert deciles != sorted(deciles)
    assert deciles != sorted(deciles, reverse=True)
    assert abs(_pearson(list(range(len(deciles))), deciles)) < 0.25


# --- blindness -------------------------------------------------------------


def test_screen_never_shows_the_score_the_decile_or_the_rank():
    """The required guarantee, checked on every one of the 100 screens.

    Progress is rendered at a fixed (0, 100) so the only digits the progress
    line can contribute are 0 and 1, and every novelty value in the fixture
    is built from the digits 2-9. So: no digit of any score appears anywhere
    on screen, and no digit at all appears outside the progress line, which
    also rules out the decile (1-10) and a rank being printed.
    """
    _, items = _blind_fixture(100)
    score_digits = set("23456789")

    for item in items:
        screen = spotcheck.visible_text(spotcheck.render_item(item, (0, 100)))
        assert not score_digits & set(screen), f"score digit on screen: {screen!r}"

        lowered = screen.lower()
        for banned in ("decile", "rank", "score", "novelty", "surprisal", "percentile"):
            assert banned not in lowered, f"{banned!r} on screen: {screen!r}"

        body = [
            line
            for line in screen.splitlines()
            if not line.startswith(spotcheck.PROGRESS_PREFIX)
        ]
        assert not any(ch.isdigit() for ch in "\n".join(body)), f"digit on screen: {screen!r}"


def test_full_session_transcript_never_prints_a_score_or_a_decile(tmp_path: Path):
    _, items = _blind_fixture(100)
    lines: list[str] = []
    spotcheck.run_spotcheck_session(
        items,
        out_path=tmp_path / "labels.jsonl",
        session_id="s1",
        next_key=_keys(["n", "b"] * 50),
        echo=lines.append,
        redo=True,
    )
    transcript = spotcheck.visible_text("\n".join(lines)).lower()
    assert "decile" not in transcript
    for item in items:
        assert f"decile {item.decile}" not in transcript
    scores = {f"{v}" for v in _digit_restricted_scores(100)}
    assert not any(score in transcript for score in scores)


def test_screen_states_the_instruction_and_the_keys():
    _, items = _blind_fixture(1)
    screen = spotcheck.visible_text(spotcheck.render_item(items[0], (0, 100)))
    assert f"judge for: {spotcheck.INSTRUCTION}" in screen
    assert "not boilerplate carried over from a prior filing" in screen
    assert f"[{spotcheck.LABEL_LEGEND}]" in screen
    assert " ".join(items[0].text.split()) in " ".join(screen.split())  # wrapped


def test_visible_text_strips_colour_escapes_only():
    assert spotcheck.visible_text("\033[2mdim\033[0m text") == "dim text"


# --- session ---------------------------------------------------------------


def test_session_writes_one_record_per_keypress(tmp_path: Path):
    _, items = _blind_fixture(3)
    out = tmp_path / "labels.jsonl"
    summary = spotcheck.run_spotcheck_session(
        items, out_path=out, session_id="s1", next_key=_keys(["n", "b", "n"]),
        echo=lambda _: None,
    )
    assert summary.labeled_this_session == 3
    assert not summary.quit_early
    records = spotcheck.load_labels(out)
    assert [(r.sentence_id, r.label) for r in records] == [
        (items[0].sentence_id, "novel"),
        (items[1].sentence_id, "boilerplate"),
        (items[2].sentence_id, "novel"),
    ]
    assert all(r.session_id == "s1" for r in records)
    assert [r.decile for r in records] == [item.decile for item in items]


def test_skip_writes_label_none(tmp_path: Path):
    _, items = _blind_fixture(1)
    out = tmp_path / "labels.jsonl"
    spotcheck.run_spotcheck_session(
        items, out_path=out, session_id="s1", next_key=_keys(["s"]), echo=lambda _: None
    )
    records = spotcheck.load_labels(out)
    assert records[0].label is None


def test_quit_stops_before_the_remaining_items(tmp_path: Path):
    _, items = _blind_fixture(4)
    out = tmp_path / "labels.jsonl"
    summary = spotcheck.run_spotcheck_session(
        items, out_path=out, session_id="s1", next_key=_keys(["n", "q"]),
        echo=lambda _: None,
    )
    assert summary.labeled_this_session == 1
    assert summary.quit_early
    assert len(spotcheck.load_labels(out)) == 1


def test_invalid_key_reprompts_without_writing(tmp_path: Path):
    _, items = _blind_fixture(1)
    out = tmp_path / "labels.jsonl"
    spotcheck.run_spotcheck_session(
        items, out_path=out, session_id="s1", next_key=_keys(["x", "3", "b"]),
        echo=lambda _: None,
    )
    records = spotcheck.load_labels(out)
    assert len(records) == 1
    assert records[0].label == "boilerplate"


def test_resumed_session_skips_already_labeled_sentences(tmp_path: Path):
    _, items = _blind_fixture(3)
    out = tmp_path / "labels.jsonl"
    spotcheck.run_spotcheck_session(
        items, out_path=out, session_id="s1", next_key=_keys(["n", "q"]),
        echo=lambda _: None,
    )
    summary = spotcheck.run_spotcheck_session(
        items, out_path=out, session_id="s2", next_key=_keys(["b", "b"]),
        echo=lambda _: None,
    )
    assert summary.already_labeled == 1
    assert summary.labeled_this_session == 2
    records = spotcheck.load_labels(out)
    assert [(r.sentence_id, r.session_id) for r in records] == [
        (items[0].sentence_id, "s1"),
        (items[1].sentence_id, "s2"),
        (items[2].sentence_id, "s2"),
    ]


def test_redo_represents_already_labeled_sentences(tmp_path: Path):
    _, items = _blind_fixture(1)
    out = tmp_path / "labels.jsonl"
    spotcheck.run_spotcheck_session(
        items, out_path=out, session_id="s1", next_key=_keys(["n"]), echo=lambda _: None
    )
    spotcheck.run_spotcheck_session(
        items, out_path=out, session_id="s2", next_key=_keys(["b"]),
        echo=lambda _: None, redo=True,
    )
    records = spotcheck.load_labels(out)
    assert [r.label for r in records] == ["novel", "boilerplate"]


def test_progress_counts_prior_sessions(tmp_path: Path):
    _, items = _blind_fixture(4)
    out = tmp_path / "labels.jsonl"
    spotcheck.run_spotcheck_session(
        items, out_path=out, session_id="s1", next_key=_keys(["n", "n", "q"]),
        echo=lambda _: None,
    )
    lines: list[str] = []
    spotcheck.run_spotcheck_session(
        items, out_path=out, session_id="s2", next_key=_keys(["b", "q"]),
        echo=lines.append,
    )
    first_screen = spotcheck.visible_text(lines[0])
    assert f"{spotcheck.PROGRESS_PREFIX}3/4" in first_screen


# --- corpus join -----------------------------------------------------------


def test_build_sample_joins_text_from_the_corpus(con, tmp_path: Path):
    scored, _ = _blind_fixture(100)
    items = spotcheck.build_sample(con, scored, per_decile=10, seed=0, echo=lambda _: None)
    assert len(items) == 100
    assert {item.decile for item in items} == set(range(1, 11))
    by_id = {item.sentence_id: item.text for item in items}
    assert by_id["sent-aa"] == _digit_free_text(0)


def test_build_sample_warns_and_shrinks_when_the_corpus_is_missing_a_sentence(
    con, tmp_path: Path
):
    scored, _ = _blind_fixture(100)
    scored.append(spotcheck.ScoredSentence(sentence_id="sent-not-in-corpus", novelty=99.0))
    warnings: list[str] = []
    items = spotcheck.build_sample(
        con, scored, per_decile=10, seed=0, echo=warnings.append
    )
    assert len(items) == 99
    assert any("not in the corpus" in w for w in warnings)


def test_build_sample_warns_when_a_decile_is_short(con):
    scored, _ = _blind_fixture(40)
    warnings: list[str] = []
    items = spotcheck.build_sample(con, scored, per_decile=10, seed=0, echo=warnings.append)
    assert len(items) == 40
    assert any("fewer than 10 sentences" in w for w in warnings)


def test_fetch_sentence_texts_on_empty_input(con):
    assert spotcheck.fetch_sentence_texts(con, []) == {}


def test_output_default_is_the_human_only_spotcheck_path():
    assert spotcheck.DEFAULT_OUT == Path("data/spotcheck/labels.jsonl")
