"""The scoring pass's as_of grid, which is where the time discipline is
easiest to get wrong by an hour rather than by a year.

`quarter_floor` reads calendar fields off a value DuckDB hands back in the
session timezone. Stamping the result UTC without converting first mixes two
calendars, and the resulting bug is invisible in any timezone west of UTC,
which is every timezone this was developed in.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "ticker_scripts_score_novelty",
    Path(__file__).resolve().parents[1] / "scripts" / "score_novelty.py",
)
score_novelty = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = score_novelty
_SPEC.loader.exec_module(score_novelty)

quarter_floor = score_novelty.quarter_floor

EASTERN = timezone(timedelta(hours=-5))
TOKYO = timezone(timedelta(hours=9))


@pytest.mark.parametrize(
    "moment,expected_month",
    [
        (datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc), 1),
        (datetime(2024, 2, 14, 12, 0, tzinfo=timezone.utc), 1),
        (datetime(2024, 3, 31, 23, 59, tzinfo=timezone.utc), 1),
        (datetime(2024, 4, 1, 0, 0, tzinfo=timezone.utc), 4),
        (datetime(2024, 12, 31, 23, 0, tzinfo=timezone.utc), 10),
    ],
)
def test_quarter_floor_lands_on_the_quarter_start(moment, expected_month):
    floor = quarter_floor(moment)
    assert (floor.year, floor.month, floor.day) == (2024, expected_month, 1)
    assert floor.tzinfo is timezone.utc


def test_quarter_floor_is_never_after_the_moment_it_floors():
    # The property the scoring pass depends on. If the floor lands after the
    # filing, the filing's own sentences satisfy filed_at < as_of, get fitted
    # into its own firm model, and are dropped from the output by the guard.
    for tz in (timezone.utc, EASTERN, TOKYO):
        for moment in (
            datetime(2020, 3, 31, 17, 0, tzinfo=tz),
            datetime(2020, 1, 1, 0, 30, tzinfo=tz),
            datetime(2025, 12, 31, 23, 59, tzinfo=tz),
        ):
            assert quarter_floor(moment) <= moment, (tz, moment)


def test_quarter_floor_reads_the_calendar_in_utc_not_the_session_zone():
    # 2020-03-31 21:00 UTC, which renders as 2020-04-01 in a UTC+9 session.
    # Reading .month off the local rendering would floor this to April, after
    # the filing itself.
    moment = datetime(2020, 4, 1, 6, 0, tzinfo=TOKYO)
    floor = quarter_floor(moment)
    assert (floor.year, floor.month) == (2020, 1)
    assert floor < moment


def test_quarter_floor_rejects_a_naive_datetime():
    # `.astimezone` on a naive value silently assumes system local time, which
    # is the same class of bug in a different disguise.
    with pytest.raises(ValueError, match="aware datetime"):
        quarter_floor(datetime(2024, 2, 14, 12, 0))
