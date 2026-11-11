from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ingatan import timeutil
from ingatan.errors import ValidationError


def test_now_is_aware_utc() -> None:
    now = timeutil.now()
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)


def test_iso_format_and_roundtrip() -> None:
    dt = datetime(2026, 11, 11, 9, 5, 7, 123456, tzinfo=timezone.utc)
    assert timeutil.iso(dt) == "2026-11-11T09:05:07Z"
    assert timeutil.parse("2026-11-11T09:05:07Z") == dt.replace(microsecond=0)


def test_iso_converts_other_zones_to_utc() -> None:
    wib = timezone(timedelta(hours=7))
    assert timeutil.iso(datetime(2026, 11, 11, 16, 0, 0, tzinfo=wib)) == "2026-11-11T09:00:00Z"


def test_parse_accepts_bare_date_and_normalize_gives_full_form() -> None:
    assert timeutil.parse("2026-12-01") == datetime(2026, 12, 1, tzinfo=timezone.utc)
    assert timeutil.normalize("2026-12-01") == "2026-12-01T00:00:00Z"
    assert timeutil.normalize("2026-12-01T10:00:00Z") == "2026-12-01T10:00:00Z"


def test_parse_rejects_garbage() -> None:
    with pytest.raises(ValidationError):
        timeutil.parse("minggu depan")


def test_set_now_freezes_and_none_unfreezes() -> None:
    timeutil.set_now(datetime(2030, 1, 2, 3, 4, 5))  # naive means UTC
    try:
        assert timeutil.iso(timeutil.now()) == "2030-01-02T03:04:05Z"
        assert timeutil.now() == timeutil.now()
    finally:
        timeutil.set_now(None)
    assert timeutil.now().year >= 2026


def test_frozen_clock_fixture_is_frozen_then_reset(frozen_clock: object) -> None:
    assert timeutil.iso(timeutil.now()) == "2026-11-11T09:00:00Z"


def test_clock_is_unfrozen_after_the_fixture_test() -> None:
    # Pytest runs tests in file order, so this follows the fixture test above.
    assert timeutil.iso(timeutil.now()) != "2026-11-11T09:00:00Z"
