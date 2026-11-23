"""Ranking details of `recall`: freshness from a fact's age, and the weak-tail filter."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from ingatan import facts, search
from ingatan.scope import Context
from ingatan.search import core

MakeContext = Callable[..., Context]
NOW = datetime(2026, 11, 11, 9, 0, 0, tzinfo=timezone.utc)


def _row(kind: str, age_days: float) -> Any:
    written = NOW - timedelta(days=age_days)
    return {"kind": kind, "valid_from": written.strftime("%Y-%m-%dT%H:%M:%SZ")}


def test_freshness_comes_from_age_is_bounded_and_leaves_stable_facts_alone() -> None:
    half = core.RECENCY_BOOST / 2
    assert core._freshness_factor(_row("volatile", 0), NOW) == pytest.approx(1 + half)
    assert core._freshness_factor(_row("volatile", core.RECENCY_HALF_LIFE_DAYS), NOW) == (
        pytest.approx(1.0)
    )
    assert core._freshness_factor(_row("volatile", 5000), NOW) == pytest.approx(1 - half)
    # Two old facts written a day apart are, for all purposes, equally stale.
    gap = core._freshness_factor(_row("volatile", 700), NOW) - core._freshness_factor(
        _row("volatile", 701), NOW
    )
    assert 0 <= gap < 1e-6
    # A stable fact does not age; a fact dated in the future is simply new.
    assert core._freshness_factor(_row("stable", 0), NOW) == 1.0
    assert core._freshness_factor(_row("stable", 5000), NOW) == 1.0
    assert core._freshness_factor(_row("volatile", -3), NOW) == pytest.approx(1 + half)
    undated: Any = {"kind": "volatile", "valid_from": "not a date"}
    assert core._freshness_factor(undated, NOW) == 1.0
    # The whole effect is smaller than the gap between two ranks three places apart.
    assert core.RECENCY_BOOST < (1 / 61 - 1 / 64) / (1 / 61)


def test_equal_matches_order_by_freshness_and_a_better_match_still_wins(
    ctx: Context, frozen_clock: Any
) -> None:
    def save(place: str, kind: str) -> str:
        saved = facts.remember(
            ctx,
            f"Harga kopi cabang {place} dua puluh ribu rupiah.",
            keywords=["harga"],
            kind=kind,
        )
        assert saved.fact_id
        return saved.fact_id

    old_volatile = save("barat", "volatile")
    old_exact = facts.remember(
        ctx,
        "Harga kopi cabang selatan khusus pelanggan grosir.",
        keywords=["harga", "grosir"],
        kind="volatile",
    ).fact_id
    frozen_clock.advance(days=400)
    stable = save("timur", "stable")
    fresh_volatile = save("utara", "volatile")

    # Three facts match the query equally. Fresh volatile first, the stable one in the
    # middle (it does not age), the 400-day-old volatile one last.
    order = [h.id for h in search.recall(ctx, "harga kopi cabang rupiah", touch=False).facts]
    assert [i for i in order if i in (fresh_volatile, stable, old_volatile)] == [
        fresh_volatile,
        stable,
        old_volatile,
    ]
    # Freshness never beats relevance: the old fact that matches every word stays on top.
    hits = search.recall(ctx, "harga kopi cabang selatan pelanggan grosir", touch=False).facts
    assert hits[0].id == old_exact


def test_a_weak_tail_hit_is_judged_against_its_own_file(make_ctx: MakeContext) -> None:
    ctx = make_ctx()
    strong = facts.remember(
        ctx,
        "Jadwal libur toko jatuh pada hari Minggu.",
        keywords=["jadwal", "libur"],
        tier="business",
        kind="stable",
    )
    for n in range(10):
        facts.remember(
            ctx, f"Catatan gudang nomor {n} soal rak.", keywords=["gudang"], kind="stable"
        )
    weak_business = facts.remember(
        ctx,
        "Kopi robusta dipesan dari pemasok lama setiap awal bulan untuk semua cabang toko.",
        keywords=["robusta"],
        tier="business",
        kind="stable",
    )
    only_global = facts.remember(
        ctx, "Budi suka kopi tubruk.", keywords=["minuman"], tier="global", kind="stable"
    )

    found = [h.id for h in search.recall(ctx, "jadwal libur kopi", limit=10, touch=False).facts]
    assert found[0] == strong.fact_id
    # The global file holds one fact, so its BM25 numbers are tiny next to the business
    # file's. It is still the best match that file has, and it is kept.
    assert only_global.fact_id in found
    # Inside one file the comparison is fair: a one-word match far below the best is dropped.
    assert weak_business.fact_id not in found
