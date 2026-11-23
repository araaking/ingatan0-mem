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


# ---- usage: how often and how recently a fact was recalled ---------------------------------


def _used(count: int, idle_days: float | None) -> Any:
    last = (
        None
        if idle_days is None
        else (NOW - timedelta(days=idle_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    return {"use_count": count, "last_used": last}


def test_never_used_fact_is_neutral() -> None:
    assert core._usage_factor(_used(0, None), NOW) == 1.0
    # A repeat of `remember` sets `last_used` without a recall ever having returned the fact.
    assert core._usage_factor(_used(0, 0), NOW) == 1.0
    assert core._usage_factor(_used(3, None), NOW) == 1.0


def test_factor_is_bounded() -> None:
    hot = core._usage_factor(_used(10_000, 0), NOW)
    assert 1.0 < hot <= 1.0 + core.USAGE_BOOST
    assert hot == pytest.approx(1.0 + core.USAGE_BOOST)
    assert core._usage_factor(_used(1, 0), NOW) < core._usage_factor(_used(5, 0), NOW) < hot


def test_usage_boost_stays_below_freshness() -> None:
    # A new volatile fact must keep its lead over an older, much used one of equal relevance.
    assert core.USAGE_BOOST <= core.RECENCY_BOOST / 2


def test_usage_cannot_pass_a_neighbour_one_place_higher() -> None:
    # Under rank fusion the closest two different ranks can be among the first forty is 1%.
    assert core.USAGE_BOOST <= (1 / 100 - 1 / 101) / (1 / 101)
    assert core.USAGE_BOOST < (1 / 61 - 1 / 62) / (1 / 62)


def test_old_popularity_fades() -> None:
    assert core._usage_factor(_used(10_000, 365), NOW) < 1.001
    recent = core._usage_factor(_used(5, 1), NOW)
    assert recent > core._usage_factor(_used(5, core.USAGE_HALF_LIFE_DAYS), NOW) > 1.0
    undated: Any = {"use_count": 5, "last_used": "not a date"}
    assert core._usage_factor(undated, NOW) == 1.0


def _remember(ctx: Context, text: str, entity: str, **kw: Any) -> str:
    result = facts.remember(ctx, text, entities=[entity], kind="stable", **kw)
    assert result.fact_id is not None
    return result.fact_id


def _order(ctx: Context, query: str, **kw: Any) -> list[str]:
    return [hit.id for hit in search.recall(ctx, query, touch=False, **kw).facts]


def test_usage_breaks_a_tie_between_two_equal_matches(ctx: Context, frozen_clock: Any) -> None:
    first = _remember(ctx, "Jadwal antar barang dari toko-a tiap Senin pagi.", "toko-a")
    frozen_clock.advance(days=1)
    second = _remember(ctx, "Jadwal antar barang dari toko-b tiap Kamis pagi.", "toko-b")
    # Equal matches: the one written last comes first.
    assert _order(ctx, "jadwal antar barang") == [second, first]
    for _ in range(5):
        assert [hit.id for hit in search.recall(ctx, "senin").facts] == [first]
    frozen_clock.advance(days=7)
    assert _order(ctx, "jadwal antar barang") == [first, second]
    # Seen as of a past moment, today's usage says nothing.
    past = "2026-11-12T12:00:00Z"
    assert _order(ctx, "jadwal antar barang", as_of=past) == [second, first]


def test_a_much_used_fact_does_not_pass_the_fact_that_corrects_it(
    ctx: Context, frozen_clock: Any
) -> None:
    # Stable facts: nothing but the order of writing separates the two. Without a rule for
    # doubted facts the older, much recalled one would lead, and be recalled even more.
    old = _remember(ctx, "Ongkir ke Surabaya Rp 12.000 per kg.", "ongkir Surabaya")
    for _ in range(8):
        search.recall(ctx, "ongkir Surabaya")
    frozen_clock.advance(days=2)
    new = _remember(ctx, "Ongkir ke Surabaya Rp 14.000 per kg.", "ongkir Surabaya")
    hits = search.recall(ctx, "ongkir Surabaya", touch=False).facts
    assert [hit.id for hit in hits] == [new, old]
    assert {hit.status for hit in hits} == {"possible conflict"}


def test_off_topic_fact_is_not_lifted(ctx: Context, frozen_clock: Any) -> None:
    popular = _remember(ctx, "Jadwal libur karyawan diumumkan tiap Jumat.", "karyawan")
    for _ in range(20):
        search.recall(ctx, "libur karyawan")
    wanted = _remember(ctx, "Supplier gula pasir adalah toko-a.", "gula pasir")
    for n in range(8):
        _remember(ctx, f"Stok barang nomor {n} tinggal sedikit.", f"barang {n}")
    found = _order(ctx, "supplier gula pasir")
    assert found[0] == wanted
    assert popular not in found
