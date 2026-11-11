from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from ingatan import episodes, facts, proposals
from ingatan.scope import Context

MakeContext = Callable[..., Context]

EPISODE_MARK = "zzepisodeteksunikzz"
OTHER_MARK = "zzepisodelainzz"


def raw_bytes(home: Path, name: str) -> bytes:
    blob = b""
    for suffix in ("", "-wal", "-shm"):
        path = home / f"{name}.db{suffix}"
        if path.exists():
            blob += path.read_bytes()
    return blob


def found(blob: bytes, marker: str) -> bool:
    return any(
        enc in blob
        for enc in (marker.encode("utf-8"), marker.encode("utf-16-le"), marker.encode("utf-16-be"))
    )


def episode_ids(ctx: Context) -> list[int]:
    return [r["id"] for r in ctx.business.execute("SELECT id FROM episodes ORDER BY id")]


def test_remember_records_episode_in_source(ctx: Context) -> None:
    episode = episodes.record(ctx, "Kopi susu sekarang Rp 18.000", "Dicatat.")
    result = facts.remember(ctx, "Harga kopi susu 18000", keywords=["kopi"], episode=episode)
    row = ctx.business.execute(
        "SELECT source FROM facts WHERE id = ?", (result.fact_id,)
    ).fetchone()
    assert json.loads(row["source"])["episode"] == episode


def test_forget_agent_fact_deletes_its_episode_bytes(ctx: Context, ingatan_home: Path) -> None:
    episode = episodes.record(ctx, f"Sandi gudang {EPISODE_MARK}", "Oke, saya ingat.")
    keep = episodes.record(ctx, f"Obrolan lain {OTHER_MARK}", "Siap.")
    result = facts.remember(ctx, "Ada kode gudang", keywords=["gudang"], episode=episode)
    assert result.fact_id is not None
    assert found(raw_bytes(ingatan_home, "tokoku"), EPISODE_MARK)  # the test guards itself

    facts.forget(ctx, result.fact_id)

    assert episode_ids(ctx) == [keep]
    blob = raw_bytes(ingatan_home, "tokoku")
    assert not found(blob, EPISODE_MARK)
    assert found(blob, OTHER_MARK)
    ctx.close()
    assert not found(raw_bytes(ingatan_home, "tokoku"), EPISODE_MARK)


def test_forget_business_fact_by_user_deletes_the_episode_of_any_agent(
    make_ctx: MakeContext, ingatan_home: Path
) -> None:
    writer = make_ctx(agent="hermes")
    admin = make_ctx(agent="claude-code")
    episode = episodes.record(writer, f"Pelanggan rahasia {EPISODE_MARK}", "Dicatat.")
    result = facts.remember(
        writer, "Pelanggan VIP ada", keywords=["pelanggan"], tier="business", episode=episode
    )
    assert result.proposal_id is not None
    fact_id = proposals.approve(admin, result.proposal_id)
    assert found(raw_bytes(ingatan_home, "tokoku"), EPISODE_MARK)

    facts.forget(admin, fact_id, by_user=True)

    assert episode_ids(admin) == []
    assert not found(raw_bytes(ingatan_home, "tokoku"), EPISODE_MARK)
    admin.close()
    writer.close()
    assert not found(raw_bytes(ingatan_home, "tokoku"), EPISODE_MARK)


def test_forget_global_fact_by_user_deletes_episode_in_business_file(
    ctx: Context, ingatan_home: Path
) -> None:
    episode = episodes.record(ctx, f"Aturan umum {EPISODE_MARK}", "Dicatat.")
    result = facts.remember(
        ctx, "Aturan umum berlaku", keywords=["aturan"], tier="global", episode=episode
    )
    assert result.proposal_id is not None
    fact_id = proposals.approve(ctx, result.proposal_id)
    assert found(raw_bytes(ingatan_home, "tokoku"), EPISODE_MARK)

    facts.forget(ctx, fact_id, by_user=True)

    assert episode_ids(ctx) == []
    assert not found(raw_bytes(ingatan_home, "tokoku"), EPISODE_MARK)
    assert not found(raw_bytes(ingatan_home, "global"), EPISODE_MARK)


def test_forget_leaves_other_agents_episode_when_not_by_user(
    make_ctx: MakeContext,
) -> None:
    ctx = make_ctx()
    other = make_ctx(agent="hermes")
    foreign = episodes.record(other, "Milik hermes", "Ya.")
    result = facts.remember(ctx, "Fakta saya", keywords=["fakta"])
    assert result.fact_id is not None
    # Forge a link to an episode this agent does not own.
    ctx.business.execute(
        "UPDATE facts SET source = json_set(source, '$.episode', ?) WHERE id = ?",
        (foreign, result.fact_id),
    )
    facts.forget(ctx, result.fact_id)
    assert episode_ids(ctx) == [foreign]


def test_forget_without_episode_still_works(ctx: Context) -> None:
    episodes.record(ctx, "Obrolan", "Balasan")
    result = facts.remember(ctx, "Fakta tanpa episode", keywords=["fakta"])
    assert result.fact_id is not None
    facts.forget(ctx, result.fact_id)
    assert len(episode_ids(ctx)) == 1
