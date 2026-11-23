"""The local embedding provider: catalogue, download, tokenizing, pooling and unloading.

Everything here runs offline. The download is exercised against a server on loopback, and
the model against small stand-ins; `test_real_model.py` covers the real files when present.
"""

from __future__ import annotations

import hashlib
import http.server
import sys
import threading
import time
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ingatan.errors import IngatanError
from ingatan.scope import EmbeddingConfig
from ingatan.search import embed, local


@pytest.fixture(autouse=True)
def _no_model_left_loaded() -> Iterator[None]:
    yield
    embed.unload_local_models()


# ---- catalogue -----------------------------------------------------------------------


def test_catalogue_default_is_the_quantized_model_and_names_ignore_case() -> None:
    assert local.resolve("").name == local.DEFAULT_LOCAL_MODEL == "xenova/multilingual-e5-small"
    assert local.resolve("  Xenova/Multilingual-E5-Small ").name == "xenova/multilingual-e5-small"
    full = local.resolve("intfloat/multilingual-e5-small")
    assert full.dims == 384 and full.max_tokens == 512
    # Both models share one tokenizer file; only the weights differ.
    assert full.tokenizer == local.resolve("").tokenizer
    assert full.weights.size > local.resolve("").weights.size


def test_unknown_local_model_names_the_supported_ones_and_the_way_out() -> None:
    with pytest.raises(IngatanError) as caught:
        local.resolve("some/other-model")
    message = caught.value.message
    assert "xenova/multilingual-e5-small" in message
    assert "ollama" in message and "openai" in message


def test_every_catalogue_file_is_pinned_to_a_revision_and_checksum() -> None:
    for model in local.LOCAL_MODELS.values():
        for spec in model.files:
            assert len(spec.revision) == 40 and all(c in "0123456789abcdef" for c in spec.revision)
            assert len(spec.sha256) == 64
            assert spec.size > 0
            url = spec.url("https://example.invalid/")
            assert url == f"https://example.invalid/{spec.repo}/resolve/{spec.revision}/{spec.path}"


def test_model_folder_defaults_to_the_data_folder(ingatan_home: Path, tmp_path: Path) -> None:
    model = local.resolve("")
    assert (
        local.model_folder("", model) == ingatan_home / "models" / "xenova--multilingual-e5-small"
    )
    assert local.model_folder(tmp_path, model) == tmp_path / "xenova--multilingual-e5-small"
    assert len(local.missing_files(model, tmp_path)) == 2


# ---- download ------------------------------------------------------------------------


def _tiny_model(payloads: dict[str, bytes]) -> local.LocalModel:
    def spec(name: str, path: str) -> local.ModelFile:
        data = payloads[path]
        return local.ModelFile(
            name=name,
            repo="org/tiny",
            revision="0" * 40,
            path=path,
            sha256=hashlib.sha256(data).hexdigest(),
            size=len(data),
        )

    return local.LocalModel(
        name="org/tiny",
        dims=3,
        max_tokens=8,
        weights=spec("model.onnx", "onnx/model.onnx"),
        tokenizer=spec("sentencepiece.bpe.model", "sentencepiece.bpe.model"),
        summary="test stand-in",
    )


@pytest.fixture
def mirror(tmp_path: Path) -> Iterator[tuple[str, dict[str, bytes], list[str]]]:
    """A loopback HTTP server laid out like the model host. Yields (endpoint, files, hits)."""
    payloads = {"onnx/model.onnx": b"weights" * 4000, "sentencepiece.bpe.model": b"pieces" * 300}
    root = tmp_path / "mirror" / "org" / "tiny" / "resolve" / ("0" * 40)
    for path, data in payloads.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    hits: list[str] = []

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(tmp_path / "mirror"), **kwargs)

        def do_GET(self) -> None:
            hits.append(self.path)
            super().do_GET()

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", payloads, hits
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_pull_downloads_verifies_and_skips_what_is_already_there(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mirror: tuple[str, dict[str, bytes], list[str]],
) -> None:
    endpoint, payloads, hits = mirror
    model = _tiny_model(payloads)
    monkeypatch.setitem(local.LOCAL_MODELS, model.name, model)
    seen: list[tuple[str, int]] = []

    fetched = local.pull(
        model.name,
        tmp_path / "models",
        endpoint=endpoint,
        progress=lambda spec, received: seen.append((spec.name, received)),
    )
    folder = tmp_path / "models" / "org--tiny"
    assert [p.name for p in fetched] == ["model.onnx", "sentencepiece.bpe.model"]
    assert (folder / "model.onnx").read_bytes() == payloads["onnx/model.onnx"]
    assert not list(folder.glob("*.part"))
    assert local.missing_files(model, tmp_path / "models") == []
    assert seen[-1] == ("sentencepiece.bpe.model", len(payloads["sentencepiece.bpe.model"]))

    # A second pull hashes the files, finds them correct, and asks the server for nothing.
    before = len(hits)
    assert local.pull(model.name, tmp_path / "models", endpoint=endpoint) == []
    assert len(hits) == before

    # The endpoint can also come from the HF_ENDPOINT environment variable (a mirror).
    (folder / "model.onnx").write_bytes(b"corrupted")
    monkeypatch.setenv("HF_ENDPOINT", endpoint)
    assert [p.name for p in local.pull(model.name, tmp_path / "models")] == ["model.onnx"]
    assert (folder / "model.onnx").read_bytes() == payloads["onnx/model.onnx"]


def test_pull_rejects_a_file_that_does_not_match_its_checksum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mirror: tuple[str, dict[str, bytes], list[str]],
) -> None:
    endpoint, payloads, _ = mirror
    size = len(payloads["onnx/model.onnx"])
    folder = tmp_path / "models" / "org--tiny"
    # Pinned to content the server does not have: the same length, and a longer file.
    for pinned in (bytes(size), bytes(size + 10)):
        model = _tiny_model({**payloads, "onnx/model.onnx": pinned})
        monkeypatch.setitem(local.LOCAL_MODELS, model.name, model)
        with pytest.raises(IngatanError, match="pinned checksum"):
            local.pull(model.name, tmp_path / "models", endpoint=endpoint)
        assert not (folder / "model.onnx").exists()
        assert not list(folder.glob("*.part"))


def test_pull_stops_a_download_that_is_larger_than_the_pinned_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mirror: tuple[str, dict[str, bytes], list[str]],
) -> None:
    endpoint, payloads, _ = mirror
    model = _tiny_model({**payloads, "onnx/model.onnx": b"much smaller than served"})
    monkeypatch.setitem(local.LOCAL_MODELS, model.name, model)
    monkeypatch.setattr(local, "_DOWNLOAD_CHUNK", 64)
    received: list[int] = []

    with pytest.raises(IngatanError, match="more than the expected 24 bytes"):
        local.pull(
            model.name,
            tmp_path / "models",
            endpoint=endpoint,
            progress=lambda _spec, count: received.append(count),
        )
    # It stopped at the first chunk past the size, long before the 28,000 bytes on offer.
    assert received == []
    folder = tmp_path / "models" / "org--tiny"
    assert not (folder / "model.onnx").exists()
    assert not list(folder.glob("*.part"))


def test_pull_reports_an_unreachable_host_and_refuses_other_schemes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _tiny_model({"onnx/model.onnx": b"a", "sentencepiece.bpe.model": b"b"})
    monkeypatch.setitem(local.LOCAL_MODELS, model.name, model)
    with pytest.raises(IngatanError, match="Could not download"):
        local.pull(model.name, tmp_path / "models", endpoint="http://127.0.0.1:9", timeout=2)
    with pytest.raises(IngatanError, match="not an http"):
        local.pull(model.name, tmp_path / "models", endpoint="file:///etc")
    # A mistyped mirror address is a clean message too, never a traceback.
    for broken in ("http://127.0.0.1:notaport", "http://[::1"):
        with pytest.raises(IngatanError, match="Could not download"):
            local.pull(model.name, tmp_path / "models", endpoint=broken, timeout=2)
    assert not list((tmp_path / "models").rglob("*.part"))


# ---- tokenizing and pooling ----------------------------------------------------------


def test_token_ids_shift_pieces_wrap_them_and_truncate() -> None:
    # SentencePiece piece 0 is <unk>; every other piece sits one below its XLM-R id.
    assert local._token_ids([5, 0, 9], 512) == [0, 6, 3, 10, 2]
    assert local._token_ids([], 512) == [0, 2]
    long = local._token_ids(list(range(1, 1000)), 512)
    assert len(long) == 512 and long[0] == 0 and long[-1] == 2


class _FakeTokenizer:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def encode(self, text: str, out_type: type = int) -> list[int]:
        self.seen.append(text)
        return [len(word) for word in text.split()]


class _FakeSession:
    """Returns each token's id as a constant 3-dim hidden state, so the pooled vector is known."""

    def __init__(self) -> None:
        self.batches: list[tuple[int, int]] = []

    def get_inputs(self) -> list[Any]:
        return [
            types.SimpleNamespace(name="input_ids"),
            types.SimpleNamespace(name="attention_mask"),
        ]

    def run(self, _outputs: Any, feed: dict[str, Any]) -> list[Any]:
        np = pytest.importorskip("numpy")
        assert set(feed) == {"input_ids", "attention_mask"}  # token_type_ids is not an input
        ids = feed["input_ids"]
        assert feed["attention_mask"].all()  # no padding: every position is a real token
        self.batches.append(tuple(ids.shape))
        hidden = np.repeat(ids[..., None].astype(np.float32), 3, axis=2)
        hidden[..., 1] = 1.0
        hidden[..., 2] = 0.0
        return [hidden]


def test_encode_runs_one_text_per_pass_and_mean_pools_it() -> None:
    np = pytest.importorskip("numpy")
    tokenizer, session = _FakeTokenizer(), _FakeSession()
    loaded = local._Loaded(local.resolve(""), session, tokenizer)
    texts = ["a", "bb ccc dddd eeeee ffffff", "  gg hh  ", "i jj"]

    vectors = loaded.encode(texts)

    assert tokenizer.seen[2] == "gg hh"  # surrounding whitespace never reaches the tokenizer
    assert len(vectors) == len(texts)
    for vec in vectors:
        assert len(vec) == 3
        assert abs(float(np.linalg.norm(np.asarray(vec))) - 1.0) < 1e-5
    # One forward pass per text, each exactly as long as that text. A text's vector can
    # therefore never depend on what it was embedded together with.
    assert session.batches == [(1, 3), (1, 7), (1, 4), (1, 4)]
    assert loaded.encode(["i jj", "a"])[1] == vectors[0]
    # "a" -> ids [0, 2, 2]: mean id 4/3, mean of the constant column 1.
    expected = np.asarray([4 / 3, 1.0, 0.0])
    assert np.allclose(np.asarray(vectors[0]), expected / np.linalg.norm(expected), atol=1e-5)


# ---- loading, failing gracefully, unloading -------------------------------------------


def test_missing_packages_or_files_never_break_remember_or_recall(
    ingatan_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = EmbeddingConfig(provider="local")
    monkeypatch.setattr(local, "missing_packages", lambda: ["onnxruntime", "sentencepiece"])
    assert embed.embed_text(cfg, "harga kopi", is_query=True) is None
    assert local.INSTALL_HINT in embed.last_embed_error()
    assert embed.local_runtime_available() is False

    monkeypatch.setattr(local, "missing_packages", lambda: [])
    assert embed.embed_text(cfg, "harga kopi", is_query=True) is None
    assert "ingatan embeddings pull" in embed.last_embed_error()
    assert embed.embed_batch(cfg, ["satu", "", "tiga"]) == [None, None, None]

    assert embed.embed_text(EmbeddingConfig(provider="local", model="nope"), "x") is None
    assert "Unknown local embedding model" in embed.last_embed_error()


def _fake_loader(calls: list[tuple[str, str, int]]) -> Any:
    def load(model: local.LocalModel, folder: Path, threads: int) -> Any:
        calls.append((model.name, str(folder), threads))
        return types.SimpleNamespace(encode=lambda texts: [[1.0, 0.0, 0.0] for _ in texts])

    return load


def test_the_model_is_loaded_once_and_dropped_when_idle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str, int]] = []
    monkeypatch.setattr(local, "_load", _fake_loader(calls))

    def embed_once(threads: int = 2, idle: int = 300) -> list[list[float]]:
        return local.embed(
            ["satu", "dua"], model_dir=tmp_path, threads=threads, idle_unload_seconds=idle
        )

    assert embed_once() == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    embed_once()
    assert len(calls) == 1
    # One idle timer serves any number of calls: no thread is started per embedding.
    first_timer = local._timer
    assert first_timer is not None
    embed_once()
    assert local._timer is first_timer

    # A different thread count is a different session: the old one is replaced.
    embed_once(threads=1)
    assert len(calls) == 2 and calls[-1][2] == 1

    # Not idle yet: the timer re-arms for the time that remains and keeps the model.
    local._on_idle()
    assert local._loaded is not None and local._timer is not None
    # Idle for longer than the limit: dropped.
    monkeypatch.setattr(local, "_last_used", time.monotonic() - 1000)
    local._on_idle()
    assert local._loaded is None
    assert local.unload() == 0

    embed_once(idle=0)
    assert len(calls) == 3
    assert local._timer is None  # 0 means "keep it loaded": no timer at all
    assert local.unload() == 1
    assert local._loaded is None
    assert local.embed([], model_dir=tmp_path) == []
    assert len(calls) == 3


def test_a_changed_idle_limit_replaces_the_timer_and_zero_keeps_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(local, "_load", _fake_loader([]))

    def embed_once(idle: int) -> None:
        local.embed(["satu"], model_dir=tmp_path, idle_unload_seconds=idle)

    embed_once(300)
    first = local._timer
    assert first is not None and local._idle_limit == 300
    # A timer armed for the old limit would fire at the wrong time: it is replaced.
    embed_once(60)
    second = local._timer
    assert second is not None and second is not first and local._idle_limit == 60
    assert first.finished.is_set()  # cancelled
    assert second.interval == 60

    # A timer that was already running when the limit became 0 must not drop the model.
    embed_once(0)
    assert local._timer is None and second.finished.is_set()
    monkeypatch.setattr(local, "_last_used", time.monotonic() - 1000)
    local._on_idle()
    assert local._loaded is not None and local._timer is None

    # The limit that counts is the current one, not the one the timer was started with.
    embed_once(300)
    monkeypatch.setattr(local, "_idle_limit", 5000)
    monkeypatch.setattr(local, "_last_used", time.monotonic() - 1000)
    local._on_idle()
    assert local._loaded is not None and local._timer is not None
    assert 3900 < local._timer.interval <= 4000


def test_numpy_is_imported_on_first_use_not_with_the_package() -> None:
    import subprocess

    code = (
        "import sys; import ingatan.search, ingatan.facts, ingatan.mcp.server; "
        "heavy = [m for m in ('numpy', 'onnxruntime', 'sentencepiece') if m in sys.modules]; "
        "print(heavy)"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]"


def test_limit_math_threads_sets_a_default_and_respects_a_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    # Set first, so monkeypatch records the original state and restores it afterwards
    # (the function under test writes to os.environ behind monkeypatch's back).
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "placeholder")
    monkeypatch.delitem(sys.modules, "numpy", raising=False)
    monkeypatch.delenv("OPENBLAS_NUM_THREADS")
    local.limit_math_threads()
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    local.limit_math_threads()
    assert os.environ["OPENBLAS_NUM_THREADS"] == "8"
    # Once NumPy is loaded the setting can no longer take effect, so it is left alone.
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))
    local.limit_math_threads()
    assert "OPENBLAS_NUM_THREADS" not in os.environ


# ---- scoring without NumPy -----------------------------------------------------------


def test_scoring_gives_the_same_answer_without_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remote providers work with no NumPy at all: the plain-Python path must agree."""
    import math

    query = embed.unpack_vector(embed.pack_vector([0.6, 0.8, 0.0]), 3)
    items = [
        ("x", embed.pack_vector([1.0, 0.0, 0.0])),  # cosine 0.6
        ("same", embed.pack_vector([3.0, 4.0, 0.0])),  # cosine 1.0
        ("z", embed.pack_vector([0.0, 0.0, 1.0])),  # cosine 0.0
        ("y", embed.pack_vector([0.0, 1.0, 0.0])),  # cosine 0.8
        ("broken", b"\x00" * 8),  # the wrong length: skipped, never an error
    ]

    def ranked(**kwargs: Any) -> list[tuple[str, float]]:
        found = embed.score_blobs(query, items, 3, **kwargs)
        return [(key, round(score, 5)) for key, score in found]

    expected = [("same", 1.0), ("y", 0.8), ("x", 0.6)]
    results = []
    if local.load_numpy() is not None:
        results.append(ranked(min_similarity=0.5))
    monkeypatch.setattr(local, "load_numpy", lambda: None)
    results.append(ranked(min_similarity=0.5))
    monkeypatch.delattr(math, "sumprod", raising=False)  # Python 3.10 and 3.11 have none
    results.append(ranked(min_similarity=0.5))
    assert all(result == expected for result in results), results

    assert ranked(min_similarity=0.5, top_k=1) == [("same", 1.0)]
    assert ranked(min_similarity=-2.0, top_k=10)[-1] == ("z", 0.0)
    assert ranked(min_similarity=1.5) == []
    assert embed.cosine_similarity(query, items[3][1], 3) == pytest.approx(0.8, abs=1e-6)
    assert embed.cosine_similarity(query, items[4][1], 3) == 0.0


# ---- the command line ----------------------------------------------------------------


def test_cli_pull_downloads_once_and_says_how_to_switch_it_on(
    ingatan_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    mirror: tuple[str, dict[str, bytes], list[str]],
) -> None:
    from typer.testing import CliRunner

    from ingatan.cli import app

    endpoint, payloads, _ = mirror
    model = _tiny_model(payloads)
    monkeypatch.setitem(local.LOCAL_MODELS, model.name, model)
    monkeypatch.setenv("HF_ENDPOINT", endpoint)
    runner = CliRunner()

    first = runner.invoke(app, ["embeddings", "pull", "--model", model.name])
    assert first.exit_code == 0, first.output
    assert "Downloaded 2 file(s)" in first.stdout and "Checksums verified" in first.stdout
    assert 'provider = "local"' in first.stdout and f'model = "{model.name}"' in first.stdout
    assert "ingatan reindex" in first.stdout
    assert (ingatan_home / "models" / "org--tiny" / "model.onnx").is_file()

    again = runner.invoke(app, ["embeddings", "pull", "--model", model.name])
    assert again.exit_code == 0
    assert "Nothing was downloaded" in again.stdout

    # With the model already configured, the hint to edit the config is not repeated.
    monkeypatch.setenv("INGATAN_EMBED_PROVIDER", "local")
    monkeypatch.setenv("INGATAN_EMBED_MODEL", model.name)
    configured = runner.invoke(app, ["embeddings", "pull"])
    assert configured.exit_code == 0
    assert "provider =" not in configured.stdout

    unknown = runner.invoke(app, ["embeddings", "pull", "--model", "no/such-model"])
    assert unknown.exit_code == 1
    assert "Unknown local embedding model" in unknown.output
