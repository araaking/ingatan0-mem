"""The built-in local embedding models: ONNX Runtime plus SentencePiece, and nothing else.

Every model listed here uses the XLM-RoBERTa vocabulary, so one 5 MB SentencePiece file
tokenizes for all of them. That choice is about memory: the same vocabulary loaded through
the Hugging Face `tokenizers` library costs about 250 MB of resident memory, more than the
quantized model itself (see `docs/RAM.md`).

Nothing in this module touches the network except `pull`, which runs only when the user
asks for it (`ingatan embeddings pull`). A missing package or model file makes `embed` raise
an `ImportError` or `OSError`; the caller turns that into "no vector" and recall carries on
with FTS5 alone.
"""

from __future__ import annotations

import contextlib
import ctypes
import gc
import hashlib
import http.client
import importlib.util
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ingatan.errors import IngatanError
from ingatan.scope import data_home

__all__ = [
    "DEFAULT_ENDPOINT",
    "DEFAULT_LOCAL_MODEL",
    "INSTALL_HINT",
    "LOCAL_MODELS",
    "RUNTIME_PACKAGES",
    "LocalModel",
    "ModelFile",
    "default_model_dir",
    "embed",
    "limit_math_threads",
    "load_numpy",
    "missing_files",
    "missing_packages",
    "model_folder",
    "pull",
    "resolve",
    "unload",
]

DEFAULT_LOCAL_MODEL = "xenova/multilingual-e5-small"
DEFAULT_ENDPOINT = "https://huggingface.co"
RUNTIME_PACKAGES = ("numpy", "onnxruntime", "sentencepiece")
# What to type to get them. It names the packages, so it works for an installation from
# a checkout as well as for one from an index (where it is the `vector` extra).
INSTALL_HINT = "pip install numpy onnxruntime sentencepiece"
# Token ids of the XLM-RoBERTa vocabulary. SentencePiece numbers its pieces one lower and
# keeps `<unk>` at 0, hence the shift in `_token_ids`.
_BOS, _EOS, _UNK = 0, 2, 3
_DOWNLOAD_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class ModelFile:
    """One file of a model: where it comes from and what it must hash to."""

    name: str  # file name inside the local model folder
    repo: str
    revision: str
    path: str  # path inside the repository
    sha256: str
    size: int

    def url(self, endpoint: str) -> str:
        return f"{endpoint.rstrip('/')}/{self.repo}/resolve/{self.revision}/{self.path}"


@dataclass(frozen=True)
class LocalModel:
    name: str
    dims: int
    max_tokens: int
    weights: ModelFile
    tokenizer: ModelFile
    summary: str

    @property
    def files(self) -> tuple[ModelFile, ModelFile]:
        return (self.weights, self.tokenizer)


_E5_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
_XLMR_TOKENIZER = ModelFile(
    name="sentencepiece.bpe.model",
    repo="intfloat/multilingual-e5-small",
    revision=_E5_REVISION,
    path="sentencepiece.bpe.model",
    sha256="cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
    size=5_069_051,
)

LOCAL_MODELS: dict[str, LocalModel] = {
    "xenova/multilingual-e5-small": LocalModel(
        name="xenova/multilingual-e5-small",
        dims=384,
        max_tokens=512,
        weights=ModelFile(
            name="model.onnx",
            repo="Xenova/multilingual-e5-small",
            revision="761b726dd34fb83930e26aab4e9ac3899aa1fa78",
            path="onnx/model_quantized.onnx",
            sha256="f80102d3f2a1229f387d3c81909990d8945513e347b0eab049f7de3c6f98c193",
            size=118_308_185,
        ),
        tokenizer=_XLMR_TOKENIZER,
        summary="multilingual-e5-small, int8 quantized, 100+ languages, 384 dims",
    ),
    "intfloat/multilingual-e5-small": LocalModel(
        name="intfloat/multilingual-e5-small",
        dims=384,
        max_tokens=512,
        weights=ModelFile(
            name="model.onnx",
            repo="intfloat/multilingual-e5-small",
            revision=_E5_REVISION,
            path="onnx/model.onnx",
            sha256="ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665",
            size=470_268_510,
        ),
        tokenizer=_XLMR_TOKENIZER,
        summary="multilingual-e5-small, full float32 weights, 100+ languages, 384 dims",
    ),
}


def resolve(name: str) -> LocalModel:
    """The catalogue entry for `name` (case-insensitive; empty means the default)."""
    key = name.strip().casefold() or DEFAULT_LOCAL_MODEL
    found = LOCAL_MODELS.get(key)
    if found is None:
        raise IngatanError(
            f"Unknown local embedding model {name.strip()!r}. The local provider supports: "
            f"{', '.join(LOCAL_MODELS)}. For any other model use provider 'ollama' or "
            "'openai' (any OpenAI-compatible endpoint)."
        )
    return found


def default_model_dir() -> Path:
    """`<INGATAN_HOME>/models`."""
    return data_home() / "models"


def model_folder(model_dir: Path | str, model: LocalModel) -> Path:
    base = Path(model_dir) if str(model_dir).strip() else default_model_dir()
    return base / model.name.replace("/", "--")


def missing_packages() -> list[str]:
    """Names of the optional runtime packages (the `vector` extra) that are not installed."""
    return [name for name in RUNTIME_PACKAGES if importlib.util.find_spec(name) is None]


def missing_files(model: LocalModel, model_dir: Path | str) -> list[Path]:
    """Model files that are absent or have the wrong size. Cheap: no hashing."""
    folder = model_folder(model_dir, model)
    missing: list[Path] = []
    for spec in model.files:
        path = folder / spec.name
        try:
            ok = path.is_file() and path.stat().st_size == spec.size
        except OSError:
            ok = False
        if not ok:
            missing.append(path)
    return missing


def limit_math_threads() -> None:
    """Keep NumPy's BLAS to one thread, unless the user chose otherwise.

    OpenBLAS reserves a buffer per thread when NumPy is imported. On a 16-core Windows
    machine that is about 490 MB of committed memory for a library Ingatan only uses for
    one small matrix-vector product per query. Call this from a process entry point, before
    anything imports NumPy; it changes nothing once NumPy is loaded.
    """
    if "numpy" not in sys.modules:
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def load_numpy() -> Any | None:
    """NumPy if it is installed, else None. Imported on first use, never at start-up."""
    found = sys.modules.get("numpy")
    if found is not None:
        return found
    try:
        import numpy  # type: ignore[import-not-found]
    except ImportError:
        return None
    return numpy


# ------------------------------------------------------------------------------ download


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_DOWNLOAD_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _download(
    spec: ModelFile,
    url: str,
    target: Path,
    timeout: float,
    progress: Callable[[ModelFile, int], None] | None,
) -> None:
    if not url.startswith(("https://", "http://")):
        raise IngatanError(f"Refusing to download from {url!r}: not an http(s) address.")
    part = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    received = 0
    try:
        # A malformed address raises ValueError or http.client.InvalidURL, a connection cut
        # short raises IncompleteRead: all of them end as one clean message, with no
        # partial file left behind.
        request = urllib.request.Request(url, headers={"User-Agent": "ingatan"})
        with urllib.request.urlopen(request, timeout=timeout) as response, part.open("wb") as out:
            while chunk := response.read(_DOWNLOAD_CHUNK):
                received += len(chunk)
                if received > spec.size:
                    # Stop before a wrong mirror fills the disk; the checksum would fail anyway.
                    raise ValueError(f"more than the expected {spec.size} bytes were sent")
                out.write(chunk)
                digest.update(chunk)
                if progress is not None:
                    progress(spec, received)
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as exc:
        with contextlib.suppress(OSError):
            part.unlink()
        raise IngatanError(
            f"Could not download {spec.path} from {url}: {type(exc).__name__}: {exc}"
        ) from exc
    if received != spec.size or digest.hexdigest() != spec.sha256:
        with contextlib.suppress(OSError):
            part.unlink()
        raise IngatanError(
            f"{spec.path} from {spec.repo} did not match its pinned checksum "
            f"({received} bytes received, {spec.size} expected). Nothing was installed."
        )
    try:
        os.replace(part, target)
    except OSError as exc:
        with contextlib.suppress(OSError):
            part.unlink()
        raise IngatanError(f"Could not write {target}: {exc}") from exc


def _is_verified(path: Path, spec: ModelFile) -> bool:
    """True when `path` holds exactly the pinned file (size first, then the hash)."""
    try:
        return path.is_file() and path.stat().st_size == spec.size and _sha256(path) == spec.sha256
    except OSError:
        return False


def pull(
    model_name: str,
    model_dir: Path | str,
    *,
    endpoint: str | None = None,
    timeout: float = 60.0,
    progress: Callable[[ModelFile, int], None] | None = None,
) -> list[Path]:
    """Download the files of one local model and verify them against pinned SHA-256 sums.

    A file that is already present and correct is left alone, so running this twice is
    cheap. Returns the paths that were downloaded. `endpoint` (or the `HF_ENDPOINT`
    environment variable) points at a mirror of the default host.
    """
    model = resolve(model_name)
    base = endpoint or (os.environ.get("HF_ENDPOINT") or "").strip() or DEFAULT_ENDPOINT
    folder = model_folder(model_dir, model)
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise IngatanError(f"Cannot create the model folder {folder}: {exc}") from exc
    fetched: list[Path] = []
    for spec in model.files:
        target = folder / spec.name
        if _is_verified(target, spec):
            continue
        _download(spec, spec.url(base), target, timeout, progress)
        fetched.append(target)
    return fetched


# ------------------------------------------------------------------------------- runtime


def _token_ids(pieces: Sequence[int], max_tokens: int) -> list[int]:
    """SentencePiece piece ids as XLM-RoBERTa input ids: `<s>` pieces... `</s>`, truncated."""
    body = [piece + 1 if piece else _UNK for piece in pieces]
    return [_BOS, *body[: max_tokens - 2], _EOS]


class _Loaded:
    """One model held in memory: the ONNX session and its tokenizer."""

    def __init__(self, model: LocalModel, session: Any, tokenizer: Any) -> None:
        self.model = model
        self.session = session
        self.tokenizer = tokenizer
        self.inputs = {entry.name for entry in session.get_inputs()}

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Mean-pooled, L2-normalized vectors, one forward pass per text.

        Texts are never batched. The quantized model computes its activation ranges over
        the whole input, so in a batch a text's vector depends on its neighbours (measured:
        cosine 0.997 against the same text embedded alone), and a batch pads every text to
        the longest one. Alone, a text always gets the same vector, the peak memory is that
        of one text, and for the quantized model it is also the faster way.
        """
        np = load_numpy()
        if np is None:
            raise ImportError("numpy is not installed")
        vectors: list[list[float]] = []
        for text in texts:
            # SentencePiece drops trailing whitespace anyway; stripping makes that explicit
            # and keeps the ids equal to what the reference tokenizer gives for the text.
            pieces = self.tokenizer.encode(text.strip(), out_type=int)
            ids = np.asarray([_token_ids(pieces, self.model.max_tokens)], dtype=np.int64)
            feed = {
                "input_ids": ids,
                "attention_mask": np.ones_like(ids),
                "token_type_ids": np.zeros_like(ids),
            }
            try:
                hidden = self.session.run(
                    None, {k: v for k, v in feed.items() if k in self.inputs}
                )[0]
            except Exception as exc:  # onnxruntime raises its own exception types
                raise RuntimeError(f"onnxruntime could not run {self.model.name}: {exc}") from exc
            pooled = hidden[0].mean(axis=0)
            norm = float(np.linalg.norm(pooled))
            vectors.append((pooled / max(norm, 1e-12)).astype(np.float32).tolist())
        return vectors


def _load(model: LocalModel, folder: Path, threads: int) -> _Loaded:
    absent = missing_packages()
    if absent:
        raise ImportError(
            f"The local embedding provider needs {', '.join(absent)}. "
            f"Install them in the environment Ingatan runs in: {INSTALL_HINT}"
        )
    weights = folder / model.weights.name
    tokenizer_path = folder / model.tokenizer.name
    for path in (weights, tokenizer_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing. Download the model once with: ingatan embeddings pull"
            )
    import onnxruntime  # type: ignore[import-not-found]
    import sentencepiece  # type: ignore[import-not-found]

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = max(1, threads)
    options.inter_op_num_threads = 1
    # The CPU arena keeps the largest batch ever run resident until the session is dropped
    # (about 1 GB after one batch of long notes). Without it that memory goes back at once.
    options.enable_cpu_mem_arena = False
    options.log_severity_level = 3
    try:
        session = onnxruntime.InferenceSession(
            str(weights), options, providers=["CPUExecutionProvider"]
        )
        tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_path))
    except Exception as exc:  # onnxruntime and sentencepiece raise their own exception types
        raise RuntimeError(f"Could not load {model.name} from {folder}: {exc}") from exc
    return _Loaded(model, session, tokenizer)


_load_lock = threading.Lock()  # one model load at a time
_state_lock = threading.Lock()  # guards the four names below
_loaded: _Loaded | None = None
_loaded_key: tuple[str, str, int] | None = None
_last_used = 0.0
_idle_limit = 0  # idle_unload_seconds of the most recent call
_timer: threading.Timer | None = None


def _trim_os_memory() -> None:
    """Hand freed heap back to the operating system (glibc keeps it otherwise)."""
    gc.collect()
    if sys.platform.startswith("linux"):
        with contextlib.suppress(OSError, AttributeError):
            ctypes.CDLL("libc.so.6").malloc_trim(0)


def _on_idle() -> None:
    """The idle timer fired. Drop the model if it really was idle for the current limit.

    The limit is read here, not captured when the timer was started, so a changed setting
    (HTTP mode reloads its configuration) and a limit of 0 are both honoured.
    """
    global _loaded, _loaded_key, _timer
    dropped = False
    with _state_lock:
        _timer = None
        if _loaded is None or _idle_limit <= 0:
            return
        remaining = _idle_limit - (time.monotonic() - _last_used)
        if remaining > 0.05:
            _start_timer_locked(remaining)
            return
        _loaded = None
        _loaded_key = None
        dropped = True
    if dropped:
        _trim_os_memory()


def _start_timer_locked(delay: float) -> None:
    global _timer
    timer = threading.Timer(delay, _on_idle)
    timer.daemon = True
    timer.start()
    _timer = timer


def _mark_used(idle_seconds: int) -> None:
    """Note the use and make sure one idle timer is running (never one per call)."""
    global _last_used, _timer, _idle_limit
    with _state_lock:
        _last_used = time.monotonic()
        changed = idle_seconds != _idle_limit
        _idle_limit = idle_seconds
        if _timer is not None and (changed or idle_seconds <= 0):
            # A timer armed for another limit would fire at the wrong time.
            _timer.cancel()
            _timer = None
        if idle_seconds > 0 and _timer is None:
            _start_timer_locked(float(idle_seconds))


def _get(model: LocalModel, folder: Path, threads: int) -> _Loaded:
    """The loaded model for this key, loading it if needed. Handing it out counts as use,
    so the idle timer cannot drop a model in the middle of a long call."""
    global _loaded, _loaded_key, _last_used
    key = (model.name, str(folder), max(1, threads))
    with _state_lock:
        if _loaded is not None and _loaded_key == key:
            _last_used = time.monotonic()
            return _loaded
    with _load_lock:
        with _state_lock:
            if _loaded is not None and _loaded_key == key:
                _last_used = time.monotonic()
                return _loaded
            # A different model or thread count replaces the one in memory.
            _loaded = None
            _loaded_key = None
        fresh = _load(model, folder, key[2])
        with _state_lock:
            _loaded = fresh
            _loaded_key = key
            _last_used = time.monotonic()
        return fresh


def embed(
    texts: Sequence[str],
    *,
    model_name: str = "",
    model_dir: Path | str = "",
    threads: int = 2,
    idle_unload_seconds: int = 300,
) -> list[list[float]]:
    """L2-normalized vectors for `texts`, in order. Loads the model on first use.

    Raises `ImportError` when a runtime package is missing, `OSError` when a model file is,
    and `RuntimeError` when the model cannot be loaded or run.
    """
    if not texts:
        return []
    model = resolve(model_name)
    loaded = _get(model, model_folder(model_dir, model), threads)
    try:
        return loaded.encode(texts)
    finally:
        _mark_used(idle_unload_seconds)


def unload() -> int:
    """Drop the model from memory now. Returns how many were loaded (0 or 1)."""
    global _loaded, _loaded_key, _timer
    with _state_lock:
        count = 0 if _loaded is None else 1
        _loaded = None
        _loaded_key = None
        if _timer is not None:
            _timer.cancel()
            _timer = None
    if count:
        _trim_os_memory()
    return count
