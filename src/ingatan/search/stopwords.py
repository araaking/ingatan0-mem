"""Lightweight, dependency-free Indonesian & English stopwords and token helpers.

Used for:
- cleaning `recall` queries so generic function words do not pull in unrelated facts;
- filtering boilerplate and generic words in `similar_facts` conflict detection;
- auto-extracting fallback `keywords` in `remember` when none are provided.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ingatan.search import stem_id

__all__ = [
    "CONFLICT_NOISE_WORDS",
    "STOPWORDS",
    "distinct_dated_snapshots",
    "extract_auto_keywords",
    "extract_content_tokens",
    "extract_date_markers",
    "extract_identifier_phrases",
    "extract_resource_ids",
]

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
# Identifiers joined by _, -, or . (e.g. sync_report, sync_report.py, meta-ads)
_IDENTIFIER = re.compile(r"[A-Za-z0-9]+(?:[_.-][A-Za-z0-9]+)+")
# Google Drive / Sheet IDs (typically 25-60 base64url chars) and URLs
_SHEET_OR_DRIVE_ID = re.compile(
    r"\b(?=[A-Za-z0-9_-]*[A-Za-z])(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]{25,}\b"
)
_URL = re.compile(r"https?://[^\s)\]>\"']+", re.IGNORECASE)

_MONTHS: dict[str, int] = {
    "jan": 1,
    "januari": 1,
    "january": 1,
    "feb": 2,
    "februari": 2,
    "february": 2,
    "mar": 3,
    "maret": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "mei": 5,
    "may": 5,
    "jun": 6,
    "juni": 6,
    "june": 6,
    "jul": 7,
    "juli": 7,
    "july": 7,
    "agu": 8,
    "agust": 8,
    "agustus": 8,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "okt": 10,
    "oktober": 10,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "des": 12,
    "desember": 12,
    "dec": 12,
    "december": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_ISO_DATE = re.compile(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b")
_DAY_MONTH = re.compile(
    rf"\b(\d{{1,2}}(?:\s*[-–]\s*\d{{1,2}})?)\s+({_MONTH_ALT})\b(?:\s+(20\d{{2}}))?",
    re.IGNORECASE,
)
_MONTH_DAY = re.compile(
    rf"\b({_MONTH_ALT})\s+(\d{{1,2}}(?:\s*[-–]\s*\d{{1,2}})?)\b(?:,?\s+(20\d{{2}}))?",
    re.IGNORECASE,
)
_WEEK_MARKER = re.compile(r"\b(?:w(\d{1,2})|minggu\s+(?:ke[- ]?)?(\d{1,2}))\b", re.IGNORECASE)

_SNAPSHOT_CUES: frozenset[str] = frozenset(
    {
        "analisa",
        "analisis",
        "audit",
        "bulanan",
        "cuaca",
        "daily",
        "harian",
        "jadwal",
        "laporan",
        "log",
        "metrik",
        "metrics",
        "mingguan",
        "monthly",
        "performa",
        "prakiraan",
        "progres",
        "progress",
        "recap",
        "rekap",
        "report",
        "ringkasan",
        "runrate",
        "snapshot",
        "summary",
        "tarikan",
        "transaksi",
        "update",
        "weekly",
        "worklog",
    }
)
_EFFECTIVE_CHANGE_CUES: tuple[str, ...] = (
    "mulai",
    "berlaku",
    "sejak",
    "effective",
    "starting",
    "naik jadi",
    "turun jadi",
    "diubah menjadi",
    "berubah menjadi",
    "ganti jadi",
    "diganti menjadi",
)

# Common Indonesian & English function words (never carry topic identity on their own).
STOPWORDS: frozenset[str] = frozenset(
    {
        # Indonesian
        "ada",
        "adalah",
        "agar",
        "akan",
        "apa",
        "apakah",
        "atau",
        "bagaimana",
        "bagi",
        "bahwa",
        "banyak",
        "baru",
        "beberapa",
        "belum",
        "berapa",
        "bersama",
        "bisa",
        "boleh",
        "bukan",
        "buat",
        "cara",
        "cuma",
        "dalam",
        "dan",
        "dapat",
        "dari",
        "dengan",
        "di",
        "dia",
        "dibuat",
        "dimana",
        "dipakai",
        "ditambah",
        "dulu",
        "hanya",
        "harus",
        "ia",
        "ini",
        "itu",
        "jadi",
        "jangan",
        "jika",
        "juga",
        "kalau",
        "kami",
        "kapan",
        "karena",
        "ke",
        "kenapa",
        "ketika",
        "kita",
        "lagi",
        "lain",
        "lalu",
        "lebih",
        "lewat",
        "mana",
        "masih",
        "mau",
        "memang",
        "mengapa",
        "mereka",
        "mulai",
        "namun",
        "nanti",
        "nya",
        "oleh",
        "pada",
        "pakai",
        "paling",
        "para",
        "per",
        "perlu",
        "pernah",
        "punya",
        "saat",
        "saja",
        "sama",
        "sampai",
        "sangat",
        "saya",
        "sebagai",
        "sebelum",
        "sebuah",
        "secara",
        "sedang",
        "sehingga",
        "sejak",
        "sekarang",
        "sekitar",
        "selain",
        "selalu",
        "selama",
        "semua",
        "sendiri",
        "seperti",
        "sering",
        "serta",
        "setelah",
        "setiap",
        "siapa",
        "sudah",
        "supaya",
        "tapi",
        "telah",
        "tentang",
        "terdapat",
        "terhadap",
        "terjadi",
        "terkait",
        "tersebut",
        "tetap",
        "tetapi",
        "tiap",
        "tidak",
        "tiga",
        "untuk",
        "waktu",
        "walau",
        "walaupun",
        "yang",
        # English
        "about",
        "after",
        "all",
        "also",
        "and",
        "any",
        "are",
        "been",
        "before",
        "being",
        "between",
        "both",
        "but",
        "can",
        "could",
        "did",
        "does",
        "done",
        "each",
        "for",
        "from",
        "had",
        "has",
        "have",
        "her",
        "here",
        "him",
        "his",
        "how",
        "into",
        "its",
        "just",
        "like",
        "made",
        "make",
        "may",
        "more",
        "most",
        "must",
        "not",
        "now",
        "off",
        "one",
        "only",
        "other",
        "our",
        "out",
        "over",
        "same",
        "see",
        "she",
        "should",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "too",
        "under",
        "until",
        "use",
        "used",
        "using",
        "very",
        "was",
        "way",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)

# High-frequency agent/tooling/document boilerplate words that cause false-positive conflicts
# when two unrelated facts mention them (e.g. "Disimpan oleh asisten", "Google Sheet", "MCP").
CONFLICT_NOISE_WORDS: frozenset[str] = frozenset(
    {
        "agent",
        "agen",
        "asisten",
        "assistant",
        "catatan",
        "claude",
        "data",
        "disimpan",
        "doc",
        "docs",
        "dokumen",
        "fakta",
        "file",
        "google",
        "hermes",
        "info",
        "informasi",
        "ingatan",
        "link",
        "log",
        "mcp",
        "memori",
        "memory",
        "note",
        "notes",
        "rekap",
        "recap",
        "server",
        "session",
        "sesi",
        "sheet",
        "sheets",
        "spreadsheet",
        "tab",
        "tool",
        "tools",
        "update",
        "via",
    }
)


def extract_content_tokens(
    *texts: str,
    exclude: frozenset[str] = frozenset(),
    include_noise: bool = False,
    min_length: int = 3,
) -> set[str]:
    """Meaningful lowercase tokens for semantic-free overlap comparison."""
    blocked = STOPWORDS | exclude if include_noise else STOPWORDS | CONFLICT_NOISE_WORDS | exclude
    out: set[str] = set()
    for text in texts:
        for word in _WORD.findall(text):
            low = word.casefold()
            if len(low) < min_length or low.isdigit() or low in blocked:
                continue
            out.add(low)
    return out


def extract_identifier_phrases(query: str) -> list[str]:
    """Turn joined identifiers like `sync_report` or `sync_report.py` into FTS5 phrase strings."""
    phrases: list[str] = []
    seen: set[str] = set()
    for match in _IDENTIFIER.findall(query):
        parts = [p for p in _WORD.findall(match) if p]
        if len(parts) < 2:
            continue
        # Strip common file extensions from the phrase so `sync_report` matches `sync_report.py`.
        if len(parts) >= 3 and parts[-1].casefold() in {"py", "md", "js", "ts", "json", "toml"}:
            parts = parts[:-1]
        phrase = " ".join(parts)
        key = phrase.casefold()
        if key not in seen:
            seen.add(key)
            phrases.append(phrase)
    return phrases


def extract_resource_ids(text: str) -> set[str]:
    """Distinct external resource IDs (Google Sheet/Drive IDs and URLs) present in `text`."""
    ids: set[str] = set()
    for url in _URL.findall(text):
        ids.add(url.rstrip(".,;:").casefold())
    for sid in _SHEET_OR_DRIVE_ID.findall(text):
        ids.add(sid)
    return ids


def _norm_day_span(raw: str) -> str:
    parts = [str(int(p)) for p in re.split(r"\s*[-–]\s*", raw.strip()) if p.isdigit()]
    return "-".join(parts)


def extract_date_markers(text: str) -> set[str]:
    """Explicit calendar dates, date ranges, or week numbers mentioned in `text`."""
    markers: set[str] = set()
    for y, m, d in _ISO_DATE.findall(text):
        markers.add(f"{int(d)}/{int(m)}/{y}")
        markers.add(f"{int(d)}/{int(m)}")
    for day_raw, mon_raw, year in _DAY_MONTH.findall(text):
        span = _norm_day_span(day_raw)
        mon = _MONTHS[mon_raw.casefold()]
        if span:
            markers.add(f"{span}/{mon}/{year}" if year else f"{span}/{mon}")
    for mon_raw, day_raw, year in _MONTH_DAY.findall(text):
        span = _norm_day_span(day_raw)
        mon = _MONTHS[mon_raw.casefold()]
        if span:
            markers.add(f"{span}/{mon}/{year}" if year else f"{span}/{mon}")
    for w1, w2 in _WEEK_MARKER.findall(text):
        num = int(w1 or w2)
        if 1 <= num <= 53:
            markers.add(f"w{num}")
    return markers


def _has_effective_change_cue(lowered: str) -> bool:
    for cue in _EFFECTIVE_CHANGE_CUES:
        if re.search(rf"(?<!\w){re.escape(cue)}(?!\w)", lowered):
            return True
    return False


def _has_snapshot_cue(text: str) -> bool:
    return any(w.casefold() in _SNAPSHOT_CUES for w in _WORD.findall(text))


def distinct_dated_snapshots(text_a: str, text_b: str) -> bool:
    """True when two texts describe periodic snapshots/logs for different dates or weeks.

    Standing updates that use effective-date phrasing ("mulai 10 Nov", "berlaku sejak ...")
    return False so real rule or price updates still trigger conflict detection.
    """
    dates_a = extract_date_markers(text_a)
    dates_b = extract_date_markers(text_b)
    if not dates_a or not dates_b or not dates_a.isdisjoint(dates_b):
        return False
    low_a, low_b = text_a.casefold(), text_b.casefold()
    if _has_effective_change_cue(low_a) or _has_effective_change_cue(low_b):
        return False
    if _has_snapshot_cue(text_a) or _has_snapshot_cue(text_b):
        return True
    # Week markers (W46 vs W47) or explicit date ranges (1-8 Nov vs 9-15 Nov) are periodic.
    periodic_a = any(m.startswith("w") or "-" in m for m in dates_a)
    periodic_b = any(m.startswith("w") or "-" in m for m in dates_b)
    return periodic_a and periodic_b


def extract_auto_keywords(
    text: str, entities: Sequence[str] = (), *, max_keywords: int = 12
) -> list[str]:
    """Derive fallback search keywords from `text` and `entities` when caller omits `keywords`."""
    seen: set[str] = set()
    out: list[str] = []

    def add(word: str) -> bool:
        key = word.casefold().strip()
        if not key or key in seen:
            return False
        seen.add(key)
        out.append(key)
        return len(out) >= max_keywords

    # Keep short entity names first, then meaningful content tokens from text.
    for entity in entities:
        for part in _WORD.findall(entity):
            low = part.casefold()
            if len(low) >= 2 and low not in STOPWORDS and add(low):
                return out

    base_words: list[str] = []
    for part in _WORD.findall(text):
        low = part.casefold()
        if len(low) >= 3 and not low.isdigit() and low not in STOPWORDS:
            base_words.append(low)
            if add(low):
                return out

    # Add Indonesian affix-stripped roots for better recall on inflected queries.
    for word in base_words:
        for root in stem_id.candidates(word):
            if root not in STOPWORDS and add(root):
                return out

    # Ultra-short text fallback (e.g. "x" or "Rp 50"): keep any word token so keywords is non-empty.
    if not out:
        for part in _WORD.findall(text):
            if add(part.casefold()):
                return out
    return out
