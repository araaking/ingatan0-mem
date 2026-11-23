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
    "extract_auto_keywords",
    "extract_content_tokens",
    "extract_identifier_phrases",
    "extract_resource_ids",
]

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
# Identifiers joined by _, -, or . (e.g. sync_report, sync_report.py, meta-ads)
_IDENTIFIER = re.compile(r"[A-Za-z0-9]+(?:[_.-][A-Za-z0-9]+)+")
# Google Drive / Sheet IDs (typically 25-60 base64url chars) and URLs
_SHEET_OR_DRIVE_ID = re.compile(r"\b(?=[A-Za-z0-9_-]*[A-Za-z])(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]{25,}\b")
_URL = re.compile(r"https?://[^\s)\]>\"']+", re.IGNORECASE)

# Common Indonesian & English function words (never carry topic identity on their own).
STOPWORDS: frozenset[str] = frozenset(
    {
        # Indonesian
        "ada", "adalah", "agar", "akan", "atau", "bagi", "bahwa", "banyak", "baru", "beberapa",
        "belum", "bersama", "bisa", "boleh", "bukan", "buat", "cara", "cuma", "dalam", "dan",
        "dapat", "dari", "dengan", "di", "dia", "dibuat", "dipakai", "ditambah", "dulu", "hanya",
        "harus", "ia", "ini", "itu", "jadi", "jangan", "jika", "juga", "kalau", "kami", "karena",
        "ke", "ketika", "kita", "lagi", "lain", "lalu", "lebih", "lewat", "masih", "mau", "memang",
        "mereka", "mulai", "namun", "nanti", "nya", "oleh", "pada", "pakai", "paling", "para",
        "per", "perlu", "pernah", "punya", "saat", "saja", "sama", "sampai", "sangat", "saya",
        "sebagai", "sebelum", "sebuah", "secara", "sedang", "sehingga", "sejak", "sekarang",
        "sekitar", "selain", "selalu", "selama", "semua", "sendiri", "seperti", "sering", "serta",
        "setelah", "setiap", "siapa", "sudah", "supaya", "tapi", "telah", "tentang", "terdapat",
        "terhadap", "terjadi", "terkait", "tersebut", "tetap", "tetapi", "tiap", "tidak", "tiga",
        "untuk", "waktu", "walau", "walaupun", "yang",
        # English
        "about", "after", "all", "also", "and", "any", "are", "been", "before", "being", "between",
        "both", "but", "can", "could", "did", "does", "done", "each", "for", "from", "had", "has",
        "have", "her", "here", "him", "his", "how", "into", "its", "just", "like", "made", "make",
        "may", "more", "most", "must", "not", "now", "off", "one", "only", "other", "our", "out",
        "over", "per", "same", "see", "she", "should", "some", "such", "than", "that", "the",
        "their", "them", "then", "there", "these", "they", "this", "those", "through", "too",
        "under", "until", "use", "used", "using", "very", "was", "way", "were", "what", "when",
        "where", "which", "while", "who", "will", "with", "would", "you", "your",
    }
)

# High-frequency agent/tooling/document boilerplate words that cause false-positive conflicts
# when two unrelated facts mention them (e.g. "Disimpan oleh asisten", "Google Sheet", "MCP").
CONFLICT_NOISE_WORDS: frozenset[str] = frozenset(
    {
        "agent", "agen", "asisten", "assistant", "catatan", "claude", "data", "disimpan", "doc",
        "docs", "dokumen", "fakta", "file", "google", "hermes", "info", "informasi", "ingatan",
        "link", "log", "mcp", "memori", "memory", "note", "notes", "rekap", "recap", "server",
        "session", "sesi", "sheet", "sheets", "spreadsheet", "tab", "tool", "tools", "update",
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
