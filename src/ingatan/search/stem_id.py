"""A small, dependency-free Indonesian affix stripper used only to expand recall queries.

This is a candidate generator, not a dictionary stemmer. For one query word it proposes a
few plausible roots (`pengiriman` -> `kirim`). It over-generates on purpose, because the
caller scores the extra terms below an exact match, and it never decides what is a real
word. Nothing it returns is stored anywhere.

Rules:

- particles `-lah -kah -pun` and possessives `-nya -ku -mu` come off first;
- derivational suffixes `-kan -an -i` come off next;
- then one prefix: `di- ke- se- ter- ber- be- per- pe-`, the compounds `memper- diper-`, and the
  `meN-` / `peN-` family with the nasal restored (`meng-` + vowel -> bare or `k`,
  `meny-` -> `s`, `mem-` -> `p` or kept `b f v p`, `men-` -> `t` or kept `d c j z`,
  `me-`/`pe-` before `l r w y m n`). `pe-` also strips before any other consonant
  (`pekerja` -> `kerja`);
- for a root that starts with k, t, s or p the assimilated form is added too (`kirim` ->
  `ngirim`), because `pengiriman` does not contain `kirim`; this is what lets the query
  `dikirim` find a text that only says `Pengiriman`;
- a candidate is kept only when it passes `_plausible`: at least `MIN_ROOT` letters, a
  vowel, a legal onset and a legal ending. That is what stops English words such as
  `mention`, `members` or `premium` from producing a root.

`MIN_ROOT` is 4, not 3. A three letter candidate (`pes`, `tak`) matches inside far too many
words in the trigram index; real three letter roots (`uji`, `pos`) are still found by their
literal form. Callers use the candidates against both indexes.
"""

from __future__ import annotations

from functools import lru_cache

__all__ = ["MAX_CANDIDATES", "MIN_ROOT", "candidates"]

MAX_CANDIDATES = 6
MIN_ROOT = 4
_MIN_WORD = 5  # shorter words are never stripped

_VOWELS = frozenset("aiueo")
# First letter of a root -> what meN-/peN- turns it into (kirim -> mengirim, tukar -> menukar).
_NASAL_FORM = {"k": "ng", "t": "n", "s": "ny", "p": "m"}
_PARTICLES = ("lah", "kah", "pun")
_POSSESSIVES = ("nya", "ku", "mu")
_ONSETS = frozenset(
    {
        "ng", "ny", "sy", "kh", "gr", "pr", "br", "tr", "dr", "kr", "fr", "cr", "sk", "st",
        "sp", "pl", "bl", "kl", "fl", "gl", "sl", "sw", "kw", "ps", "sm", "sn", "tw",
    }
)  # fmt: skip

# Function words that look derived. They are not expanded.
_STOP = frozenset(
    {
        "seperti", "sebuah", "setiap", "semua", "sebagai", "sebelum", "sesudah", "selalu",
        "sering", "sendiri", "sampai", "kepada", "ketika", "bersama", "beberapa", "belum",
        "perlu", "mereka", "terhadap", "tersebut", "sekarang", "sedang", "selain", "serta",
        "seluruh", "sekolah", "telah", "salah", "sebab", "sekitar", "dengan", "kembali",
        "pernah", "berapa", "bagaimana", "kenapa", "mengapa", "terima", "dimana", "kemana",
    }
)  # fmt: skip


def _is_vowel(ch: str) -> bool:
    return ch in _VOWELS


def _plausible(c: str) -> bool:
    """Could `c` be an Indonesian root? Cheap shape test, not a lookup."""
    if len(c) < MIN_ROOT or not (c.isascii() and c.isalpha()):
        return False
    if not any(ch in _VOWELS for ch in c):
        return False
    if c[0] not in _VOWELS and c[1] not in _VOWELS and c[:2] not in _ONSETS:
        return False
    last, before = c[-1], c[-2]
    if last not in _VOWELS and last != "y" and (before in _VOWELS or c.endswith("ng")):
        return True
    return last in _VOWELS


def _nasal(s: str, pre: str, out: list[str]) -> None:
    """`meN-` / `peN-` with the dropped first letter of the root restored."""
    if not s.startswith(pre) or len(s) < len(pre) + 4:
        return
    r = s[len(pre) :]
    if r.startswith("ng"):
        r2 = r[2:]
        if _is_vowel(r2[0]):
            out += [r2, "k" + r2]
        else:
            out.append(r2)
    elif r.startswith("ny"):
        r2 = r[2:]
        if _is_vowel(r2[0]):
            out += ["s" + r2, r]
    elif r[0] == "m":
        r1 = r[1:]
        if _is_vowel(r1[0]):
            out.append("p" + r1)
        elif r1[0] in "bfvp":
            out.append(r1)
        if _is_vowel(r[1]):
            out.append(r)
    elif r[0] == "n":
        r1 = r[1:]
        if _is_vowel(r1[0]):
            out += ["t" + r1, r]
        elif r1[0] in "dcjz":
            out.append(r1)
    elif r[0] in "lrwy" and _is_vowel(r[1]):
        out.append(r)
    if pre == "pe" and r[0] not in _VOWELS:
        out.append(r)  # pekerja, petani, pedagang


def _prefix_roots(s: str) -> list[str]:
    out: list[str] = []
    for pre in ("memper", "diper"):
        if s.startswith(pre):
            out.append(s[len(pre) :])
    _nasal(s, "me", out)
    _nasal(s, "pe", out)
    for pre in ("di", "ke", "se", "ter", "ber", "per", "be"):
        if s.startswith(pre):
            out.append(s[len(pre) :])
    if s.startswith(("bel", "pel")) and s[3:].startswith("ajar"):
        out.append("ajar")  # belajar, pelajar
    return out


def _outer_variants(word: str) -> list[str]:
    """The word, then without a particle, then without a possessive as well."""
    variants = [word]
    current = word
    for particle in _PARTICLES:
        if current.endswith(particle) and len(current) - len(particle) >= MIN_ROOT:
            current = current[: -len(particle)]
            variants.append(current)
            break
    for possessive in _POSSESSIVES:
        if current.endswith(possessive) and len(current) - len(possessive) >= MIN_ROOT:
            variants.append(current[: -len(possessive)])
            break
    return variants


def _suffix_variants(v: str) -> list[str]:
    """Most stripped forms first, the word itself last."""
    forms: list[str] = []
    if v.endswith("kan"):
        forms.append(v[:-3])
    if v.endswith("an"):
        forms.append(v[:-2])
    if v.endswith("i"):
        forms.append(v[:-1])
    forms.append(v)
    return forms


@lru_cache(maxsize=4096)
def _candidates(word: str) -> tuple[str, ...]:
    if len(word) < _MIN_WORD or not (word.isascii() and word.isalpha()) or word in _STOP:
        return ()
    found: dict[str, None] = {}
    bare: dict[str, None] = {}  # suffix or particle stripped only: weaker, so listed last
    for variant in _outer_variants(word):
        for sv in _suffix_variants(variant):
            if len(sv) >= _MIN_WORD:
                for root in _prefix_roots(sv):
                    if _plausible(root):
                        found.setdefault(root)
            if sv != word and _plausible(sv):
                bare.setdefault(sv)
    # The reverse direction: `kirim` is not inside `pengiriman` (the k fell away), but the
    # assimilated form `ngirim` is. These come after the real roots and before the bare ones.
    for root in list(found):
        if root[0] in _NASAL_FORM and root[1] in _VOWELS:
            found.setdefault(_NASAL_FORM[root[0]] + root[1:])
    found.update(bare)
    found.pop(word, None)
    return tuple(found)[:MAX_CANDIDATES]


def candidates(word: str) -> list[str]:
    """Plausible roots of `word` (lowercase letters), at most `MAX_CANDIDATES`.

    Never contains `word` itself and never anything shorter than `MIN_ROOT`. The caller keeps
    the original word and treats these as weaker evidence.
    """
    return list(_candidates(word.casefold()))
