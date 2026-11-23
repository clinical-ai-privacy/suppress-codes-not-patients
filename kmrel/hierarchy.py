"""Code hierarchies for k^m releases of diagnosis codes.

A code is a version-prefixed string, "9:4280" (ICD-9-CM, no dot) or "10:I480" (ICD-10-CM, no dot),
so the two versions never match each other below the root. A hierarchy maps a code and a level
l = 0..top to the string shown at that level. Strings are compared as strings (a released item is
its string): two codes that share an ancestor string share that item.

Two hierarchies:

* ``Trunc`` (the truncation hierarchy of the two earlier local methods): level 0 raw, level 1 three
  characters, level 2 two characters for ICD-9 and one character for ICD-10.
* ``Official``: level 0 raw code, 1 category (three characters; four for ICD-9 E codes),
  2 block (the finest named range of categories of the official tabular list: ICD-9-CM 2011
  sections, ICD-10-CM FY2026 sections), 3 chapter (ICD-9-CM: 17 chapters + V + E; ICD-10-CM: 22
  chapters), 4 root (one token for every code of either version).

Both are nested: every level-l string lies under exactly one level-(l+1) string. ``check_nesting``
verifies this on the codes of a cohort. Codes absent from the official files are placed by their
category, which is stable across editions; a category outside every named range gets its own
block ("<ver>:B<category>") so that nesting still holds, and is counted.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

HDATA = Path(__file__).resolve().parents[1] / "hierarchy_data"
ROOT_TOKEN = "DX"

# ICD-9-CM chapters (numeric part of the category, inclusive)
ICD9_CHAPTERS = ((1, 139), (140, 239), (240, 279), (280, 289), (290, 319), (320, 389), (390, 459),
                 (460, 519), (520, 579), (580, 629), (630, 679), (680, 709), (710, 739), (740, 759),
                 (760, 779), (780, 799), (800, 999))


def icd9_category(c: str) -> str:
    return c[:4] if c.startswith("E") else c[:3]


def icd9_chapter(c: str) -> str:
    # chapter tokens are "CH..": "10:C17" would be the ICD-10 category C17 (C00-C26 exist)
    if c.startswith("V"):
        return "9:CHV"
    if c.startswith("E"):
        return "9:CHE"
    n = int(c[:3])
    for i, (lo, hi) in enumerate(ICD9_CHAPTERS, 1):
        if lo <= n <= hi:
            return f"9:CH{i:02d}"
    raise ValueError("ICD-9 code outside every chapter")


def _icd9_key(cat: str):
    """Sort key of an ICD-9 category: (prefix, number)."""
    if cat[0] in "VE":
        return (cat[0], int(cat[1:]))
    return ("", int(cat))


@lru_cache(maxsize=1)
def icd9_sections():
    """Finest named ranges of ICD-9-CM categories from the 2011 tabular list (Dtab12.rtf).

    Returns a list of (lo_key, hi_key, block_id), the chapter-wide ranges included.
    """
    t = (HDATA / "Dtab12.rtf").read_text(encoding="latin-1")
    s = re.sub(r"\\[a-z]+-?\d* ?", "", t)
    s = re.sub(r"[{}]", "", s)
    s = re.sub(r"(?<=\d)\n(?=\d)", "", s)      # a range broken across lines, "(71\n0-719)"
    # headings start after a blank line, are upper case (they may wrap), and end with the range
    hits = re.findall(r"\n[A-Z][A-Z0-9 ,\-'/&()\[\]\n]{3,}?\s*\((E\d{3}|V\d{2}|\d{3})"
                      r"(?:-(E\d{3}|V\d{2}|\d{3}))?\)", s)
    # chapter-wide ranges are kept: they are the widest, so a category gets the chapter range as
    # its block only when no named section contains it (e.g. chapter 4, 280-289, has none)
    out = {(f"{lo:03d}", f"{hi:03d}") for lo, hi in ICD9_CHAPTERS}
    out |= {("V01", "V91"), ("E000", "E999")}
    out.add(("E849", "E849"))   # "PLACE OF OCCURRENCE (\\*E849\\*E849E849)": the anchor breaks the pattern
    for lo, hi in hits:
        out.add((lo, hi or lo))
    # some section headings are written after an anchor, "\\*010-018\\*010-018TUBERCULOSIS(...)"
    for lo, hi in re.findall(r"\\\*(E\d{3}|V\d{2}|\d{3})-(E\d{3}|V\d{2}|\d{3})\\\*", s):
        out.add((lo, hi))
    secs = []
    for lo, hi in out:
        # E sections are written with three digits after E; categories are E + 3 digits
        secs.append((_icd9_key(lo), _icd9_key(hi), f"9:B{lo}-{hi}"))
    return secs


@lru_cache(maxsize=None)
def icd9_block(cat: str) -> tuple[str, bool]:
    """(block id, found) for an ICD-9 category; the finest named range containing it."""
    k = _icd9_key(cat)
    best = None
    for lo, hi, bid in icd9_sections():
        if lo[0] == k[0] and lo[1] <= k[1] <= hi[1]:
            width = hi[1] - lo[1]
            if best is None or width < best[0]:
                best = (width, bid)
    if best is None:
        return f"9:B{cat}", False
    return best[1], True


@lru_cache(maxsize=1)
def icd10_tables():
    """From the ICD-10-CM FY2026 tabular XML: category -> (section id, chapter number) and the
    list of section ranges (first, last, id, chapter)."""
    root = ET.parse(HDATA / "icd10cm-tabular-2026.xml").getroot()
    cat2 = {}
    ranges = []
    for ch in root.iter("chapter"):
        chn = int(ch.findtext("name"))
        for sec in ch.findall("section"):
            sid = sec.get("id")
            first, _, last = sid.partition("-")
            last = last or first
            ranges.append((first, last, sid, chn))
            for d in sec.findall("diag"):
                cat2[d.findtext("name").replace(".", "")[:3]] = (sid, chn)
    return cat2, ranges


@lru_cache(maxsize=None)
def icd10_block_chapter(cat: str) -> tuple[str, str, bool]:
    cat2, ranges = icd10_tables()
    if cat in cat2:
        sid, chn = cat2[cat]
        return f"10:B{sid}", f"10:CH{chn:02d}", True
    for first, last, sid, chn in ranges:  # a category absent from FY2026: place by range
        if first[:3] <= cat <= last[:3]:
            return f"10:B{sid}", f"10:CH{chn:02d}", True
    return f"10:B{cat}", "10:CH??", False


class Hierarchy:
    name = "base"
    top = 0
    level_names: tuple = ()

    def gen(self, code: str, lvl: int) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def chain(self, code: str) -> tuple:
        """(gen(code, 0), ..., gen(code, top))."""
        return tuple(self.gen(code, l) for l in range(self.top + 1))


class Trunc(Hierarchy):
    """The truncation hierarchy of the two earlier local methods."""
    name = "trunc"
    top = 2
    level_names = ("raw", "3 chars", "2 chars / 1 char")

    def gen(self, code, lvl):
        if lvl <= 0:
            return code
        ver, c = code.split(":", 1)
        return f"{ver}:{c[:3] if lvl == 1 else c[:1 if ver == '10' else 2]}"


class TruncChapter(Trunc):
    """Extension of the truncation hierarchy for eICU: level 3 = ICD-9 chapter, level 4 = one token."""
    name = "trunc_chapter"
    top = 4
    level_names = ("raw", "3 chars", "2 chars", "chapter", "root")

    def gen(self, code, lvl):
        if lvl <= 2:
            return super().gen(code, lvl)
        if lvl == 3:
            ver, c = code.split(":", 1)
            if ver != "9":
                raise ValueError("TruncChapter is ICD-9 only")
            return icd9_chapter(c)
        return ROOT_TOKEN


class Official(Hierarchy):
    name = "official"
    top = 4
    level_names = ("raw", "category", "block", "chapter", "root")

    def __init__(self):
        self.unplaced = set()

    @lru_cache(maxsize=None)
    def _chain(self, code):
        ver, c = code.split(":", 1)
        if ver == "9":
            cat = icd9_category(c)
            blk, ok = icd9_block(cat)
            chap = icd9_chapter(c)
        elif ver == "10":
            cat = c[:3]
            blk, chap, ok = icd10_block_chapter(cat)
        else:
            raise ValueError("unknown ICD version")
        if not ok:
            self.unplaced.add(f"{ver}:{cat}")
        return (code, f"{ver}:{cat}", blk, chap, ROOT_TOKEN)

    def gen(self, code, lvl):
        return self._chain(code)[min(max(lvl, 0), self.top)]

    def chain(self, code):
        return self._chain(code)


HIERARCHIES = {"trunc": Trunc, "trunc_chapter": TruncChapter, "official": Official}


def get(name: str) -> Hierarchy:
    return HIERARCHIES[name]()


def check_nesting(h: Hierarchy, codes) -> dict:
    """Every level-l string has exactly one parent at level l+1 (over the given codes)."""
    bad = {}
    for l in range(h.top):
        up = {}
        for c in codes:
            a, b = h.gen(c, l), h.gen(c, l + 1)
            up.setdefault(a, set()).add(b)
        bad[l] = sum(1 for v in up.values() if len(v) > 1)
    nodes = {l: len({h.gen(c, l) for c in codes}) for l in range(h.top + 1)}
    return dict(nodes=nodes, non_nested=bad)
