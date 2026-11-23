#!/usr/bin/env python3
"""Compare every number printed in the tables of the k^m paper with the aggregate result files.

    python3 -I check_tables.py <manuscript.tex> <results_cluster dir> [--verbose] [--selftest]

The printed numbers are read from the .tex source itself (nothing is typed by hand); the values
come from results_cluster/*.jsonl, keeping the last written row of every configuration, as
code/scripts/report.py does. A printed number passes when it equals the result rounded to the
printed number of decimals (tolerance: half a unit of the last printed digit). A mismatch is
marked "rounded twice" when the printed number is what one gets by rounding the value first as
RESULTS.md prints it (two decimals for percentages, four for AUROC) and then again, half up.

Covered: Tables 2-11 and Table S1 (every number in the table bodies and the raw AUROC in the
section headers of Tables 5 and S1), plus seven sentences of the running text that quote ranges
or ratios. Not covered: captions, the abstract, and other numbers of the running text.

--selftest changes every checked printed number by one unit of its last digit, one at a time,
and requires that exactly that cell changes its verdict; it also perturbs each result value used
by a table cell and requires the cell to fail. Standard library only; reads aggregates only.
"""
import json
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

NUM = re.compile(r"(?<![\w.,])[+-]?(?:\d{1,3}(?:,\d{3})+(?![\d.])|\d+(?:\.\d+)?)")
MASK = [re.compile(p) for p in (
    r"\\midrule", r"\\cmidrule\([a-z]*\)\{\d+-\d+\}",
    r"\\multicolumn\{\d+\}\{(?:[^{}]|\{[^{}]*\})*\}", r"\\multirow\{\d+\}\{\*\}",
    r"\\ref\{[^}]*\}", r"\\label\{[^}]*\}", r"\\shortstack\[[a-z]\]")]
LEVEL = {"category": 1, "block": 2, "chapter": 3, "root": 4}
OUTCOME = {"in-hospital death": "death", "30-day readmission": "readmit30",
           "death": "death", "readmission": "readmit30"}
DB = {"eICU-CRD": "eicu", "eICU": "eicu", "MIMIC-IV": "mimic"}


# ------------------------------------------------------------------------------ results ----
class Results:
    def __init__(self, folder):
        self.folder = Path(folder)

        def rows(name):
            f = self.folder / f"{name}.jsonl"
            return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]

        self.coh = {r["name"]: r for r in rows("cohort")}
        self.hc = {(r["db"], r["hierarchy"]): r for r in rows("hierarchy_check")}
        self.fl = {(r["db"], r["hierarchy"], r["K"], r["m"], r["L"]): r for r in rows("floor_levels")}
        self.rel = {}
        for r in rows("releases"):          # strata missing in the first runs = sex x age band
            key = (r["db"], r["hierarchy"].split("@")[0], r["L"], r["m"], r["K"],
                   r.get("strata") or "band", r["method"])
            self.rel[key] = r
        self.arm = {(r["db"], r["outcome"], r["arm"]): r for r in rows("utility_arms")}
        self.armrf = {(r["db"], r["outcome"], r["arm"]): r for r in rows("utility_arms_rf")}
        self.armm = {(r["db"], r["outcome"], r["arm"]): r for r in rows("utility_arms_match")}
        self.pairm = {(r["db"], r["outcome"], r["a"], r["b"]): r for r in rows("utility_pairs_match")}
        self.ra = {r["key"]: r for r in rows("reaudit")}


class Val:
    """A value taken from the results: number, where it comes from, and (row, field) if direct."""

    def __init__(self, value, src, ref=None, scale=1.0, const=False):
        self.value, self.src, self.ref, self.scale, self.const = value, src, ref, scale, const


def field(index, name, key, fld, scale=1.0):
    row = index[key]
    v = row[fld]
    if v is None:
        raise KeyError(f"{name}{key}.{fld} is null")
    return Val(scale * v, f"{name} {key} {fld}", (row, fld), scale)


# -------------------------------------------------------------------------- tex parsing ----
class Tok:
    def __init__(self, text, off):
        self.text, self.off = text, off
        self.value = float(text.replace(",", ""))
        self.dec = len(text.split(".")[1]) if "." in text else 0
        self.comma = "," in text

    def render(self, v):
        s = f"{int(round(abs(v))):,}" if self.comma else f"{abs(v):.{self.dec}f}"
        sign = "-" if v < 0 else ("+" if self.text.startswith("+") else "")
        return sign + s


class Cell:
    def __init__(self, orig, masked, off):
        self.orig, self.masked, self.off = orig, masked, off
        self.label = " ".join(orig.split())
        self.toks = [Tok(m.group(0), off + m.start()) for m in NUM.finditer(masked)]


def strip_changes(tex):
    """The text as printed without change markup: \\chgdel{id}{old} and \\chgmark{id} are dropped,
    \\chgm{new} is replaced by its content."""
    def group(j):
        depth, k = 0, j
        while True:
            c = tex[k]
            if c == "\\":
                k += 2
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return tex[j + 1:k], k + 1
            k += 1

    out, i = [], 0
    while i < len(tex):
        for name, nargs, keep in (("\\chgdel{", 2, False), ("\\chgmark{", 1, False), ("\\chgm{", 1, True)):
            if tex.startswith(name, i):
                j, content = i + len(name) - 1, ""
                for _ in range(nargs):
                    content, j = group(j)
                if keep:
                    out.append(strip_changes(content))
                i = j
                break
        else:
            out.append(tex[i])
            i += 1
    return "".join(out)


def mask(text):
    for rx in MASK:
        text = rx.sub(lambda m: " " * len(m.group(0)), text)
    return text


def split_top(text, base, sep):
    """Split at `sep` ('\\\\' row break or '&') outside braces; yields (start offset, piece)."""
    out, depth, i, start = [], 0, 0, 0
    while i < len(text):
        c = text[i]
        if c == "\\":
            if sep == "\\\\" and text[i:i + 2] == "\\\\" and depth == 0:
                out.append((base + start, text[start:i]))
                i += 2
                start = i
                continue
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == "&" and sep == "&" and depth == 0:
            out.append((base + start, text[start:i]))
            start = i + 1
        i += 1
    out.append((base + start, text[start:]))
    return out


def table(tex, label):
    """(header text, rows); a row is a list of Cell. Body = first \\midrule .. \\bottomrule."""
    i = tex.index("\\label{%s}" % label)
    b = tex.index("\\begin{tabular", i)
    e = tex.index("\\end{tabular", b)
    m = tex.index("\\midrule", b, e)
    start, end = m + len("\\midrule"), tex.rindex("\\bottomrule", b, e)
    body, mbody = tex[start:end], mask(tex[start:end])
    rows = []
    for off, piece in split_top(mbody, start, "\\\\"):
        if not piece.strip():
            continue
        cells = [Cell(tex[o:o + len(p)], p, o) for o, p in split_top(piece, off, "&")]
        rows.append(cells)
    return " ".join(tex[b:m].split()), rows


# ------------------------------------------------------------------------------ checker ----
class Checker:
    def __init__(self):
        self.rec, self.extra, self.struct, self.text = [], [], [], []

    def cell(self, tab, row, col, tok, val):
        ok = abs(tok.value - val.value) <= 0.5 * 10 ** (-tok.dec) + 1e-9
        self.rec.append(dict(tab=tab, row=row, col=col, tok=tok, val=val,
                             status=("CONST" if val.const else "OK") if ok else "MISMATCH"))

    def row(self, tab, label, cols, toks, vals):
        if len(toks) != len(vals):
            self.struct.append(f"{tab} | {label}: {len(toks)} printed numbers, {len(vals)} expected "
                               f"({[t.text for t in toks]})")
            return
        for i, (t, v) in enumerate(zip(toks, vals)):
            self.cell(tab, label, cols[i] if i < len(cols) else f"#{i + 1}", t, v)

    def same(self, tab, label, what, a, b, dec):
        if round(a, dec) != round(b, dec):
            self.extra.append(f"{tab} | {label}: {what}: {a:.5f} vs {b:.5f}")

    def header(self, tab, header, *needles):
        for n in needles:
            if n not in header:
                self.struct.append(f"{tab}: column header changed, expected to find '{n}'")


def toks(cells):
    return [t for c in cells for t in c.toks]


def release_method(label):
    lab = label.replace("\\ ", " ")
    table_ = [("Uniform", None), ("Floor = Apriori", ["floor"]), ("Floor", ["floor"]),
              ("B3s / B3gs", ["b3s", "b3gs"]), ("B3 / B3g", ["b3", "b3g"]), ("B3gs", ["b3gs"]),
              ("B3g", ["b3g"]), ("B3s", ["b3s"]), ("S-loc$_0$", ["sloc0"]), ("S-loc", ["slocL"]),
              ("A (fails)", ["A_import"]), ("A", ["A_import"]), ("B2 (fails)", ["b2_import"]),
              ("B2", ["b2_import"])]
    for k, v in table_:
        if lab.startswith(k):
            return v
    raise KeyError(f"unknown release label '{label}'")


def level_of(label, methods):
    lab = label.replace("\\ ", " ")
    if lab.startswith("trunc."):
        return "trunc", 2
    if lab == "code":                       # S-loc_0: raw codes, cached with the block-level runs
        return "official", 2
    return "official", LEVEL[lab]


# ------------------------------------------------------------------------- table checks ----
def t_cohorts(tex, R, C):
    tab = "Table 2 (cohorts)"
    head, rows = table(tex, "tab:cohorts")
    C.header(tab, head, "& MIMIC-IV & eICU-CRD")
    m, e = R.coh["mimic"], R.coh["eicu"]
    hm, he = R.hc[("mimic", "official")], R.hc[("eicu", "official")]

    def f(row, fld, s=1.0):
        return Val(s * row[fld], f"cohort {row['name']} {fld}", (row, fld), s)

    def nodes(h, l):
        return Val(h["nodes"][str(l)], f"hierarchy_check {h['db']} official level {l}", (h["nodes"], str(l)))

    spec = [("Patients", [f(m, "patients"), f(e, "patients")]),
            ("Strata", [f(m, "strata"), f(m, "smallest_stratum"), f(e, "strata"), f(e, "smallest_stratum")]),
            ("Distinct codes", [f(m, "codes_distinct"), f(e, "codes_distinct")]),
            ("Categories", [nodes(hm, 1), nodes(hm, 2), nodes(hm, 3), nodes(he, 1), nodes(he, 2), nodes(he, 3)]),
            ("Median codes", [f(m, "median_codes"), f(m, "share_gt14", 100), f(e, "median_codes"), f(e, "share_gt14", 100)])]
    if len(rows) != len(spec):
        C.struct.append(f"{tab}: {len(rows)} rows, expected {len(spec)}")
        return
    for cells, (lab, vals) in zip(rows, spec):
        if not cells[0].label.startswith(lab):
            C.struct.append(f"{tab}: row '{cells[0].label}' where '{lab}' was expected")
            continue
        C.row(tab, lab, ["MIMIC-IV", "MIMIC-IV", "MIMIC-IV", "eICU", "eICU", "eICU"], toks(cells[1:]), vals)


def built(R, db=None, m=None, K=5):
    """Releases built by the paper's algorithms (not the imported A / B2), sex x age strata."""
    return [r for k, r in R.rel.items() if (db is None or k[0] == db) and (m is None or k[3] == m)
            and k[4] == K and k[5] == "band" and not k[6].endswith("_import")]


def t_validity(tex, R, C):
    tab = "Table 3 (validity)"
    head, rows = table(tex, "tab:validity")
    C.header(tab, head, "\\multicolumn{2}{c}{MIMIC-IV} & \\multicolumn{2}{c}{eICU-CRD}",
             "$m=3$ & $m=2$ & $m=3$ & $m=2$")
    cols = [("mimic", 3), ("mimic", 2), ("eicu", 3), ("eicu", 2)]
    fld = {"violators, loose count": "loose_violator_share", "violators, strict count": "strict_violator_share",
           "unique, strict count": "strict_unique_share", "attack, mean success": "attack_mean",
           "attack, certain": "attack_certain"}
    group = None
    for cells in rows:
        g = re.search(r"\\multirow\{\d+\}\{\*\}\{(.*)\}\s*$", cells[0].orig, re.S)
        if g:
            group = "ALL" if "All releases" in g.group(1) else g.group(1).strip()
        meas = cells[1].label
        vals = []
        for db, m in cols:
            if group in ("A", "B2"):
                meth = {"A": "A_import", "B2": "b2_import"}[group]
                vals.append(field(R.rel, "releases", (db, "trunc", 2, m, 5, "band", meth), fld[meas], 100))
            else:
                rs = built(R, db, m)
                if meas == "violators, both counts":
                    v = max(max(r["strict_violator_share"], r["loose_violator_share"]) for r in rs)
                else:
                    v = max(r[fld[meas]] for r in rs if r.get(fld[meas]) is not None)
                vals.append(Val(100 * v, f"max over {len(rs)} built releases, {db} m={m} K=5, sex x age strata"))
        C.row(tab, f"{group} | {meas}", [f"{d} m={m}" for d, m in cols], toks(cells[2:]), vals)


def t_floor(tex, R, C):
    tab = "Table 4 (floor)"
    head, rows = table(tex, "tab:floor")
    C.header(tab, head, "\\multicolumn{4}{c}{MIMIC-IV} & \\multicolumn{4}{c}{eICU-CRD}",
             "code & category & block & chapter & code & category & block & chapter")
    for cells in rows:
        K, m = int(cells[0].label), int(cells[1].label)
        vals = [field(R.fl, "floor_levels", (db, "official", K, m, L), "refused_share", 100)
                for db in ("mimic", "eicu") for L in range(4)]
        C.row(tab, f"K={K} m={m}", [f"{db} L{L}" for db in ("mimic", "eicu") for L in range(4)],
              toks(cells[2:]), vals)


def t_cost(tex, R, C, label, tab, m):
    head, rows = table(tex, label)
    C.header(tab, head, "Release & Level & Refused & Removed & Raw & $\\Delta$AUROC")
    db = oc = None
    for cells in rows:
        h = re.search(r"\\emph\{(eICU-CRD|MIMIC-IV), (in-hospital death|30-day readmission) "
                      r"\(raw AUROC (\d\.\d+)\)\}", cells[0].orig)
        if h:
            db, oc = DB[h.group(1)], OUTCOME[h.group(2)]
            t = Tok(h.group(3), cells[0].off + h.start(3))
            C.cell(tab, f"{db} {oc} header", "raw AUROC", t, field(R.arm, "utility_arms", (db, oc, "raw:all"), "auroc"))
            continue
        lab = f"{db} {oc} | {cells[0].label} | {cells[1].label}"
        meths = release_method(cells[0].label)
        hier, L = level_of(cells[1].label, meths)
        ref, rem, raw, dau = cells[2], cells[3], cells[4], cells[5]
        if meths is None:                                   # uniform coding, nobody refused
            arm = (db, oc, f"official_L{L}:all")
            vals = [Val(0.0, "uniform level: no entry at level 0, by definition", const=True)]
            vals += [field(R.arm, "utility_arms", arm, f) for f in ("d_raw", "d_raw_lo", "d_raw_hi")]
            C.row(tab, lab, ["Raw", "dAUROC", "CI low", "CI high"], toks([ref, rem, raw, dau]), vals)
            continue
        rk = [(db, hier, L, m, 5, "band", me) for me in meths]
        ak = [(db, oc, f"{db}_{hier}{L}_m{m}_K5_{me}") for me in meths]
        for cell, fld, name in ((ref, "refused_share", "Refused"), (rem, "entries_removed_share", "Removed"),
                                (raw, "share_level0", "Raw")):
            vs = [field(R.rel, "releases", k, fld, 100) for k in rk]
            if len(cell.toks) == 1 and len(vs) == 2:        # one number printed for two methods
                C.row(tab, lab, [name], cell.toks, vs[:1])
                C.same(tab, lab, f"{name}: one number printed for {meths[0]} and {meths[1]}",
                       vs[0].value, vs[1].value, cell.toks[0].dec)
            else:
                C.row(tab, lab, [f"{name} {me}" for me in meths], cell.toks, vs)
        n = len(dau.toks)
        d = [field(R.arm, "utility_arms", k, "d_raw") for k in ak]
        ci = [field(R.arm, "utility_arms", ak[-1], f) for f in ("d_raw_lo", "d_raw_hi")]
        if n == 0:                                          # "not estimable": one class in training
            a = R.arm[ak[0]]["auroc"]
            if abs(a - 0.5) > 1e-12:
                C.extra.append(f"{tab} | {lab}: printed as not estimable, but AUROC = {a:.4f}")
        elif n == len(meths):
            C.row(tab, lab, [f"dAUROC {me}" for me in meths], dau.toks, d)
        elif n == len(meths) + 2:
            C.row(tab, lab, [f"dAUROC {me}" for me in meths] + [f"CI low {meths[-1]}", f"CI high {meths[-1]}"],
                  dau.toks, d + ci)
        else:
            C.struct.append(f"{tab} | {lab}: {n} numbers in the dAUROC cell")
        if cells[0].label.startswith("Floor = Apriori"):    # the row claims the two are identical
            fr, ar = R.rel[rk[0]], R.rel[(db, hier, L, m, 5, "band", "apriori")]
            for f in ("refused_share", "entries_removed_share", "share_level0"):
                C.same(tab, lab, f"floor vs apriori {f}", 100 * fr[f], 100 * ar[f], 1)
            C.same(tab, lab, "floor vs apriori dAUROC", R.arm[ak[0]]["d_raw"],
                   R.arm[(db, oc, f"{db}_{hier}{L}_m{m}_K5_apriori")]["d_raw"], 3)


SETTINGS = [("eicu", "death"), ("mimic", "death"), ("mimic", "readmit30")]


def t_k(tex, R, C):
    tab = "Table 6 (K = 5/10/20)"
    head, rows = table(tex, "tab:k")
    C.header(tab, head, "eICU death & MIMIC-IV death & MIMIC-IV readmission")
    for cells in rows:
        if "Attack" in cells[0].orig:
            vals = []
            for K in (5, 10, 20):
                rs = built(R, K=K)
                vals.append(Val(100 * max(r["attack_mean"] for r in rs if r.get("attack_mean") is not None),
                                f"max attack_mean over {len(rs)} built releases, K={K}, sex x age strata"))
            C.row(tab, "attack, max mean success", ["K=5", "K=10", "K=20"], toks(cells[1:]), vals)
            continue
        meth = release_method(cells[0].label)[0]
        hier, L = level_of(cells[1].label, [meth])
        for cell, (db, oc) in zip(cells[2:], SETTINGS):
            if not cell.toks:
                continue
            vals = [field(R.arm, "utility_arms", (db, oc, f"{db}_{hier}{L}_m3_K{K}_{meth}"), "d_raw")
                    for K in (5, 10, 20)]
            C.row(tab, f"{cells[0].label} | {cells[1].label} | {db} {oc}", ["K=5", "K=10", "K=20"], cell.toks, vals)


def t_hosp(tex, R, C):
    tab = "Table 7 (hospital alone)"
    head, rows = table(tex, "tab:hosp")
    C.header(tab, head, "Level & Floor refused (\\%) & S-loc removed (\\%) & Floor $\\Delta$AUROC & "
                        "S-loc $\\Delta$AUROC & B3gs $\\Delta$AUROC")
    for cells in rows:
        L = LEVEL[cells[0].label]

        def rel(strata, meth, fld):
            return field(R.rel, "releases", ("eicu", "official", L, 3, 5, strata, meth), fld, 100)

        def arm(tag, meth):
            return field(R.arm, "utility_arms", ("eicu", "death", f"eicu_official{L}_m3_K5{tag}_{meth}"), "d_raw")

        vals = [rel("band", "floor", "refused_share"), rel("hospital", "floor", "refused_share"),
                rel("band", "slocL", "entries_removed_share"), rel("hospital", "slocL", "entries_removed_share"),
                arm("", "floor"), arm("_hosp", "floor"), arm("", "slocL"), arm("_hosp", "slocL"),
                arm("", "b3gs"), arm("_hosp", "b3gs")]
        cols = [f"{a} {b}" for a in ("floor refused", "S-loc removed", "floor dAUROC", "S-loc dAUROC", "B3gs dAUROC")
                for b in ("network", "hospital")]
        C.row(tab, cells[0].label, cols, toks(cells[1:]), vals)


def t_rf(tex, R, C):
    tab = "Table 8 (LR / RF)"
    head, rows = table(tex, "tab:rf")
    C.header(tab, head, "eICU death & MIMIC-IV death & MIMIC-IV readmission", "LR / RF & LR / RF & LR / RF")
    for cells in rows:
        meth = release_method(cells[0].label)[0]
        lv = cells[1].label.replace("\\ ", " ")
        hier, L = ("trunc", 2) if lv.startswith("trunc.") else ("official", LEVEL[lv])
        for cell, (db, oc) in zip(cells[2:], SETTINGS):
            if not cell.toks:
                continue
            key = (db, oc, f"{db}_{hier}{L}_m3_K5_{meth}")
            C.row(tab, f"{cells[0].label} | {cells[1].label} | {db} {oc}", ["LR", "RF"], cell.toks,
                  [field(R.arm, "utility_arms", key, "d_raw"), field(R.armrf, "utility_arms_rf", key, "d_raw")])


def t_match(tex, R, C):
    tab = "Table 9 (matched control)"
    head, rows = table(tex, "tab:match")
    C.header(tab, head, "Setting & Level & Floor & Control & Floor $-$ control")
    for cells in rows:
        s = re.match(r"(eICU|MIMIC-IV) (death|readmission), \$m=(\d)\$", cells[0].label)
        db, oc, m = DB[s.group(1)], OUTCOME[s.group(2)], int(s.group(3))
        key = f"{db}_official{LEVEL[cells[1].label]}_m{m}_K5_floor"
        vals = [field(R.armm, "utility_arms_match", (db, oc, key), "d_raw"),
                field(R.armm, "utility_arms_match", (db, oc, key + ":randmatch"), "d_raw")]
        vals += [field(R.pairm, "utility_pairs_match", (db, oc, key, key + ":randmatch"), f) for f in ("d", "lo", "hi")]
        lab = f"{cells[0].label} | {cells[1].label}"
        C.row(tab, lab, ["Floor", "Control", "difference", "CI low", "CI high"], toks(cells[2:]), vals)
        C.same(tab, lab, "floor dAUROC in utility_arms vs utility_arms_match",
               R.arm[(db, oc, key)]["d_raw"], R.armm[(db, oc, key)]["d_raw"], 3)


def setting_rows(rows):
    cur = None
    for cells in rows:
        s = re.match(r"(eICU|MIMIC-IV), (category|block|chapter)", cells[0].label)
        if s:
            cur = (DB[s.group(1)], LEVEL[s.group(2)])
        yield cur, cells


def t_queries(tex, R, C):
    tab = "Table 10 (count queries)"
    head, rows = table(tex, "tab:queries")
    C.header(tab, head, "Setting & Release & single MARE & support lost & pair MARE & pair support lost")
    for (db, L), cells in setting_rows(rows):
        meth = release_method(cells[1].label)[0]
        key = f"{db}_official{L}_m3_K5_{meth}"
        vals = [field(R.ra, "reaudit", key, f"q{L}_{f}", 100)
                for f in ("single_mare", "single_support_lost", "pair_mare", "pair_support_lost")]
        C.row(tab, f"{db} L{L} | {cells[1].label}", ["single MARE", "support lost", "pair MARE", "pair support lost"],
              toks(cells[2:]), vals)


def t_leaks(tex, R, C):
    tab = "Table 11 (leaks)"
    head, rows = table(tex, "tab:leaks")
    C.header(tab, head, "Setting & Release & viol.\\ (band) & unique (band) & viol.\\ (length strata) & visible removal")
    for (db, L), cells in setting_rows(rows):
        meth = release_method(cells[1].label)[0]
        key = f"{db}_official{L}_m3_K5_{meth}"
        vis = "refused_no_candidate_share" if meth == "floor" else "suppressed_no_candidate_share"
        lab = f"{db} L{L} | {cells[1].label}"
        C.row(tab, lab, ["violators (band)"], cells[2].toks, [field(R.ra, "reaudit", key, "lenaud_violator_share", 100)])
        C.row(tab, lab, ["unique (band)"], cells[3].toks, [field(R.ra, "reaudit", key, "lenaud_unique_share", 100)])
        if cells[4].toks:
            C.row(tab, lab, ["violators (length strata)"], cells[4].toks,
                  [field(R.ra, "reaudit", f"{db}_official{L}_m3_K5_len_{meth}", "lenaud_violator_share", 100)])
        C.row(tab, lab, ["visible removal"], cells[5].toks, [field(R.ra, "reaudit", key, vis, 100)])


# -------------------------------------------------------------------------- text claims ----
def text_claims(tex, R, C):
    flat = " ".join(tex.split())
    mid = ["20s", "30s", "40s", "50s", "60s", "70s", "80s"]

    def report(name, found, printed, computed, ok):
        C.text.append((name, found, printed, computed, ok))

    def rng(key, pref):
        v = [100 * R.ra[key][pref + b] for b in mid]
        return min(v), max(v)

    def claim_range(name, rx, lo, hi, detail):
        m = re.search(rx, flat)
        if not m:
            return report(name, False, "", detail, None)
        a, b = float(m.group(1)), float(m.group(2))
        report(name, True, f"{m.group(1)}-{m.group(2)}", detail, round(lo) == a and round(hi) == b)

    lo, hi = rng("eicu_official2_m3_K5_floor", "refused_band_")
    claim_range("refusal by age, eICU block floor, bands 20s-80s",
                r"against (\d+)--(\d+)\\,\\% in the (?:middle bands\. The|bands from twenty to eighty-nine)",
                lo, hi, f"{lo:.1f}-{hi:.1f} %")
    lo, hi = rng("eicu_official2_m3_K5_slocL", "removed_band_")
    m2 = re.search(r"against (\d+)--(\d+)\\,\\% in the twenties and thirties and (\d+)--(\d+)\\,\\% from forty "
                   r"to eighty-nine", flat)
    if m2:
        v = {b: 100 * R.ra["eicu_official2_m3_K5_slocL"]["removed_band_" + b] for b in mid}
        young, old = [v["20s"], v["30s"]], [v[b] for b in mid[2:]]
        want = [round(min(young)), round(max(young)), round(min(old)), round(max(old))]
        report("S-loc removal by age, eICU block, 20s-30s and 40s-80s", True, "-".join(m2.groups()),
               f"{min(young):.1f}-{max(young):.1f} % and {min(old):.1f}-{max(old):.1f} %",
               [int(x) for x in m2.groups()] == want)
    else:
        claim_range("S-loc removal by age, eICU block, bands 20s-80s",
                    r"against (\d+)--(\d+)\\,\\% in the middle bands, because", lo, hi, f"{lo:.1f}-{hi:.1f} %")
    # price of suppression inside the hybrids
    worst = {}
    for db, oc in SETTINGS:
        for me in ("b3s", "b3gs"):
            base = R.arm[(db, oc, f"{db}_official2_m3_K5_{me}")]["d_raw"]
            for w in ("0.25", "0.5", "2", "4"):
                d = abs(R.arm[(db, oc, f"{db}_official2_m3_K5_{me}_w{w}")]["d_raw"] - base)
                worst[db] = max(worst.get(db, 0), d)
    det = ", ".join(f"{d}: max |d(w) - d(1)| = {v:.4f}" for d, v in sorted(worst.items()))
    m2 = re.search(r"the choice does not matter on MIMIC-IV \(every cell within (0\.\d+) AUROC\) and \$w=0\.25\$ "
                   r"gains (0\.\d+) to (0\.\d+) on eICU", flat)
    m = re.search(r"at block level the choice does not matter \(every cell within (0\.\d+) AUROC\)", flat)
    if m2:
        g = sorted(R.arm[("eicu", "death", f"eicu_official2_m3_K5_{me}_w0.25")]["d_raw"]
                   - R.arm[("eicu", "death", f"eicu_official2_m3_K5_{me}")]["d_raw"] for me in ("b3s", "b3gs"))
        report("w sweep at block level (MIMIC-IV within, eICU gain at w = 0.25)", True, "/".join(m2.groups()),
               det + f"; eICU gains {g[0]:.4f} and {g[1]:.4f}",
               worst["mimic"] <= float(m2.group(1)) + 0.0005 and round(g[0], 3) == float(m2.group(2))
               and round(g[1], 3) == float(m2.group(3)))
    else:
        report("w sweep at block level", bool(m), m.group(1) if m else "", det,
               None if not m else all(v <= float(m.group(1)) + 0.0005 for v in worst.values()))
    best = {}
    for db, oc in (("eicu", "death"), ("mimic", "death")):
        ds = {me: R.arm[(db, oc, f"{db}_official3_m3_K5_{me}_w0.25")]["d_raw"] for me in ("b3s", "b3gs")}
        best[db] = max(ds, key=ds.get) + " (" + ", ".join(f"{k} {v:+.4f}" for k, v in ds.items()) + ")"
    det = "; ".join(f"{d}: {v}" for d, v in best.items())
    m2 = re.search(r"B3s on MIMIC-IV and B3gs on eICU \(\$(-0\.\d+)\$\)", flat)
    m = re.search(r"With \$w=0\.25\$, B3s becomes the best valid chapter-level release on both databases", flat)
    if m2:
        e = R.arm[("eicu", "death", "eicu_official3_m3_K5_b3gs_w0.25")]["d_raw"]
        report("best chapter-level release at w = 0.25", True, f"B3s on MIMIC-IV, B3gs on eICU ({m2.group(1)})", det,
               best["mimic"].startswith("b3s ") and best["eicu"].startswith("b3gs ") and round(e, 3) == float(m2.group(1)))
    else:
        report("best chapter-level release at w = 0.25", bool(m), "B3s on both databases" if m else "", det,
               None if not m else all(v.startswith("b3s ") for v in best.values()))
    keys = [("eicu", 2), ("eicu", 1), ("mimic", 2), ("mimic", 3)]          # the rows of Table 11
    for meth, pat, name in (("floor", r"show (\d+)--(\d+)\\,\\% violators for the floor", "floor"),
                            ("slocL", r"violators for the floor and (\d+)--(\d+)\\,\\% for S-loc", "S-loc")):
        v = [100 * R.ra[f"{db}_official{L}_m3_K5_{meth}"]["lenaud_violator_share"] for db, L in keys]
        claim_range(f"length-augmented violators, {name}, rows of Table 11", pat, min(v), max(v),
                    f"{min(v):.1f}-{max(v):.1f} %")
    d1 = max(abs(r["d_raw"] - R.arm[("eicu", "death_day1", a)]["d_raw"])
             for (db, oc, a), r in R.arm.items() if db == "eicu" and oc == "death" and ("eicu", "death_day1", a) in R.arm)
    m = re.search(r"follows in-hospital death within (0\.\d+) AUROC in every setting", flat)
    report("first-day outcome vs in-hospital death", bool(m), m.group(1) if m else "", f"largest difference {d1:.4f}",
           None if not m else d1 <= float(m.group(1)) + 1e-9)
    rat = {}
    for L in (1, 3):
        h = R.rel[("eicu", "official", L, 3, 5, "hospital", "slocL")]["entries_removed_share"]
        n = R.rel[("eicu", "official", L, 3, 5, "band", "slocL")]["entries_removed_share"]
        rat[L] = h / n
    m = re.search(r"\((\d+) times more at category level, (\d+) times more at chapter level\)", flat)
    report("S-loc removal, hospital / network", bool(m), f"{m.group(1)}x, {m.group(2)}x" if m else "",
           f"{rat[1]:.1f}x, {rat[3]:.1f}x",
           None if not m else round(rat[1]) == int(m.group(1)) and round(rat[3]) == int(m.group(2)))


def rounded_twice(tok, val):
    """The printed number, if it equals the result rounded as RESULTS.md prints it and then again."""
    inter = Decimal(f"{val.value:.{2 if val.scale == 100 else 4}f}")
    if inter.quantize(Decimal(1).scaleb(-tok.dec), rounding=ROUND_HALF_UP) == Decimal(tok.text.replace(",", "")):
        return str(inter)
    return None


# --------------------------------------------------------------------------------- main ----
def run(tex, R, with_text=True):
    C = Checker()
    steps = [("Table 2", lambda: t_cohorts(tex, R, C)), ("Table 3", lambda: t_validity(tex, R, C)),
             ("Table 4", lambda: t_floor(tex, R, C)),
             ("Table 5", lambda: t_cost(tex, R, C, "tab:cost", "Table 5 (cost, m = 3)", 3)),
             ("Table 6", lambda: t_k(tex, R, C)), ("Table 7", lambda: t_hosp(tex, R, C)),
             ("Table 8", lambda: t_rf(tex, R, C)), ("Table 9", lambda: t_match(tex, R, C)),
             ("Table 10", lambda: t_queries(tex, R, C)), ("Table 11", lambda: t_leaks(tex, R, C)),
             ("Table S1", lambda: t_cost(tex, R, C, "tab:cost2", "Table S1 (cost, m = 2)", 2))]
    if with_text:
        steps.append(("text", lambda: text_claims(tex, R, C)))
    for name, fn in steps:
        try:
            fn()
        except (KeyError, ValueError, AttributeError, IndexError, TypeError) as e:
            C.struct.append(f"{name}: stopped, {type(e).__name__}: {e}")
    return C


def coverage(tex):
    """Numbers printed in the data cells of the checked tables (label columns excluded)."""
    n = {}
    for lab, skip in (("tab:cohorts", 1), ("tab:validity", 2), ("tab:floor", 2), ("tab:cost", 2), ("tab:k", 2),
                      ("tab:hosp", 1), ("tab:rf", 2), ("tab:match", 2), ("tab:queries", 2), ("tab:leaks", 2),
                      ("tab:cost2", 2)):
        _, rows = table(tex, lab)
        c = 0
        for cells in rows:
            if len(cells) == 1:                                 # section header: the raw AUROC
                c += len(re.findall(r"raw AUROC \d\.\d+", cells[0].orig))
            elif "Attack" in cells[0].orig:
                c += len(toks(cells[1:]))
            else:
                c += len(toks(cells[skip:]))
        n[lab] = c
    return n


def selftest(tex, R):
    base = run(tex, R, with_text=False)
    b = [r["status"] for r in base.rec]
    bad_print = bad_other = 0
    for i, r in enumerate(base.rec):
        t, v = r["tok"], r["val"].value
        if r["status"] == "MISMATCH":
            new, want = t.render(round(v, t.dec)), "OK"         # set to the right value: must pass
        else:
            step = 10 ** (-t.dec)
            new, want = t.render(t.value + (step if t.value >= v else -step)), "MISMATCH"
        mut = run(tex[:t.off] + new + tex[t.off + len(t.text):], R, with_text=False)
        s = [x["status"] for x in mut.rec]
        if len(s) != len(b) or mut.struct != base.struct:
            bad_other += 1
            continue
        got = "OK" if s[i] in ("OK", "CONST") else "MISMATCH"
        bad_print += got != want
        bad_other += any(s[j] != b[j] for j in range(len(b)) if j != i)
    refs, bad_ref = {}, 0
    for i, r in enumerate(base.rec):
        if r["val"].ref is not None:
            refs.setdefault((id(r["val"].ref[0]), r["val"].ref[1]), (r["val"].ref, []))[1].append(i)
    for (row, fld), idx in refs.values():
        old = row[fld]
        row[fld] = old * 1.5 + 0.37
        s = [x["status"] for x in run(tex, R, with_text=False).rec]
        row[fld] = old
        bad_ref += any(s[i] != "MISMATCH" for i in idx if b[i] != "MISMATCH")
    direct = sum(len(v[1]) for v in refs.values())
    print(f"self-test: {len(base.rec)} printed numbers changed one at a time: "
          f"{bad_print} not detected, {bad_other} runs with a side effect on another cell")
    print(f"self-test: {len(refs)} result values perturbed one at a time (they feed {direct} cells): "
          f"{bad_ref} not detected; {len(base.rec) - direct} cells use a maximum or a constant and are not perturbed")
    return bad_print == 0 and bad_other == 0 and bad_ref == 0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        sys.exit(__doc__)
    tex, R = Path(args[0]).read_text(), Results(args[1])
    if "\\chgm{" in tex or "\\chgdel{" in tex:
        tex = strip_changes(tex)
        print("change markup found: checking the clean text (new wording, deleted text ignored)\n")
    C = run(tex, R)
    cov = coverage(tex)
    by = {}
    for r in C.rec:
        by.setdefault(r["tab"], []).append(r)
    print(f"{'table':28s} {'printed':>8s} {'checked':>8s} {'match':>6s} {'const':>6s} {'MISMATCH':>9s}")
    labs = list(cov)
    for tab, lab in zip(by, labs):
        rs = by[tab]
        print(f"{tab:28s} {cov[lab]:8d} {len(rs):8d} {sum(r['status'] == 'OK' for r in rs):6d} "
              f"{sum(r['status'] == 'CONST' for r in rs):6d} {sum(r['status'] == 'MISMATCH' for r in rs):9d}")
    tot, chk = sum(cov.values()), len(C.rec)
    mism = [r for r in C.rec if r["status"] == "MISMATCH"]
    print(f"{'all tables':28s} {tot:8d} {chk:8d} {sum(r['status'] == 'OK' for r in C.rec):6d} "
          f"{sum(r['status'] == 'CONST' for r in C.rec):6d} {len(mism):9d}")
    if tot != chk:
        print(f"!! {tot - chk} printed numbers were not checked")
    for title, items in (("structure problems (rows or cells the script could not map)", C.struct),
                         ("consistency notes (two results printed as one, or claimed identical)", C.extra)):
        print(f"\n{title}: {len(items)}")
        for s in items:
            print("  -", s)
    twice = [rounded_twice(r["tok"], r["val"]) for r in mism]
    zeros = sum(r["tok"].value == 0 for r in C.rec)
    print(f"\nmismatches in tables: {len(mism)}, of which rounded twice: {sum(x is not None for x in twice)} "
          f"({zeros} of the {chk} checked numbers are printed zeros)")
    for r, tw in zip(mism, twice):
        t, v = r["tok"], r["val"]
        why = f"rounded twice, via {tw}" if tw else "NOT explained by rounding"
        print(f"  - {r['tab']} | {r['row']} | {r['col']}: printed {t.text}, result {v.value:.5f} "
              f"-> {t.render(round(v.value, t.dec))} ({why})   [{v.src}]")
    print("\nsentences of the running text:")
    for name, found, printed, computed, ok in C.text:
        state = "sentence not found (reworded?)" if not found else ("agrees" if ok else "DIFFERS")
        print(f"  - {name}: printed {printed or '-'}; results: {computed}  => {state}")
    if "--verbose" in sys.argv:
        print("\nall checked cells:")
        for r in C.rec:
            print(f"  {r['status']:8s} {r['tab']} | {r['row']} | {r['col']}: {r['tok'].text} vs {r['val'].value:.5f} [{r['val'].src}]")
    ok = True
    if "--selftest" in sys.argv:
        print()
        ok = selftest(tex, R)
    sys.exit(0 if ok and not mism and not C.struct and tot == chk else 1)


if __name__ == "__main__":
    main()
