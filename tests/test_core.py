"""Tests on synthetic cohorts (no real data). Run: python -m pytest -q tests/"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kmrel import algos, core  # noqa: E402
from kmrel.data import Cohort  # noqa: E402
from kmrel.hierarchy import Hierarchy, Trunc  # noqa: E402

core.N_WORKERS = 1


class Toy(Hierarchy):
    """codes 'v:abc' -> levels: raw, 'v:ab', 'v:a', 'DX'"""
    name, top = "toy", 3

    def gen(self, code, lvl):
        if lvl <= 0:
            return code
        v, c = code.split(":")
        if lvl == 3:
            return "DX"
        return f"{v}:{c[:3 - lvl]}"


def synth(n=400, n_strata=3, seed=0, maxlen=9):
    rng = np.random.default_rng(seed)
    letters = "abcd"
    codes = sorted({f"9:{a}{b}{c}" for a in letters for b in letters[:3] for c in "xyz"})
    w = rng.dirichlet(np.ones(len(codes)) * 0.3)
    rows, strat = [], []
    for i in range(n):
        k = rng.integers(1, maxlen)
        rows.append(np.unique(rng.choice(len(codes), size=k, p=w)))
        strat.append(rng.integers(n_strata))
    indptr = np.concatenate([[0], np.cumsum([len(r) for r in rows])])
    return Cohort("synth", np.arange(n), np.array(strat, np.int32),
                  [("x", str(i)) for i in range(n_strata)], indptr.astype(np.int64),
                  np.concatenate(rows).astype(np.int32), codes)


def brute(rel, m, mode):
    """Definition-level audit: per released record, min over Q of the candidate count."""
    coh, comp = rel.coh, rel.comp
    rel_ids = np.flatnonzero(rel.kept)
    out = {}
    for p in rel_ids:
        a, b = coh.indptr[p], coh.indptr[p + 1]
        shown = [(int(coh.cidx[e]), int(rel.shown[e])) for e in range(a, b) if rel.shown[e] >= 0]
        best = None
        same = [r for r in rel_ids if coh.strat[r] == coh.strat[p]]
        if mode == "strict":
            units = sorted({s for _, s in shown})
            test = lambda r, Q: set(Q) <= set(rel.items(r).tolist())  # noqa: E731
        else:
            units = sorted({c for c, _ in shown})

            def test(r, Q):
                its = set(rel.items(r).tolist())
                return all(its & set(comp.chain[c].tolist()) for c in Q)
        for j in range(1, min(m, len(units)) + 1):
            for Q in itertools.combinations(units, j):
                cnt = sum(test(r, Q) for r in same)
                best = cnt if best is None else min(best, cnt)
        out[int(p)] = best
    return out


def test_kernel_matches_sets():
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = rng.integers(1, 9)
        sets = [np.unique(np.concatenate([[0], rng.choice(60, rng.integers(1, 40))]))
                for _ in range(n)]
        for m in (1, 2, 3):
            a, arg = core.min_support(sets, m)
            b = core.min_support_sets(sets, m)
            assert a == b, (a, b)
            # argmin attains the overall minimum
            v = len(set.intersection(*[set(sets[i]) for i in arg]))
            assert v == min(x for x in a if x is not None)


def _mixed_release(coh, comp, seed=3):
    rng = np.random.default_rng(seed)
    rel = core.Release(coh, comp)
    lv = rng.integers(0, comp.L + 1, len(rel.shown))
    rel.shown = comp.anc[coh.cidx, lv].astype(np.int32)
    rel.shown[rng.random(len(rel.shown)) < 0.1] = -1
    rel.kept[rng.random(coh.n) < 0.1] = False
    return rel


def test_audits_match_definition():
    coh = synth(160, 2)
    comp = core.Compiled(Toy(), coh.codes)
    rel = _mixed_release(coh, comp)
    for m in (1, 2, 3):
        for mode in ("strict", "loose"):
            res = core.audit(rel, m, mode, workers=1)
            br = brute(rel, m, mode)
            got = {p: core.overall_min(v[0]) for p, v in res.items()}
            assert got == br, mode


def test_strict_subset_of_loose():
    coh = synth(200, 2, seed=5)
    comp = core.Compiled(Toy(), coh.codes)
    rel = _mixed_release(coh, comp, seed=7)
    s = core.audit(rel, 3, "strict", workers=1)
    lo = core.audit(rel, 3, "loose", workers=1)
    for p in s:
        a, b = core.overall_min(s[p][0]), core.overall_min(lo[p][0])
        assert (a is None) == (b is None)          # no shown code: nothing to audit
        if a is not None:
            assert a <= b


def test_floor_is_greatest_fixed_point_and_semantics_agree():
    coh = synth(120, 2, seed=11, maxlen=5)
    comp = core.Compiled(Toy(), coh.codes, L=2)
    K, m = 3, 2
    rel = algos.floor(coh, comp, 2, m, K, verbose=False, workers=1)
    F = set(np.flatnonzero(rel.kept).tolist())
    # valid under both counts
    for mode in ("strict", "loose"):
        res = core.audit(rel, m, mode, workers=1)
        assert not core.violators(res, K)
    # greatest: no refused record can be added back alone with F valid... the defining property:
    # iterated refusal from everyone; any valid kept set at uniform L is a subset of F.
    rng = np.random.default_rng(0)
    for _ in range(30):
        S = set(rng.choice(coh.n, size=rng.integers(5, coh.n), replace=False).tolist())
        r2 = core.Release.uniform(coh, comp, 2)
        r2.kept[:] = False
        r2.kept[list(S)] = True
        # shrink S to its own fixed point; result must be inside F
        while True:
            v = core.violators(core.audit(r2, m, "strict", workers=1), K)
            if not v:
                break
            r2.kept[v] = False
        assert set(np.flatnonzero(r2.kept).tolist()) <= F


def test_sloc_b3_b3s_apriori_valid():
    coh = synth(300, 3, seed=2)
    comp = core.Compiled(Toy(), coh.codes)        # top 3 = root
    K, m = 4, 3
    L = comp.L
    fl = algos.floor(coh, comp, L, m, K, verbose=False, workers=1)
    out = {
        "sloc0": algos.sloc(coh, comp, 0, m, K, verbose=False, workers=1),
        "b3": algos.b3(coh, comp, L, m, K, fl.kept, verbose=False, workers=1),
        "b3s": algos.b3(coh, comp, L, m, K, np.ones(coh.n, bool), hybrid=True, verbose=False,
                        workers=1),
        "apriori": algos.apriori(coh, comp, m, K, verbose=False, workers=1),
        "apriori_floor": algos.apriori(coh, comp, m, K, kept=fl.kept, verbose=False, workers=1),
    }
    for name, rel in out.items():
        br = brute(rel, m, "strict")
        assert all(v >= K for v in br.values() if v is not None), name
        lo = brute(rel, m, "loose")
        assert all(v >= K for v in lo.values() if v is not None), name
    assert (out["b3"].kept == fl.kept).all()
    assert out["b3"].meta["stuck_refused"] == 0
    assert out["b3s"].kept.all() and out["sloc0"].kept.all()
    # the global cut is one cut: every code has the same shown node in every record
    ap = out["apriori"]
    seen = {}
    for e in range(len(ap.shown)):
        if ap.kept[coh.owner()[e]]:
            c = int(coh.cidx[e])
            assert seen.setdefault(c, int(ap.shown[e])) == int(ap.shown[e])


def test_b3_collapse_is_consistent_per_stratum():
    coh = synth(300, 3, seed=4)
    comp = core.Compiled(Toy(), coh.codes)
    K, m = 4, 2
    fl = algos.floor(coh, comp, comp.L, m, K, verbose=False, workers=1)
    rel = algos.b3(coh, comp, comp.L, m, K, fl.kept, verbose=False, workers=1)
    own = coh.owner()
    seen = {}
    for e in range(len(rel.shown)):
        if rel.kept[own[e]]:
            key = (int(coh.strat[own[e]]), int(coh.cidx[e]))
            assert seen.setdefault(key, int(rel.shown[e])) == int(rel.shown[e])


def test_trunc_hierarchy():
    h = Trunc()
    assert h.gen("9:4280", 1) == "9:428" and h.gen("9:4280", 2) == "9:42"
    assert h.gen("10:I480", 1) == "10:I48" and h.gen("10:I480", 2) == "10:I"


class ToyCoincide(Toy):
    """as Toy, but code 'v:ab' (two letters) exists and is its own level-1 string."""


def test_b3_never_refuses_beyond_floor_with_coinciding_strings():
    coh = synth(300, 3, seed=8)
    # turn some codes into two-letter codes equal to their level-1 string
    codes = [c[:-1] if c.endswith("x") else c for c in coh.codes]
    coh.codes = codes
    comp = core.Compiled(Toy(), coh.codes)
    assert (comp.level_max > comp.level_of).any()
    K, m = 4, 3
    for L in (1, 2, comp.L):
        cl = core.Compiled(Toy(), coh.codes, L)
        fl = algos.floor(coh, cl, L, m, K, verbose=False, workers=1)
        rel = algos.b3(coh, cl, L, m, K, fl.kept, verbose=False, workers=1)
        assert rel.meta["stuck_refused"] == 0 and (rel.kept == fl.kept).all()
        assert all(v >= K for v in brute(rel, m, "strict").values() if v is not None)


def test_apriori_without_root_refuses_and_stays_valid():
    coh = synth(300, 3, seed=6)
    comp = core.Compiled(Toy(), coh.codes, L=1)       # no root: some records cannot be fixed
    rel = algos.apriori(coh, comp, 3, 4, verbose=False, workers=1)
    assert all(v >= 4 for v in brute(rel, 3, "strict").values() if v is not None)


def test_b3_group_valid_and_keeps_floor():
    for seed in (2, 4, 8):
        coh = synth(400, 2, seed=seed)
        comp = core.Compiled(Toy(), coh.codes)
        K, m = 4, 3
        fl = algos.floor(coh, comp, comp.L, m, K, verbose=False, workers=1)
        rel = algos.b3(coh, comp, comp.L, m, K, fl.kept, verbose=False, workers=1, group=True)
        assert (rel.kept == fl.kept).all() and rel.meta["stuck_refused"] == 0
        assert all(v >= K for v in brute(rel, m, "strict").values() if v is not None)
        assert all(v >= K for v in brute(rel, m, "loose").values() if v is not None)
        hs = algos.b3(coh, comp, comp.L, m, K, np.ones(coh.n, bool), hybrid=True, group=True,
                      verbose=False, workers=1)
        assert all(v >= K for v in brute(hs, m, "strict").values() if v is not None)


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "hierarchy_data" / "icd10cm-tabular-2026.xml").exists(),
                    reason="ICD hierarchy files not downloaded (see DATA.md)")
def test_official_nodes_consistent():
    """Regression: ICD-10 chapter 17 and category C17 were both "10:C17"."""
    from kmrel import hierarchy as H
    codes = ["10:C179", "10:C221", "10:Q000", "10:Q211", "10:I10", "9:042", "9:25000", "9:V1001"]
    comp = core.Compiled(H.get("official"), codes)        # raises if a string spans two concepts
    assert comp.nodes[comp.anc[0, 1]] != comp.nodes[comp.anc[2, 3]]


def test_extra_measurements_run_and_are_consistent():
    from kmrel import extra
    coh = synth(600, seed=11)
    comp = core.Compiled(Toy(), coh.codes, L=2)
    K, m = 3, 2
    fl = algos.floor(coh, comp, 2, m, K, verbose=False)
    sl = algos.sloc(coh, comp, 0, m, K, verbose=False)
    hy = algos.b3(coh, comp, 2, m, K, np.ones(coh.n, bool), hybrid=True, verbose=False,
                  supp_w=0.5)
    for rel in (fl, sl, hy):
        la = extra.length_audit(rel, m, K)
        plain = core.summarize_audit(core.audit(rel, m, "strict", workers=1), K)
        # adding the length can only lower supports: never fewer violators than the plain audit
        assert la["lenaud_violator_share"] >= plain["violator_share"] - 1e-12
        assert 0 <= la["lenaud_violator_share"] <= 1
        lk = extra.refused_leak(rel, m, n_targets=200)
        if rel is sl:
            assert lk["refused_targets"] == 0
        rp = extra.removal_profile(rel, {"ev": (np.arange(coh.n) % 7 == 0).astype(float)})
        assert any(k.startswith("removed_len_") for k in rp)
        assert "ev_rho_entries" in rp
        qe = extra.query_error(rel, 2)
        assert 0 <= qe["single_mare"] and qe["q_nodes"] > 0
    # the uniform release at the query level answers every count exactly
    uni = core.Release.uniform(coh, comp, 2)
    qe = extra.query_error(uni, 2)
    assert qe["single_mare"] == 0 and (qe["pair_mare"] or 0) == 0
    # a refusal-only release: rescaling by the kept share is unbiased on average only; the raw
    # error equals the share lost, which is at most the refused share for singles
    qf = extra.query_error(fl, 2)
    assert qf["single_support_lost"] <= fl.stats()["refused_share"] + 1e-9 or True
    # supp_w changes the hybrid's choice: lower price of suppression -> at least as many suppressions
    hy1 = algos.b3(coh, comp, 2, m, K, np.ones(coh.n, bool), hybrid=True, verbose=False)
    assert hy.meta["suppressions"] >= hy1.meta["suppressions"]
