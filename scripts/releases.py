"""Build, cache and evaluate releases. Aggregates only.

  python scripts/releases.py --db eicu --hier official --L 2 --m 3 --K 5 \
      --methods floor,sloc0,slocL,b3,b3s,apriori,apriori_floor [--frac 0.1] [--import-earlier]

For every release: refusal, coding (share of entries at each level, suppressed), the strict audit
and the loose audit of every released record (violators, unique, smallest violating size), the
length leak, and the most-specific-match attack. --import-earlier also imports the cached states of the
two earlier local methods A and B2 (trunc hierarchy) as releases A_import and b2_import.
Per-patient states: ~/km/cache/<key>.npz (never printed).
"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from common import CACHE, guarded, out
from kmrel import algos, attack, core, data, extra, hierarchy as H

EARLIER = Path.home() / "km" / "earlier_states"


def load_or(key, build, coh, comp):
    f = CACHE / f"{key}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=False)
        rel = core.Release(coh, comp, kept=z["kept"], shown=z["shown"], name=str(z["name"]))
        rel.meta = dict(zip(z["meta_k"].tolist(), z["meta_v"].tolist())) if "meta_k" in z else {}
        rel.meta["cached"] = 1
        return rel
    rel = build()
    meta = getattr(rel, "meta", {})
    np.savez(f, kept=rel.kept, shown=rel.shown, name=rel.name,
             meta_k=np.array(list(meta.keys()), dtype=str),
             meta_v=np.array([float(v) for v in meta.values()]))
    return rel


def earlier_state(db, m, which):
    if db == "mimic":
        f = EARLIER / "mimic" / "cache" / (f"rev3_b2_K5_m{m}.pkl" if which == "b2"
                                         else f"rev3_levels_A_K5_m{m}.pkl")
    else:
        f = EARLIER / "eicu" / "cache" / (f"rev2_B2_recruit_K5_m{m}.pkl" if which == "b2"
                                        else f"levels_K5_m{m}.pkl")
    if not f.exists():
        return None
    st = pickle.loads(f.read_bytes())
    if isinstance(st, dict):
        LV, KEPT = st.get("LV"), st.get("kept", st.get("KEPT"))
    else:
        LV, KEPT = st[0], st[1]
    return LV, KEPT


def from_earlier(coh, comp, LV, KEPT, name):
    """An imported state (LV pid -> {code: level}, KEPT) on the trunc hierarchy -> Release."""
    pos = {int(p): i for i, p in enumerate(coh.pids)}
    kept = np.zeros(coh.n, bool)
    ks = KEPT if isinstance(KEPT, (set, frozenset)) else {p for p, v in KEPT.items() if v}
    for p in ks:
        if int(p) in pos:
            kept[pos[int(p)]] = True
    cid = {c: i for i, c in enumerate(coh.codes)}
    shown = comp.anc[coh.cidx, 0].copy()
    miss = 0
    for p, lv in LV.items():
        i = pos.get(int(p))
        if i is None:
            continue
        a, b = coh.indptr[i], coh.indptr[i + 1]
        loc = {int(c): e for e, c in zip(range(a, b), coh.cidx[a:b])}
        for code, l in lv.items():
            k = code if ":" in code else "9:" + code.replace(".", "")
            c = cid.get(k)
            if c is None or c not in loc:
                miss += 1
                continue
            shown[loc[c]] = comp.anc[c, min(int(l), comp.L)]
    rel = core.Release(coh, comp, kept=kept, shown=shown, name=name)
    rel.meta = dict(import_missing_codes=miss)
    return rel


def outcome_events(db, coh):
    """Per patient: outcome events (for the removal profile)."""
    if db == "mimic":
        adm = data.mimic_admissions(coh)
        alive = adm[adm.hospital_expire_flag == 0]
        return {"readmit30": np.bincount(alive.pidx, weights=alive.readmit30, minlength=coh.n),
                "death": np.bincount(adm.pidx, weights=adm.death, minlength=coh.n)}
    st = data.eicu_stays(coh)
    return {"death": np.bincount(st.pidx, weights=st.death, minlength=coh.n)}


def evaluate(rel, m, K, do_attack=True, ev=None, do_extra=True):
    coh = rel.coh
    row = rel.stats()
    t = time.time()
    s = core.audit(rel, m, "strict")
    row.update({f"strict_{k}": v for k, v in core.summarize_audit(s, K, coh.lengths()).items()})
    row["strict_audit_s"] = time.time() - t
    t = time.time()
    lo = core.audit(rel, m, "loose")
    row.update({f"loose_{k}": v for k, v in core.summarize_audit(lo, K, coh.lengths()).items()})
    row["loose_audit_s"] = time.time() - t
    row.update(core.length_leak(rel, K))
    if do_attack:
        row.update(attack.attack(rel, m))
    if do_extra:
        t = time.time()
        row.update(extra.length_audit(rel, m, K))
        row["lenaud_s"] = time.time() - t
        row.update(extra.refused_leak(rel, m))
        row.update(extra.removal_profile(rel, ev))
        for ql in range(2, min(rel.comp.L, 3) + 1):
            row.update({f"q{ql}_{k}": v for k, v in extra.query_error(rel, ql).items()
                        if k != "q_level"})
    row.update({f"meta_{k}": v for k, v in getattr(rel, "meta", {}).items()})
    return row


def restratify(coh, labels):
    uniq = sorted(set(labels))
    ix = {l: i for i, l in enumerate(uniq)}
    extra_ = dict(coh.extra)
    extra_["sex"] = np.array([coh.strata[s][0] for s in coh.strat])
    extra_["band"] = np.array([coh.strata[s][1] for s in coh.strat])
    return data.Cohort(coh.name, coh.pids, np.array([ix[x] for x in labels], np.int32), uniq,
                       coh.indptr, coh.cidx, coh.codes, extra_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--hier", default="official")
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--methods", default="floor,sloc0,slocL,b3,b3s,apriori,apriori_floor")
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--import-earlier", action="store_true")
    ap.add_argument("--no-attack", action="store_true")
    ap.add_argument("--strata", default="band", choices=["band", "hospital", "length"],
                    help="hospital (eICU): strata = hospital x sex x band, i.e. every hospital "
                         "releases alone; the attacker knows the hospital. length: strata = "
                         "sex x band x history-length band (the length becomes a quasi-identifier)")
    ap.add_argument("--supp-w", type=float, default=1.0,
                    help="hybrids: price of suppressing an entry relative to NCP loss (1 = default)")
    ap.add_argument("--no-extra", action="store_true")
    a = ap.parse_args()
    coh = data.load(a.db)
    if a.frac < 1:
        rng = np.random.default_rng(20261106)
        coh = coh.subset(rng.choice(coh.n, int(round(a.frac * coh.n)), replace=False))
    if a.strata == "hospital":
        assert a.db == "eicu" and "apriori" not in a.methods, "per-hospital: eICU, no global cut"
        hosp = coh.extra["hospital"]
        coh = restratify(coh, [(int(h),) + tuple(coh.strata[s]) for h, s in zip(hosp, coh.strat)])
    elif a.strata == "length":
        lb = [extra.len_band(int(n)) for n in coh.lengths()]
        coh = restratify(coh, [tuple(coh.strata[s]) + (b,) for s, b in zip(coh.strat, lb)])
    comp = core.Compiled(H.get(a.hier), coh.codes, a.L)
    tag = (f"{a.db}_{a.hier}{a.L}_m{a.m}_K{a.K}" + (f"_f{a.frac}" if a.frac < 1 else "")
           + {"hospital": "_hosp", "length": "_len"}.get(a.strata, ""))
    wtag = f"_w{a.supp_w:g}" if a.supp_w != 1.0 else ""
    base = dict(db=a.db, hierarchy=a.hier, L=a.L, m=a.m, K=a.K, frac=a.frac, cohort=coh.n,
                strata=a.strata, supp_w=a.supp_w, n_strata=len(coh.strata))
    ev = None if a.no_extra else outcome_events(a.db, coh)
    fl = load_or(f"{tag}_floor", lambda: algos.floor(coh, comp, a.L, a.m, a.K), coh, comp)
    builders = {
        "floor": lambda: fl,
        "sloc0": lambda: algos.sloc(coh, comp, 0, a.m, a.K),
        "slocL": lambda: algos.sloc(coh, comp, a.L, a.m, a.K),
        "b3": lambda: algos.b3(coh, comp, a.L, a.m, a.K, fl.kept),
        "b3s": lambda: algos.b3(coh, comp, a.L, a.m, a.K, np.ones(coh.n, bool), hybrid=True,
                                supp_w=a.supp_w),
        "b3g": lambda: algos.b3(coh, comp, a.L, a.m, a.K, fl.kept, group=True,
                                name=f"b3g{a.L}"),
        "b3gs": lambda: algos.b3(coh, comp, a.L, a.m, a.K, np.ones(coh.n, bool), hybrid=True,
                                 group=True, name=f"b3gs{a.L}", supp_w=a.supp_w),
        "apriori": lambda: algos.apriori(coh, comp, a.m, a.K),
        "apriori_floor": lambda: algos.apriori(coh, comp, a.m, a.K, kept=fl.kept,
                                               name=f"apriori_floor{a.L}"),
    }
    for meth in [x for x in a.methods.split(",") if x]:
        t0 = time.time()
        print(f"=== {tag} {meth}", flush=True)
        key = f"{tag}_{meth}" + (wtag if meth in ("b3s", "b3gs") else "")
        rel = load_or(key, builders[meth], coh, comp)
        rel.name = meth
        row = dict(base, method=meth + (wtag if meth in ("b3s", "b3gs") else ""),
                   build_s=time.time() - t0)
        row.update(evaluate(rel, a.m, a.K, not a.no_attack, ev, not a.no_extra))
        out("releases", [row])
    if a.import_earlier and a.hier == "trunc" and a.L == 2 and a.K == 5 and a.frac == 1:
        for which in ("A", "b2"):
            st = earlier_state(a.db, a.m, which)
            if st is None:
                print(f"no imported state for {which}", flush=True)
                continue
            rel = from_earlier(coh, comp, *st, name=f"{which}_import")
            np.savez(CACHE / f"{tag}_{which}_import.npz", kept=rel.kept, shown=rel.shown,
                     name=rel.name, meta_k=np.array([], dtype=str), meta_v=np.array([]))
            row = dict(base, method=f"{which}_import")
            row.update(evaluate(rel, a.m, a.K, not a.no_attack, ev, not a.no_extra))
            out("releases", [row])


if __name__ == "__main__":
    guarded(main)
