"""The refusal floor as a curve. Aggregates only.

  python scripts/floor_grid.py --db mimic --part levels|fractions|strata|hospital

levels     every level of the official tree (0 raw .. 4 root) and of the truncation tree
           (0..2), m in {2, 3}, K in {5, 10, 20}
fractions  seeded random sub-populations (1, 5, 10, 20, 50, 100 %), K = 5, three seeds below 100 %
strata     strata width: sex x 10-year band (default), sex x 20-year band, sex only,
           10-year band only; K = 5
hospital   eICU only: release per hospital (strata = hospital x sex x 10-year band) vs network;
           refusal by hospital-size bin (no hospital identifier is written)

Every floor row: refused share, refused share among records with >= m codes, and the share of
outcome events removed with the refused patients (MIMIC: 30-day readmission events on all
admissions, in-hospital deaths; eICU: in-hospital deaths of health-system stays), with the
selectivity ratio rho = events share / patients share.
"""
import argparse
import time

import numpy as np
import pandas as pd

from common import guarded, out
from kmrel import algos, core, data, hierarchy as H
from kmrel.data import Cohort


def outcome_counts(db, coh):
    """Per patient (cohort index): events per outcome."""
    if db == "mimic":
        adm = data.mimic_admissions(coh)
        ev = {}
        alive = adm[adm.hospital_expire_flag == 0]
        ev["readmit30"] = np.bincount(alive.pidx, weights=alive.readmit30, minlength=coh.n)
        ev["death"] = np.bincount(adm.pidx, weights=adm.death, minlength=coh.n)
        ev["admissions"] = np.bincount(adm.pidx, minlength=coh.n).astype(float)
    else:
        st = data.eicu_stays(coh)
        ev = {"death": np.bincount(st.pidx, weights=st.death, minlength=coh.n),
              "stays": np.bincount(st.pidx, minlength=coh.n).astype(float)}
    return ev


def floor_row(coh, comp, L, m, K, ev, **extra):
    t = time.time()
    rel = algos.floor(coh, comp, L, m, K, verbose=False)
    ref = ~rel.kept
    lens = coh.lengths()
    sh = float(ref.mean())
    row = dict(extra, L=L, level=comp.h.level_names[L], m=m, K=K, cohort=coh.n,
               refused_share=sh, refused_share_ge_m=float(ref[lens >= m].mean()),
               refused_share_le14=float(ref[lens <= 14].mean()),
               refused_share_gt14=float(ref[lens > 14].mean()) if (lens > 14).any() else None,
               rounds=rel.meta["rounds"], runtime_s=time.time() - t)
    if ev is not None:
        for k, v in ev.items():
            tot = v.sum()
            e = float(v[ref].sum() / tot) if tot else float("nan")
            row[f"{k}_removed_share"] = e
            row[f"{k}_rho"] = e / sh if sh else None
    return row, rel


def restrat(coh, labels_of):
    """New strata from a function (old label) -> new label."""
    new = [labels_of(lab) for lab in coh.strata]
    uniq = sorted(set(new))
    idx = {l: i for i, l in enumerate(uniq)}
    m = np.array([idx[x] for x in new], dtype=np.int32)
    return Cohort(coh.name, coh.pids, m[coh.strat], uniq, coh.indptr, coh.cidx, coh.codes,
                  coh.extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--part", required=True)
    a = ap.parse_args()
    coh = data.load(a.db)
    ev = outcome_counts(a.db, coh)
    comps = {"official": core.Compiled(H.get("official"), coh.codes),
             "trunc": core.Compiled(H.get("trunc"), coh.codes)}
    base = dict(db=a.db, part=a.part)
    if a.part == "levels":
        for hn, comp in comps.items():
            for K in (5, 10, 20):
                for m in (3, 2):
                    for L in range(comp.L + 1):
                        cl = core.Compiled(comp.h, coh.codes, L)
                        row, _ = floor_row(coh, cl, L, m, K, ev, hierarchy=hn, **base)
                        out("floor_levels", [row])
    elif a.part == "fractions":
        for frac in (0.01, 0.05, 0.1, 0.2, 0.5, 1.0):
            seeds = (1, 2, 3) if frac < 1 else (0,)
            for seed in seeds:
                rng = np.random.default_rng(20261106 + seed)
                idx = rng.choice(coh.n, int(round(frac * coh.n)), replace=False)
                sub = coh.subset(idx)
                evs = {k: v[np.sort(idx)] for k, v in ev.items()}
                for hn, comp in comps.items():
                    for m in (3, 2):
                        for L in range(1, comp.L + 1):
                            cl = core.Compiled(comp.h, coh.codes, L)
                            row, _ = floor_row(sub, cl, L, m, 5, evs, hierarchy=hn, frac=frac,
                                               seed=seed, **base)
                            out("floor_fractions", [row])
    elif a.part == "strata":
        def band20(b):
            return f"{int(b[:-1]) // 20 * 20}s"
        variants = {"sex_band10": lambda l: l, "sex_band20": lambda l: (l[0], band20(l[1])),
                    "sex": lambda l: (l[0],), "band10": lambda l: (l[1],)}
        for vn, f in variants.items():
            sub = restrat(coh, f)
            for hn, comp in comps.items():
                for m in (3, 2):
                    for L in range(1, comp.L + 1):
                        cl = core.Compiled(comp.h, coh.codes, L)
                        row, _ = floor_row(sub, cl, L, m, 5, ev, hierarchy=hn, strata=vn,
                                           n_strata=len(sub.strata), **base)
                        out("floor_strata", [row])
    elif a.part == "hospital":
        assert a.db == "eicu"
        hosp = coh.extra["hospital"]
        lab = [(int(h),) + tuple(coh.strata[s]) for h, s in zip(hosp, coh.strat)]
        uniq = sorted(set(lab))
        idx = {l: i for i, l in enumerate(uniq)}
        hc = Cohort(coh.name, coh.pids, np.array([idx[x] for x in lab], np.int32), uniq,
                    coh.indptr, coh.cidx, coh.codes, coh.extra)
        size = pd.Series(hosp).map(pd.Series(hosp).value_counts()).values
        bins = [0, 100, 250, 500, 1000, 2000, 5000, 10 ** 9]
        names = ["<100", "100-249", "250-499", "500-999", "1000-1999", "2000-4999", ">=5000"]
        hb = pd.cut(size, bins, right=False, labels=names)
        for hn, comp in comps.items():
            for m in (3, 2):
                for L in range(1, comp.L + 1):
                    cl = core.Compiled(comp.h, coh.codes, L)
                    row, rel = floor_row(hc, cl, L, m, 5, ev, hierarchy=hn, strata="hospital",
                                         hospitals=int(len(np.unique(hosp))), **base)
                    ref = ~rel.kept
                    for b in names:
                        sel = np.asarray(hb == b)
                        row[f"refused_share_hosp_{b}"] = float(ref[sel].mean()) if sel.sum() >= 10 else None
                        row[f"patients_hosp_{b}"] = int(sel.sum())
                        row[f"hospitals_{b}"] = int(len(np.unique(hosp[sel])))
                    out("floor_hospital", [row])
                    rown, _ = floor_row(coh, cl, L, m, 5, ev, hierarchy=hn, strata="network",
                                        **base)
                    out("floor_hospital", [rown])


if __name__ == "__main__":
    guarded(main)
