"""Further measurements on releases already cached (aggregates only).

  python scripts/reaudit.py --db eicu [--pattern 'eicu_official*_K5_*'] [--skip-done]

For every cached state of the database: the length-augmented strict audit, the refused /
suppressed "nobody found" leak, the removal profile by subgroup and outcome, and the count-query
error at block and chapter level. One row per release in results/reaudit.jsonl, keyed by the
cache name. Per-hospital and length-strata caches are re-stratified exactly as releases.py does.
"""
import argparse
import json
import re
import time

import numpy as np

from common import CACHE, RESULTS, guarded, out
from kmrel import core, data, extra, hierarchy as H
from releases import outcome_events, restratify

KEY = re.compile(r"(mimic|eicu)_(official|trunc|trunc_chapter)(\d)_m(\d)_K(\d+)"
                 r"(?:_f([\d.]+))?(?:_(hosp|len))?_(.+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--pattern", default=None)
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args()
    done = set()
    f = RESULTS / "reaudit.jsonl"
    if a.skip_done and f.exists():
        done = {json.loads(l)["key"] for l in open(f)}
    coh0 = data.load(a.db)
    cohs, comps, evs = {}, {}, {}
    files = sorted(CACHE.glob(a.pattern or f"{a.db}_*.npz"))
    for fp in files:
        key = fp.stem
        mt = KEY.match(key)
        if not mt or key in done:
            continue
        db, hier, L, m, K, frac, strata, meth = mt.groups()
        if frac is not None:
            continue
        L, m, K = int(L), int(m), int(K)
        strata = strata or "band"
        if strata not in cohs:
            coh = coh0
            if strata == "hosp":
                hosp = coh.extra["hospital"]
                coh = restratify(coh, [(int(h),) + tuple(coh.strata[s])
                                       for h, s in zip(hosp, coh.strat)])
            elif strata == "len":
                lb = [extra.len_band(int(n)) for n in coh.lengths()]
                coh = restratify(coh, [tuple(coh.strata[s]) + (b,) for s, b in zip(coh.strat, lb)])
            cohs[strata] = coh
            evs[strata] = outcome_events(db, coh)
        coh, ev = cohs[strata], evs[strata]
        if (hier, L) not in comps:
            comps[(hier, L)] = core.Compiled(H.get(hier), coh.codes, L)
        comp = comps[(hier, L)]
        z = np.load(fp, allow_pickle=False)
        rel = core.Release(coh, comp, kept=z["kept"], shown=z["shown"], name=meth)
        t0 = time.time()
        print(f"=== {key}", flush=True)
        row = dict(key=key, db=db, hierarchy=hier, L=L, m=m, K=K, strata=strata, method=meth)
        row.update(extra.length_audit(rel, m, K))
        row.update(extra.refused_leak(rel, m))
        row.update(extra.removal_profile(rel, ev))
        for ql in range(2, min(L, 3) + 1):
            row.update({f"q{ql}_{k}": v for k, v in extra.query_error(rel, ql).items()
                        if k != "q_level"})
        row["reaudit_s"] = time.time() - t0
        out("reaudit", [row])


if __name__ == "__main__":
    guarded(main)
