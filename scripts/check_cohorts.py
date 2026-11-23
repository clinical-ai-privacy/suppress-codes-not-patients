"""Cross-check of the cohorts and the hierarchies: cohort summaries (MIMIC-IV: 223,291 patients,
18 strata, 28,583 codes; eICU-CRD: 118,728 patients, 20 strata, 857 codes), coverage of the official
ICD hierarchy on the real codes, node counts per level and, with --floor, the floor at level 2 of
the truncation hierarchy. Aggregates only."""
import argparse
import time

from common import guarded, out
from kmrel import algos, core, data, hierarchy as H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--floor", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    coh = data.load(a.db)
    print(f"[{a.db}] loaded in {time.time() - t0:.1f}s", flush=True)
    out("cohort", [coh.summary()])
    rows = []
    for hn in ("trunc", "official") + (("trunc_chapter",) if a.db == "eicu" else ()):
        h = H.get(hn)
        nest = H.check_nesting(h, coh.codes)
        r = dict(db=a.db, hierarchy=hn, nodes=nest["nodes"], non_nested=nest["non_nested"])
        if hn == "official":
            for c in coh.codes:
                h.chain(c)
            cats_all = {":".join([c.split(":")[0], (c.split(":")[1][:4] if c.startswith("9:E")
                        else c.split(":")[1][:3])]) for c in coh.codes}
            r.update(categories=len(cats_all), unplaced_categories=len(h.unplaced),
                     unplaced_list=sorted(h.unplaced) if len(h.unplaced) <= 40 else "many")
        rows.append(r)
    out("hierarchy_check", rows)
    if a.floor:
        comp = core.Compiled(H.get("trunc"), coh.codes)
        for m in (3, 2):
            t1 = time.time()
            rel = algos.floor(coh, comp, 2, m, 5)
            st = rel.stats()
            out("trunc_floor_check", [dict(db=a.db, hierarchy="trunc", L=2, m=m, K=5,
                                        refused_share=st["refused_share"],
                                        refused=st["refused"], rounds=rel.meta["rounds"],
                                        runtime_s=time.time() - t1)])


if __name__ == "__main__":
    guarded(main)
