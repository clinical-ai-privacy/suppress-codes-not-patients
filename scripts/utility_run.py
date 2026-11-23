"""Utility of every cached release, both outcomes per database. Aggregates only.

  python scripts/utility_run.py --db mimic --outcome readmit30 --keys <cache keys, comma-separated>
         [--levels official,trunc] [--model lr|rf]

--model rf: a random forest on the same TF-IDF documents (second model family; results go to
utility_arms_rf / utility_pairs_rf). For refusal releases (floor, apriori, b3, b3g) the arm
':randmatch' trains on a random set of patients of the same size as the release keeps in that
split, at the release's uniform coding (level L for everyone): a matched control that separates
the volume of refusal from its selection.

Arms: raw:all; <hier>_L<l>:all for every level (the cost of the coding alone, nobody refused);
one arm per release key (training rows of released patients, the release's coding; refused
patients' test rows at the release's top level); for S-loc releases also ':nosupp' (a
test convention: test rows without suppression, at the variant's level).
Output per arm: AUROC mean over 25 splits, vs raw:all and vs every other release arm (paired NB
95% CI), training rows and vocabulary.
"""
import argparse
import re
from itertools import combinations

import numpy as np

from common import CACHE, guarded, out
from kmrel import core, data, hierarchy as H, utility as U


def parse_key(key):
    m = re.match(r"(mimic|eicu)_(official|trunc|trunc_chapter)(\d)_m(\d)_K(\d+)(?:_f([\d.]+))?(?:_hosp|_len)?_(.+)$", key)
    db, hier, L, mm, K, frac, meth = m.groups()
    return dict(db=db, hier=hier, L=int(L), m=int(mm), K=int(K), method=meth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--outcome", required=True)
    ap.add_argument("--keys", default="")
    ap.add_argument("--levels", default="official,trunc")
    ap.add_argument("--model", default="lr", choices=["lr", "rf"])
    ap.add_argument("--out-suffix", default=None, help="results file suffix (default: '' for lr, _rf)")
    a = ap.parse_args()
    U.MODEL = a.model
    suffix = a.out_suffix if a.out_suffix is not None else ("" if a.model == "lr" else f"_{a.model}")
    coh = data.load(a.db)
    pidx, cids, y = U.rows_for(a.db, coh, a.outcome)
    groups = pidx
    seed = U.SEED_MIMIC if a.db == "mimic" else U.SEED_EICU
    comps = {}

    def comp_of(h, L):
        if (h, L) not in comps:
            comps[(h, L)] = core.Compiled(H.get(h), coh.codes, L)
        return comps[(h, L)]

    arms = {}
    c0 = comp_of("official", 4)
    arms["raw:all"] = (U.docs(core.Release.uniform(coh, c0, 0), pidx, cids), None)
    for h in [x for x in a.levels.split(",") if x]:
        top = H.get(h).top
        for l in range(1, top + 1):
            arms[f"{h}_L{l}:all"] = (U.docs(core.Release.uniform(coh, comp_of(h, top), l),
                                            pidx, cids), None)
    rel_arms = []
    for key in [k for k in a.keys.split(",") if k]:
        f = CACHE / f"{key}.npz"
        if not f.exists():
            print(f"missing state {key}", flush=True)
            continue
        info = parse_key(key)
        comp = comp_of(info["hier"], info["L"])
        z = np.load(f)
        rel = core.Release(coh, comp, kept=z["kept"], shown=z["shown"], name=key)
        mask = rel.kept[pidx]
        arms[key] = (U.docs(rel, pidx, cids), mask)
        rel_arms.append(key)
        if info["method"].endswith("_import"):
            arms[key + ":held"] = (U.docs(rel, pidx, cids, mode="held"), mask)
        if info["method"] in ("floor", "apriori", "apriori_floor", "b3", "b3g"):
            uni = f"{info['hier']}_L{info['L']}:all"
            if uni in arms:
                arms[key + ":randmatch"] = (arms[uni][0], mask)
        if info["method"].startswith("sloc"):
            lv = 0 if info["method"] == "sloc0" else info["L"]
            rel.meta = dict(L_doc=lv)
            arms[key + ":nosupp"] = (U.docs(rel, pidx, cids, mode="nosupp"), None)
    auc, ratio, meta = U.evaluate(arms, y, groups, seed, workers=core.N_WORKERS)
    base = dict(db=a.db, outcome=a.outcome, rows=int(len(y)), base_rate=float(y.mean()),
                ratio_test_train=ratio, model=a.model)
    rows = []
    for arm in arms:
        d = U.nb_ci(auc[arm] - auc["raw:all"], ratio)
        rows.append(dict(base, arm=arm, auroc=float(auc[arm].mean()), d_raw=d[0], d_raw_lo=d[1],
                         d_raw_hi=d[2], **meta[arm]))
    out("utility_arms" + suffix, rows)
    pairs = []
    for x, z in combinations(rel_arms, 2):
        d = U.nb_ci(auc[x] - auc[z], ratio)
        pairs.append(dict(base, a=x, b=z, d=d[0], lo=d[1], hi=d[2]))
    for x in rel_arms:                      # release vs its matched random control
        if x + ":randmatch" in arms:
            d = U.nb_ci(auc[x] - auc[x + ":randmatch"], ratio)
            pairs.append(dict(base, a=x, b=x + ":randmatch", d=d[0], lo=d[1], hi=d[2]))
    out("utility_pairs" + suffix, pairs)


if __name__ == "__main__":
    guarded(main)
