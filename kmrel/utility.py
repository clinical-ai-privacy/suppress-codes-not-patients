"""What a release costs a prediction model.

Rows: MIMIC admissions of the 8,000-patient sample (seed 20261007) with diagnosis codes; outcome
30-day readmission (in-hospital deaths dropped) or in-hospital death (all
admissions). eICU: health-system stays with known discharge status; outcome
in-hospital death. One TF-IDF document per row over the strings its codes show in the release
(token_pattern \\S+, min_df = 3 on the training rows; a string shared by several codes repeats),
LogisticRegression(max_iter = 2000), 25 patient-grouped 75/25 splits (GroupShuffleSplit,
random_state = SEED + rep). Refusal removes training rows only; the test rows are always every
test patient. Coding of a row: the release's own strings for released patients (suppressed codes
absent); refused patients are coded at the release's top level L (what the floor would show).
Paired differences over the splits with the Nadeau-Bengio corrected resampled-t interval,
var = (1/J + n_test/n_train) S^2.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np
from scipy import stats

SEED_MIMIC = 20261007
SEED_EICU = 20261018
N8 = 8000
REPS = 25

_G = {}
MODEL = "lr"       # "lr": logistic regression; "rf": random forest (second model family)


def mimic_sample(coh):
    return np.random.default_rng(SEED_MIMIC).choice(coh.pids, min(N8, coh.n), replace=False)


def rows_for(db, coh, outcome):
    """(pidx per row, code ids per row, y) for the outcome."""
    from . import data
    if db == "mimic":
        adm = data.mimic_admissions(coh, sample=mimic_sample(coh))
        if outcome == "readmit30":
            adm = adm[adm.hospital_expire_flag == 0]
            y = adm.readmit30.values
        else:
            y = adm.death.values
        return adm.pidx.values, list(adm.cids.values), y.astype(int)
    st = data.eicu_stays(coh, first_day_only=(outcome == "death_day1"))
    return st.pidx.values, list(st.cids.values), st.death.values.astype(int)


def docs(rel, pidx, cids, mode="release"):
    """One document per row. mode='release': released patients show their strings (suppressed
    codes absent), refused ones the top level L. mode='nosupp': as 'release', but a suppressed
    code shows its string at the uniform level rel.meta['L_doc'] (a test convention in
    which a deployed model sees complete records). mode='held': refused patients show the
    strings they had when refused (the convention of the imported B2 runs; not a release)."""
    coh, comp = rel.coh, rel.comp
    top = comp.anc[:, comp.L]
    nodes = comp.nodes
    out = []
    cache = {}
    Ldoc = getattr(rel, "meta", {}).get("L_doc")
    for p, cs in zip(pidx, cids):
        mp_ = cache.get(p)
        if mp_ is None:
            a, b = coh.indptr[p], coh.indptr[p + 1]
            if rel.kept[p] or mode == "held":
                sh = rel.shown[a:b]
                if mode == "nosupp" and Ldoc is not None:
                    sh = np.where(sh >= 0, sh, comp.anc[coh.cidx[a:b], Ldoc])
                mp_ = dict(zip(coh.cidx[a:b].tolist(), sh.tolist()))
            else:
                mp_ = dict(zip(coh.cidx[a:b].tolist(), top[coh.cidx[a:b]].tolist()))
            cache[p] = mp_
        out.append(" ".join(nodes[mp_[c]] for c in cs.tolist() if mp_.get(c, -1) >= 0))
    return out


def _fit(task):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit
    arm, rep = task
    X = _G["docs"][arm]
    y, groups, seed = _G["y"], _G["groups"], _G["seed"]
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                    random_state=seed + rep).split(X, y, groups))
    n_full = len(tr)
    mask = _G["mask"].get(arm)
    if mask is not None and arm.endswith(":randmatch"):
        # random patients, as many as the release keeps among the training patients of this split
        rng = np.random.default_rng(seed + 1000 + rep)
        tp = np.unique(groups[tr])
        n_keep = len(np.unique(groups[tr[mask[tr]]]))
        chosen = rng.choice(tp, size=n_keep, replace=False)
        tr = tr[np.isin(groups[tr], chosen)]
    elif mask is not None:
        tr = tr[mask[tr]]
    trd = [X[i] for i in tr]
    if len(np.unique(y[tr])) < 2:   # a release that keeps too few patients: one class left
        return arm, rep, 0.5, len(tr), len(te), n_full, -1
    if not any(d.strip() for d in trd):
        return arm, rep, 0.5, len(tr), len(te), n_full, 0
    vec = TfidfVectorizer(token_pattern=r"\S+", min_df=3)
    try:
        Xtr = vec.fit_transform(trd)
    except ValueError:              # empty vocabulary
        return arm, rep, 0.5, len(tr), len(te), n_full, 0
    if MODEL == "rf":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_features="sqrt",
                                     n_jobs=1, random_state=seed + rep)
    else:
        clf = LogisticRegression(max_iter=2000)
    p = clf.fit(Xtr, y[tr]).predict_proba(vec.transform([X[i] for i in te]))[:, 1]
    return arm, rep, float(roc_auc_score(y[te], p)), len(tr), len(te), n_full, len(vec.vocabulary_)


def nb_ci(d, ratio):
    d = np.asarray(d, float)
    J = len(d)
    mu, s2 = d.mean(), d.var(ddof=1)
    se = np.sqrt((1 / J + ratio) * s2)
    q = stats.t.ppf(0.975, J - 1)
    return float(mu), float(mu - q * se), float(mu + q * se)


def evaluate(arms, y, groups, seed, workers=8, reps=REPS):
    """arms: name -> (docs list, train mask over rows or None). Returns per-arm AUROC arrays and
    the test/train row ratio."""
    _G.update(docs={k: v[0] for k, v in arms.items()}, mask={k: v[1] for k, v in arms.items()},
              y=y, groups=groups, seed=seed)
    tasks = [(a, r) for a in arms for r in range(reps)]
    res = {}
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            for out in pool.imap_unordered(_fit, tasks):
                res[(out[0], out[1])] = out
    else:
        for t in tasks:
            out = _fit(t)
            res[(out[0], out[1])] = out
    _G.clear()
    auc = {a: np.array([res[(a, r)][2] for r in range(reps)]) for a in arms}
    ratio = float(np.mean([res[(a, r)][4] / res[(a, r)][5] for a in arms for r in range(reps)]))
    meta = {a: dict(train_rows=float(np.mean([res[(a, r)][3] for r in range(reps)])),
                    vocab=float(np.mean([max(0, res[(a, r)][6]) for r in range(reps)])),
                    one_class_splits=int(sum(res[(a, r)][6] < 0 for r in range(reps))))
            for a in arms}
    return auc, ratio, meta
