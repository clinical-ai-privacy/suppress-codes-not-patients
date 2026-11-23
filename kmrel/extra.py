"""Further measurements on a release (aggregates only).

length_audit    strict audit with the released length as an always-known pseudo-item: the
                candidate sets of record p are intersected with the released records of its
                stratum that show the same number of items. Shares of violators and unique
                records under this augmented audit (the length leak combined with the codes).
removal_profile what a release removes, by subgroup: entries removed (refused or suppressed) by
                sex, age band, history-length band and chapter of the raw code; patients refused
                by the same subgroups; and the concentration of removed entries / refused
                patients on outcome events (rho = share of events removed / share removed).
refused_leak    an adversary who knows the stratum and m true codes of a REFUSED patient and
                finds no candidate at all learns that the patient was withheld (refusal is
                visible). Share of refused targets with an empty loose candidate set; the same
                for released targets whose known codes include a suppressed one.
query_error     count-query error (Terrovitis et al. 2008): for every node of a query level
                (block, chapter) the number of patients holding a code under it, and for the
                top-N pairs of such nodes the number holding both; estimated on the release from
                the shown strings (a string above the query level is unknown and does not
                support the query). Mean absolute relative error, as published and rescaled by
                the kept share (refusal correction).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .core import Release, build_postings, min_support, record_sets, summarize_audit

LEN_BANDS = [(1, 2), (3, 5), (6, 10), (11, 20), (21, 10 ** 9)]


def len_band(n):
    for i, (a, b) in enumerate(LEN_BANDS):
        if a <= n <= b:
            return i
    return len(LEN_BANDS) - 1


def len_band_label(i):
    a, b = LEN_BANDS[i]
    return f"{a}-{b}" if b < 10 ** 9 else f">={a}"


# ---------------------------------------------------------------- length-augmented audit ----
def length_audit(rel: Release, m, K):
    coh = rel.coh
    pos, post = build_postings(rel)
    kept = np.flatnonzero(rel.kept)
    nl = np.array([len(rel.items(p)) for p in kept], dtype=np.int64)
    # postings of released length per stratum
    lenpost = {}
    st = coh.strat[kept]
    order = np.lexsort((pos[kept], nl, st))
    for i in order:
        p = kept[i]
        lenpost.setdefault((int(st[i]), int(nl[i])), []).append(int(pos[p]))
    lenpost = {k: np.array(v, dtype=np.int32) for k, v in lenpost.items()}
    # dedupe by (stratum, items)
    groups = {}
    for p in kept:
        key = (int(coh.strat[p]), tuple(rel.items(p).tolist()))
        groups.setdefault(key, []).append(int(p))
    res = {}
    for key, ps in groups.items():
        s = key[0]
        p = ps[0]
        sets, lab = record_sets(rel, p, post[s], "strict")
        if not sets:
            r = ([None] * m, ())
        else:
            lp = lenpost[(s, len(lab))]
            sets = [np.intersect1d(x, lp, assume_unique=True) for x in sets]
            mins, arg = min_support(sets, m)
            r = (mins, tuple(lab[i] for i in arg))
        for q in ps:
            res[q] = r
    out = summarize_audit(res, K, coh.lengths())
    return {f"lenaud_{k}": v for k, v in out.items()
            if k in ("audited", "violators", "violator_share", "unique_share", "violator_share_gt14")}


# ------------------------------------------------------------------- removal profile --------
def removal_profile(rel: Release, ev=None):
    """ev: dict outcome -> events per patient (cohort index), optional."""
    coh, comp = rel.coh, rel.comp
    own = coh.owner()
    removed_e = (~rel.kept[own]) | (rel.shown < 0)
    supp_e = rel.kept[own] & (rel.shown < 0)
    refused = ~rel.kept
    lens = coh.lengths()
    out = {}
    # subgroups of patients
    sex = coh.extra.get("sex")
    band = coh.extra.get("band")
    if sex is None or band is None:          # plain (sex, band) strata
        sex = np.array([coh.strata[s][0] for s in coh.strat])
        band = np.array([coh.strata[s][1] for s in coh.strat])
    lb = np.array([len_band(n) for n in lens])

    def by(name, labels_p, labels_e):
        for lab in sorted(set(labels_p.tolist())):
            mp = labels_p == lab
            me = labels_e == lab
            lab_s = str(lab).replace(" ", "")
            out[f"removed_{name}_{lab_s}"] = float(removed_e[me].mean()) if me.any() else None
            out[f"refused_{name}_{lab_s}"] = float(refused[mp].mean()) if mp.any() else None

    by("sex", sex, sex[own])
    by("band", band, band[own])
    by("len", np.array([len_band_label(i) for i in lb]),
       np.array([len_band_label(i) for i in lb[own]]))
    # chapter of the raw code (official hierarchy: level 3; trunc: top level)
    lch = min(3, comp.L)
    ch_e = comp.anc[coh.cidx, lch]
    for n in np.unique(ch_e):
        me = ch_e == n
        out[f"removed_chapter_{comp.nodes[n]}"] = float(removed_e[me].mean())
        out[f"supp_chapter_{comp.nodes[n]}"] = float(supp_e[me].mean())
    # concentration on outcome events: entries removed from patients with / without events
    if ev is not None:
        sh_e = float(removed_e.mean())
        sh_p = float(refused.mean())
        for k, v in ev.items():
            has = v > 0
            tot = v.sum()
            # refusal: share of events removed (as floor_grid)
            e_ref = float(v[refused].sum() / tot) if tot else None
            out[f"{k}_events_refused_share"] = e_ref
            out[f"{k}_rho_refusal"] = (e_ref / sh_p) if (sh_p and e_ref is not None) else None
            # entries removed among event patients vs non-event patients
            he = has[own]
            out[f"{k}_removed_entries_event"] = float(removed_e[he].mean()) if he.any() else None
            out[f"{k}_removed_entries_noevent"] = float(removed_e[~he].mean()) if (~he).any() else None
            out[f"{k}_rho_entries"] = (out[f"{k}_removed_entries_event"] / sh_e) if sh_e else None
    return out


# ------------------------------------------------------------------------ refused leak ------
def refused_leak(rel: Release, m, n_targets=20000, seed=20261106):
    coh, comp = rel.coh, rel.comp
    rng = np.random.default_rng(seed)
    pos, posts = build_postings(rel)
    n_of = np.bincount(coh.strat[rel.kept], minlength=len(coh.strata))

    def no_candidate(p):
        s = int(coh.strat[p])
        post = posts[s]
        n_s = int(n_of[s])
        if n_s == 0:
            return True
        row = coh.row(p)
        Q = rng.choice(row, size=min(m, len(row)), replace=False)
        ok = np.ones(n_s, bool)
        for q in Q:
            hit = np.zeros(n_s, bool)
            for l in range(comp.L, -1, -1):
                a = post.get(int(comp.anc[q, l]))
                if a is not None:
                    hit[a] = True
            ok &= hit
            if not ok.any():
                return True
        return False

    out = {}
    ref = np.flatnonzero((~rel.kept) & (coh.lengths() > 0))
    if len(ref):
        tg = rng.choice(ref, size=min(n_targets, len(ref)), replace=False)
        out["refused_targets"] = int(len(tg))
        out["refused_no_candidate_share"] = float(np.mean([no_candidate(p) for p in tg]))
    else:
        out["refused_targets"] = 0
        out["refused_no_candidate_share"] = None
    # released patients with at least one suppressed code: does a Q that includes it find nobody?
    own = coh.owner()
    has_supp = np.zeros(coh.n, bool)
    np.logical_or.at(has_supp, own, rel.kept[own] & (rel.shown < 0))
    sp = np.flatnonzero(has_supp)
    if len(sp):
        tg = rng.choice(sp, size=min(n_targets, len(sp)), replace=False)
        out["suppressed_targets"] = int(len(tg))
        out["suppressed_no_candidate_share"] = float(np.mean([no_candidate(p) for p in tg]))
    else:
        out["suppressed_targets"] = 0
        out["suppressed_no_candidate_share"] = None
    return out


# -------------------------------------------------------------------------- query error -----
def _patient_node_matrix(coh, node_of_entry, n_nodes, keep_entry=None):
    own = coh.owner()
    sel = node_of_entry >= 0
    if keep_entry is not None:
        sel &= keep_entry
    M = sparse.csr_matrix((np.ones(int(sel.sum()), dtype=np.int32),
                           (own[sel], node_of_entry[sel])), shape=(coh.n, n_nodes))
    M.data[:] = 1
    M.sum_duplicates()
    M.data[:] = 1
    return M


def query_error(rel: Release, q_level, top_pairs=100):
    """Count queries at level q_level (official: 2 block, 3 chapter)."""
    coh, comp = rel.coh, rel.comp
    nn = len(comp.nodes)
    # map any node to its ancestor at q_level (nested hierarchy): via the codes
    up = np.full(nn, -1, dtype=np.int64)
    for l in range(0, q_level + 1):
        up[comp.anc[:, l]] = comp.anc[:, q_level]
    # raw: node at q_level of every entry
    raw_e = comp.anc[coh.cidx, q_level]
    # release: shown string -> ancestor at q_level if its level <= q_level, else unknown
    sh = rel.shown
    rel_e = np.where(sh >= 0, up[np.maximum(sh, 0)], -1)
    rel_e = np.where((sh >= 0) & (comp.level_max[np.maximum(sh, 0)] <= q_level), rel_e, -1)
    own = coh.owner()
    R = _patient_node_matrix(coh, raw_e, nn)
    E = _patient_node_matrix(coh, rel_e, nn, keep_entry=rel.kept[own])
    kept_share = float(rel.kept.mean())
    # singles
    r1 = np.asarray(R.sum(0)).ravel()
    e1 = np.asarray(E.sum(0)).ravel()
    nodes = np.flatnonzero(r1 > 0)
    rel_err = np.abs(e1[nodes] - r1[nodes]) / r1[nodes]
    rel_err_sc = np.abs(e1[nodes] / max(kept_share, 1e-9) - r1[nodes]) / r1[nodes]
    out = dict(q_level=q_level, q_nodes=int(len(nodes)),
               single_mare=float(rel_err.mean()), single_mare_rescaled=float(rel_err_sc.mean()),
               single_support_lost=float(1 - e1[nodes].sum() / r1[nodes].sum()))
    # pairs: top-N by raw co-occurrence
    Rc = R.tocsc()[:, nodes]
    C = (Rc.T @ Rc).toarray()
    np.fill_diagonal(C, 0)
    iu = np.triu_indices(len(nodes), 1)
    vals = C[iu]
    order = np.argsort(-vals)[:top_pairs]
    order = order[vals[order] > 0]
    Ec = E.tocsc()[:, nodes]
    CE = (Ec.T @ Ec).toarray()
    pr = vals[order].astype(float)
    pe = CE[iu][order].astype(float)
    out.update(pairs=int(len(order)),
               pair_mare=float(np.mean(np.abs(pe - pr) / pr)) if len(pr) else None,
               pair_mare_rescaled=float(np.mean(np.abs(pe / max(kept_share, 1e-9) - pr) / pr))
               if len(pr) else None,
               pair_support_lost=float(1 - pe.sum() / pr.sum()) if len(pr) else None,
               pair_min_raw=float(pr.min()) if len(pr) else None)
    return out
