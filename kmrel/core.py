"""Releases and their exact k^m audits.

Release. Every kept patient p is published as (A_p, S_p): A_p his stratum, S_p the set of strings
(items) shown for his codes. ``shown`` is aligned with the cohort's CSR entries: the node id shown
for that (patient, code), or -1 if the code is suppressed. Refused patients are not published.

Two audits of a released record p (stratum s, candidates = released records of s):

* strict (exact-match, E): for every set Q' of p's items, 1 <= |Q'| <= m, the number of released
  records of s whose items contain Q' is >= K. Equivalently, for every set Q of p's shown true
  codes, |E(Q)| >= K, where E(Q) = records showing exactly the strings p shows for Q. This is
  standard k^m-anonymity of the published table, items treated as opaque strings (Terrovitis et
  al. 2008).
* loose (ancestor, C): for every set Q of p's shown true codes, the number of released records of
  s that show, for every q in Q, q or one of its ancestors (as strings) is >= K. This is the
  count under which the two earlier local methods, A and B2, were audited.

E(Q) is a subset of C(Q), so strict-valid implies loose-valid; with one uniform level the two
coincide (up to string coincidences across levels, which the audit of both reports).

The kernel ``min_support`` computes min |S_{i1} & ... & S_{ij}| over every 1 <= j <= m exactly, by
dense 0/1 matrix products on the columns of the union of the sets,
and returns the per-size minima and an argmin. ``min_support_sets`` is a plain
set-based re-implementation used by the tests and the cross-checks.

Data protection: only aggregate counts leave this module's callers.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from itertools import combinations

import numpy as np

from .hierarchy import Hierarchy

N_WORKERS = int(os.environ.get("KM_WORKERS", max(1, (os.cpu_count() or 2) - 1)))


# ------------------------------------------------------------------ compiled hierarchy ------
class Compiled:
    """A hierarchy compiled for a cohort's codes and truncated at top level L.

    anc[c, l]   node id of code c at level l (0..L)
    nodes       node id -> string; level_of[n] = the lowest level at which n occurs
    parent[n]   node id one level up (for a node at its lowest level l < L), -1 at L
    leaves[n]   number of cohort codes under n (for information loss)
    """

    def __init__(self, h: Hierarchy, codes, L=None):
        self.h = h
        self.L = h.top if L is None else L
        self.name = f"{h.name}@{self.L}"
        idx = {}
        nodes = []
        anc = np.empty((len(codes), self.L + 1), dtype=np.int32)
        for c, code in enumerate(codes):
            ch = h.chain(code)
            for l in range(self.L + 1):
                s = ch[l]
                n = idx.get(s)
                if n is None:
                    n = idx[s] = len(nodes)
                    nodes.append(s)
                anc[c, l] = n
        self.anc, self.nodes, self.idx = anc, nodes, idx
        nn = len(nodes)
        level_of = np.full(nn, self.L + 1, dtype=np.int32)
        for l in range(self.L + 1):
            np.minimum.at(level_of, anc[:, l], l)
        self.level_of = level_of
        # highest level at which a string occurs ("9:042" is a raw code and its own category);
        # the depth used to compare nodes and to find the entries under a node
        level_max = np.zeros(nn, dtype=np.int32)
        for l in range(self.L + 1):
            np.maximum.at(level_max, anc[:, l], l)
        self.level_max = level_max
        # a string must be the same node in every chain: the highest level it reaches must not
        # depend on the code (two different concepts written alike, e.g. a category and a chapter
        # both "C17", break parents, entry lookups and local recoding)
        top_min = np.full(nn, 99, dtype=np.int32)
        for l in range(self.L + 1):
            a = anc[:, l]
            ends = np.ones(len(a), bool) if l == self.L else a != anc[:, l + 1]
            np.minimum.at(top_min, a[ends], l)
        bad = np.flatnonzero(top_min != level_max)
        if len(bad):
            raise ValueError(f"{len(bad)} node strings reach different top levels in different "
                             f"chains, e.g. {[nodes[i] for i in bad[:5]]}")
        # parent = the next DIFFERENT ancestor string (nesting makes it unique)
        parent = np.full(nn, -1, dtype=np.int32)
        for l in range(self.L):
            a, b = anc[:, l], anc[:, l + 1]
            sel = a != b
            parent[a[sel]] = b[sel]
        self.parent = parent
        leaves = np.zeros(nn, dtype=np.int64)
        for c in range(len(codes)):
            for n in set(anc[c].tolist()):
                leaves[n] += 1
        self.leaves = leaves
        tot = len(codes)
        # NCP-style information loss of showing node n for one code (0 raw, 1 = all codes)
        self.loss = (leaves - 1) / max(1, tot - 1)
        # chains with duplicates removed (ancestors-or-self as distinct strings)
        self.chain = [np.unique(anc[c]) for c in range(len(codes))]

    def ancestors_or_self(self, n):
        out = [n]
        while self.parent[out[-1]] >= 0:
            out.append(int(self.parent[out[-1]]))
        return out


# ---------------------------------------------------------------------------- release -------
class Release:
    def __init__(self, coh, comp: Compiled, kept=None, shown=None, name=""):
        self.coh, self.comp, self.name = coh, comp, name
        self.kept = np.ones(coh.n, bool) if kept is None else kept.copy()
        if shown is None:
            shown = comp.anc[coh.cidx, 0].copy()
        self.shown = shown.astype(np.int32).copy()

    @classmethod
    def uniform(cls, coh, comp, level, kept=None, name=""):
        return cls(coh, comp, kept, comp.anc[coh.cidx, level], name or f"uniform{level}")

    def copy(self, name=None):
        return Release(self.coh, self.comp, self.kept, self.shown, name or self.name)

    def items(self, i):
        a, b = self.coh.indptr[i], self.coh.indptr[i + 1]
        s = self.shown[a:b]
        return np.unique(s[s >= 0])

    def level_of_entries(self):
        """Level of the shown node for every entry (-1 suppressed): the level l with
        anc[c, l] == shown, the lowest one."""
        anc = self.comp.anc[self.coh.cidx]
        lv = np.full(len(self.shown), -1, dtype=np.int32)
        for l in range(self.comp.L, -1, -1):
            hit = anc[:, l] == self.shown
            lv[hit] = l
        return lv

    def stats(self):
        coh = self.coh
        own = coh.owner()
        kept_e = self.kept[own]
        lv = self.level_of_entries()
        n_e = len(self.shown)
        out = dict(release=self.name, hierarchy=self.comp.name, patients=int(coh.n),
                   refused=int((~self.kept).sum()), refused_share=float((~self.kept).mean()),
                   entries=int(n_e),
                   entries_removed_share=float(((~kept_e) | (self.shown < 0)).mean()),
                   suppressed_share_of_kept=float((self.shown[kept_e] < 0).mean())
                   if kept_e.any() else 0.0)
        for l in range(self.comp.L + 1):
            out[f"share_level{l}"] = float((lv[kept_e] == l).mean()) if kept_e.any() else 0.0
        sh = self.shown[kept_e]
        out["mean_loss"] = float(np.where(sh >= 0, self.comp.loss[np.maximum(sh, 0)], 1.0).mean()) \
            if kept_e.any() else 1.0
        # patients left with no item / affected (refused or lost detail)
        empty = np.zeros(coh.n, bool)
        anyloss = np.zeros(coh.n, bool)
        for i in np.flatnonzero(self.kept):
            a, b = coh.indptr[i], coh.indptr[i + 1]
            s = self.shown[a:b]
            empty[i] = (s < 0).all() and b > a
            anyloss[i] = (lv[a:b] != 0).any()
        out["empty_share"] = float(empty.mean())
        out["affected_share"] = float(((~self.kept) | anyloss).mean())
        out["distinct_items"] = int(len(np.unique(sh[sh >= 0])))
        return out


# ---------------------------------------------------------------------------- kernel --------
def min_support(sets, m, floor=1):
    """Exact per-size minima of |S_{i1} & ... & S_{ij}|, j = 1..min(m, n), over distinct index
    sets, and an argmin (indices into `sets`) of the overall minimum (smallest size first).

    sets: list of sorted int arrays. floor: a value the caller guarantees as a lower bound (1 when
    the target lies in every set); a scan stops once its minimum reaches it, which keeps every
    per-size minimum exact (minima are non-increasing in j and bounded below by floor).
    Returns (mins: list of length m with None beyond n, arg tuple).
    """
    n = len(sets)
    mins = [None] * m
    if n == 0:
        return mins, ()
    sz = np.fromiter((len(s) for s in sets), dtype=np.int64, count=n)
    order = np.argsort(sz, kind="stable")
    sz_o = sz[order]
    mins[0] = int(sz_o[0])
    best, arg = mins[0], (int(order[0]),)
    if m < 2 or n < 2:
        return mins, arg
    if best <= floor:
        for j in range(1, min(m, n)):
            mins[j] = best
        return mins, arg
    so = [sets[i] for i in order]
    cols = np.unique(np.concatenate(so))
    B = np.zeros((n, len(cols)), dtype=np.float32)
    colidx = [np.searchsorted(cols, s) for s in so]
    for i, ci in enumerate(colidx):
        B[i, ci] = 1.0
    G = B @ B.T
    iu0, iu1 = np.triu_indices(n, 1)
    pv = G[iu0, iu1]
    j = int(pv.argmin())
    mins[1] = int(pv[j])
    if mins[1] < best:
        best, arg = mins[1], (int(order[iu0[j]]), int(order[iu1[j]]))
    if m < 3 or n < 3:
        return mins, arg
    if best <= floor:
        for jj in range(2, min(m, n)):
            mins[jj] = best
        return mins, arg
    b3, a3 = None, None
    for i in range(n - 2):
        Bi = B[i + 1:, colidx[i]]
        Ci = Bi @ Bi.T
        t0, t1 = np.triu_indices(Ci.shape[0], 1)
        v = Ci[t0, t1]
        k = int(v.argmin())
        if b3 is None or v[k] < b3:
            b3 = int(v[k])
            a3 = (int(order[i]), int(order[i + 1 + t0[k]]), int(order[i + 1 + t1[k]]))
            if b3 <= floor:
                break
    mins[2] = b3
    if b3 < best:
        best, arg = b3, a3
    if m < 4 or n < 4:
        return mins, arg
    if best <= floor:
        for jj in range(3, min(m, n)):
            mins[jj] = best
        return mins, arg
    # size 4: for every pair (i, j), i < j, restrict to the columns of both and take the pair
    # minimum over the sets after j
    b4, a4 = None, None
    for i in range(n - 3):
        Bi = B[i + 1:, colidx[i]]
        for jj in range(Bi.shape[0] - 2):
            cj = np.flatnonzero(Bi[jj])
            if len(cj) == 0:
                b4, a4 = 0, (int(order[i]), int(order[i + 1 + jj]), int(order[i + 2 + jj]),
                             int(order[i + 3 + jj]))
                break
            Bj = Bi[jj + 1:, cj]
            Cj = Bj @ Bj.T
            t0, t1 = np.triu_indices(Cj.shape[0], 1)
            v = Cj[t0, t1]
            k = int(v.argmin())
            if b4 is None or v[k] < b4:
                b4 = int(v[k])
                a4 = (int(order[i]), int(order[i + 1 + jj]),
                      int(order[i + 2 + jj + t0[k]]), int(order[i + 2 + jj + t1[k]]))
            if b4 <= floor:
                break
        if b4 is not None and b4 <= floor:
            break
    mins[3] = b4
    if b4 < best:
        best, arg = b4, a4
    if m > 4:
        raise NotImplementedError("m <= 4")
    return mins, arg


def min_support_sets(sets, m):
    """Plain set-based per-size minima (the independent re-implementation)."""
    ss = [set(map(int, s)) for s in sets]
    mins = [None] * m
    for j in range(1, min(m, len(ss)) + 1):
        mins[j - 1] = min(len(set.intersection(*[ss[i] for i in comb]))
                          for comb in combinations(range(len(ss)), j))
    return mins


# ------------------------------------------------------------------------- postings ---------
def stratum_index(rel: Release, s, members, mode):
    """Postings of stratum s over the released records `members` (patient indices, sorted).

    Returns (pos_of: dict patient -> position, post: dict key -> sorted int32 positions).
    strict: key = released item (node id). loose: key = node id as well; a true code's set is
    the union of the postings of its ancestors-or-self (built in record_sets).
    """
    coh = rel.coh
    pos_of = {int(p): i for i, p in enumerate(members)}
    keys, poss = [], []
    for i, p in enumerate(members):
        it = rel.items(p)
        keys.append(it)
        poss.append(np.full(len(it), i, dtype=np.int32))
    if keys:
        k = np.concatenate(keys)
        v = np.concatenate(poss)
    else:
        k = v = np.zeros(0, np.int32)
    o = np.lexsort((v, k))
    k, v = k[o], v[o]
    cut = np.flatnonzero(np.diff(k)) + 1
    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut, [len(k)]])
    post = {int(k[a]): v[a:b] for a, b in zip(starts, ends)} if len(k) else {}
    return pos_of, post


def build_postings(rel: Release):
    """Postings of every stratum at once (vectorized). Returns (pos, post): pos[p] = position of
    released patient p among the released records of its stratum (sorted by patient index; -1 if
    refused), post[s] = {item: sorted int32 positions of released records of s showing item}."""
    coh = rel.coh
    pos = np.full(coh.n, -1, dtype=np.int64)
    order = np.lexsort((np.arange(coh.n), coh.strat))
    order = order[rel.kept[order]]
    st = coh.strat[order]
    first = np.concatenate([[0], np.flatnonzero(np.diff(st)) + 1]) if len(st) else np.zeros(0, int)
    run = np.zeros(len(order), dtype=np.int64)
    if len(order):
        starts = np.zeros(len(order), dtype=np.int64)
        starts[first] = first
        starts = np.maximum.accumulate(starts)
        run = np.arange(len(order)) - starts
    pos[order] = run
    own = coh.owner()
    sel = rel.kept[own] & (rel.shown >= 0)
    e_s = coh.strat[own[sel]].astype(np.int64)
    e_i = rel.shown[sel].astype(np.int64)
    e_p = pos[own[sel]]
    NI = int(rel.shown.max()) + 2 if len(rel.shown) else 2
    key = (e_s * NI + e_i) * (coh.n + 1) + e_p
    key = np.unique(key)
    si = key // (coh.n + 1)
    pp = (key % (coh.n + 1)).astype(np.int32)
    cut = np.flatnonzero(np.diff(si)) + 1
    starts = np.concatenate([[0], cut]) if len(si) else np.zeros(0, int)
    ends = np.concatenate([cut, [len(si)]]) if len(si) else np.zeros(0, int)
    post = {s_: {} for s_ in range(len(coh.strata))}
    for a, b in zip(starts.tolist(), ends.tolist()):
        k = int(si[a])
        post[k // NI][k % NI] = pp[a:b]
    return pos, post


_E = np.zeros(0, dtype=np.int32)


def record_sets(rel: Release, p, post, mode, memo=None):
    """Candidate sets of record p (list of arrays) and their labels (items or code ids)."""
    coh = rel.coh
    if mode == "strict":
        it = rel.items(p)
        return [post[int(x)] for x in it], [int(x) for x in it]
    a, b = coh.indptr[p], coh.indptr[p + 1]
    out, lab = [], []
    seen = {}
    for e in range(a, b):
        if rel.shown[e] < 0:
            continue
        c = int(coh.cidx[e])
        key = tuple(rel.comp.chain[c].tolist())
        x = seen.get(key)
        if x is None:
            x = memo.get(key) if memo is not None else None
            if x is None:
                arrs = [post[n] for n in key if n in post]
                x = arrs[0] if len(arrs) == 1 else (np.unique(np.concatenate(arrs)) if arrs else _E)
                if memo is not None:
                    memo[key] = x
            seen[key] = x
            out.append(x)
            lab.append(c)
    return out, lab


# ------------------------------------------------------------------------------ audit -------
_G = {}


def _audit_chunk(args):
    s, recs = args
    rel, m, mode = _G["rel"], _G["m"], _G["mode"]
    post = _G["post"][s]
    memo = {}
    res = []
    for p in recs:
        sets, lab = record_sets(rel, p, post, mode, memo)
        mins, arg = min_support(sets, m)
        res.append((p, mins, tuple(lab[i] for i in arg)))
    return res


def audit(rel: Release, m, mode="strict", targets=None, workers=None, chunk=200):
    """Exact audit of released records. Returns dict patient -> (mins per size, argmin labels).

    targets: iterable of patient indices (released ones are audited; default every released).
    Records of one stratum with identical candidate keys are audited once (exact).
    """
    coh = rel.coh
    workers = N_WORKERS if workers is None else workers
    post = build_postings(rel)[1]
    if targets is None:
        tg = np.flatnonzero(rel.kept)
    else:
        tg = np.array(sorted(set(int(t) for t in targets)), dtype=np.int64)
        tg = tg[rel.kept[tg]] if len(tg) else tg
    # dedupe by (stratum, key)
    groups = {}
    for p in tg:
        if mode == "strict":
            key = (int(coh.strat[p]), tuple(rel.items(p).tolist()))
        else:
            a, b = coh.indptr[p], coh.indptr[p + 1]
            sh = rel.shown[a:b] >= 0
            key = (int(coh.strat[p]),
                   tuple(sorted({tuple(rel.comp.chain[c].tolist()) for c in coh.cidx[a:b][sh]})))
        groups.setdefault(key, []).append(int(p))
    reps = [(k[0], v[0]) for k, v in groups.items()]
    # chunk by stratum, heavy records spread
    lens = coh.lengths()
    by_s = {}
    for s, p in reps:
        by_s.setdefault(s, []).append(p)
    tasks = []
    for s, ps in by_s.items():
        ps.sort(key=lambda p: -lens[p])
        for i in range(0, len(ps), chunk):
            tasks.append((s, ps[i:i + chunk]))
    tasks.sort(key=lambda t: -sum(int(lens[p]) ** 3 for p in t[1]))
    _G.update(rel=rel, m=m, mode=mode, post=post)
    out_rep = {}
    if workers <= 1 or len(tasks) <= 1:
        for t in tasks:
            for p, mins, lab in _audit_chunk(t):
                out_rep[p] = (mins, lab)
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            for r in pool.imap_unordered(_audit_chunk, tasks, chunksize=1):
                for p, mins, lab in r:
                    out_rep[p] = (mins, lab)
    out = {}
    for k, v in groups.items():
        r = out_rep[v[0]]
        for p in v:
            out[p] = r
    _G.clear()
    return out


def overall_min(mins):
    v = [x for x in mins if x is not None]
    return min(v) if v else None


def violators(res, K):
    return [p for p, (mins, _) in res.items() if (overall_min(mins) or K) < K]


def summarize_audit(res, K, lens=None):
    """Aggregate shares: violators (< K), unique (= 1), smallest violating size."""
    n = len(res)
    viol = uniq = 0
    first = {1: 0, 2: 0, 3: 0, 4: 0}
    long_viol = long_n = 0
    for p, (mins, _) in res.items():
        mn = overall_min(mins)
        if mn is None:
            continue
        if mn < K:
            viol += 1
            j = next(j for j, x in enumerate(mins, 1) if x is not None and x < K)
            first[j] += 1
        if mn == 1:
            uniq += 1
        if lens is not None and lens[p] > 14:
            long_n += 1
            long_viol += mn < K
    out = dict(audited=n, violators=viol, violator_share=viol / n if n else 0.0,
               unique=uniq, unique_share=uniq / n if n else 0.0,
               first_size_1=first[1], first_size_2=first[2], first_size_3=first[3],
               first_size_4=first[4])
    if lens is not None:
        out["violator_share_gt14"] = long_viol / long_n if long_n else 0.0
    return out


def length_leak(rel: Release, K):
    """Share of released records whose number of released items is held by fewer than K released
    records of the stratum, and the share with a unique length."""
    coh = rel.coh
    kept = np.flatnonzero(rel.kept)
    nl = np.array([len(rel.items(p)) for p in kept])
    st = coh.strat[kept]
    key = st.astype(np.int64) * 100000 + nl
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    c = cnt[inv]
    return dict(length_lt_K_share=float((c < K).mean()) if len(c) else 0.0,
                length_unique_share=float((c == 1).mean()) if len(c) else 0.0)
