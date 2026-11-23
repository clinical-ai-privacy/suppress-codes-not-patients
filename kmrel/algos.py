"""Release algorithms. All of them are audited with the strict count (core.audit mode='strict').

floor(L)       uniform level L, then iterated whole-record refusal of strict violators to the
               fixed point: the refusal floor F_L. At a uniform level the strict and the loose
               counts coincide, so this is the greatest fixed point.
sloc(L)        S-loc: uniform level L, nobody refused; Jacobi rounds in which every
               violator drops the smallest-support item of its worst itemset; re-audit of the
               records that held a dropped item, to the fixed point.
b3(L)          consistent lifting by stratum collapse, on F_L (nobody else refused). Start from
               raw codes. A violator's worst itemset W gives the item x of W with the smallest
               support; the subtree of parent(x) is collapsed IN THAT STRATUM: every released
               record of the stratum holding a code under parent(x) now shows exactly parent(x).
               Collapse never lowers the support of any record's itemset (the release becomes a
               function of the old one), so non-violators stay valid and only violators are
               re-audited. Levels only rise and F_L is valid at uniform L, so b3 terminates with a
               strict-valid release of exactly F_L.
b3s(L)         hybrid: nobody refused, start from raw codes; for each violator either collapse as
               in b3 or suppress x from the violator, whichever loses less information
               (NCP-style loss: showing node n for a code costs (leaves(n)-1)/(codes-1),
               suppression costs 1). An item at level L can only be suppressed.
apriori(L)     Terrovitis et al. 2008 (Apriori-based anonymization, AA) with the strata as the
               adversary's background: a GLOBAL cut. Itemsets are checked by size 1, then 2, then
               3; a violating itemset generalizes the item whose global collapse costs least;
               optionally on F_L first (floor_first).

Rounds are Jacobi with a guard: within a round, a violator whose worst itemset touches a node
already changed in this round (in its stratum, or anywhere for the global cut) waits for the
next round, which approximates processing violations one at a time.
"""
from __future__ import annotations

import time

import numpy as np

from .core import Release, audit, overall_min, violators


class EntryIndex:
    """Entries of the cohort CSR grouped by (stratum, ancestor at level l), for every level."""

    def __init__(self, coh, comp):
        self.coh, self.comp = coh, comp
        own = coh.owner()
        self.strat_e = coh.strat[own]
        self.own = own
        self.by = []
        for l in range(comp.L + 1):
            a = comp.anc[coh.cidx, l].astype(np.int64)
            key = self.strat_e.astype(np.int64) * (len(comp.nodes) + 1) + a
            o = np.argsort(key, kind="stable")
            self.by.append((key[o], o))
        self.nn = len(comp.nodes) + 1

    def entries(self, s, node):
        """Entry indices of stratum s (s=None: every stratum) whose code lies under node."""
        l = int(self.comp.level_max[node])
        ks, o = self.by[l]
        if s is None:
            out = []
            for st in range(len(self.coh.strata)):
                k = st * self.nn + node
                a, b = np.searchsorted(ks, k), np.searchsorted(ks, k, side="right")
                out.append(o[a:b])
            return np.concatenate(out)
        k = s * self.nn + node
        a, b = np.searchsorted(ks, k), np.searchsorted(ks, k, side="right")
        return o[a:b]


def _log(verbose, *a):
    if verbose:
        print(*a, flush=True)


def _item_support(rel, _unused=None):
    """{(s, item): number of released records of stratum s showing item} (vectorized)."""
    coh = rel.coh
    own = coh.owner()
    sel = rel.kept[own] & (rel.shown >= 0)
    NI = int(rel.shown.max()) + 2
    k = (coh.strat[own[sel]].astype(np.int64) * NI + rel.shown[sel]) * (coh.n + 1) + own[sel]
    k = np.unique(k) // (coh.n + 1)
    u, c = np.unique(k, return_counts=True)
    return {(int(x // NI), int(x % NI)): int(n) for x, n in zip(u, c)}


def _holders(rel, pairs):
    """Released patients showing any (stratum, item) of `pairs` (vectorized)."""
    if not pairs:
        return set()
    coh = rel.coh
    own = coh.owner()
    NI = int(max(rel.shown.max(), max(x for _, x in pairs))) + 2
    key = coh.strat[own].astype(np.int64) * NI + rel.shown
    want = np.array(sorted(s * NI + x for s, x in pairs), dtype=np.int64)
    hit = np.isin(key, want) & rel.kept[own] & (rel.shown >= 0)
    return set(np.unique(own[hit]).tolist())


def floor(coh, comp, L, m, K, verbose=True, workers=None):
    t0 = time.time()
    rel = Release.uniform(coh, comp, L, name=f"floor{L}")
    targets = None
    rounds = 0
    members = coh.stratum_members()
    while True:
        res = audit(rel, m, "strict", targets=targets, workers=workers)
        v = violators(res, K)
        _log(verbose, f"  [floor L={L} m={m} K={K}] round {rounds}: audited {len(res)}, "
                      f"violators {len(v)}")
        if not v:
            break
        rel.kept[v] = False
        # re-audit kept records that share an item (in the stratum) with a refused one
        vs = np.array(v)
        own = coh.owner()
        sel = np.isin(own, vs)
        pairs = set(zip(coh.strat[own[sel]].tolist(), rel.shown[sel].tolist()))
        targets = [p for p in _holders(rel, pairs)]
        rounds += 1
        if not targets:
            break
    rel.meta = dict(rounds=rounds, runtime_s=time.time() - t0)
    return rel


def _suppress(rel, p, x):
    a, b = rel.coh.indptr[p], rel.coh.indptr[p + 1]
    seg = rel.shown[a:b]
    seg[seg == x] = -1


def sloc(coh, comp, L, m, K, verbose=True, workers=None, start=None):
    t0 = time.time()
    rel = start.copy() if start is not None else Release.uniform(coh, comp, L, name=f"sloc{L}")
    rel.name = f"sloc{L}"
    members = coh.stratum_members()
    targets = None
    rounds = 0
    while True:
        res = audit(rel, m, "strict", targets=targets, workers=workers)
        v = violators(res, K)
        _log(verbose, f"  [sloc L={L} m={m} K={K}] round {rounds}: audited {len(res)}, "
                      f"violators {len(v)}")
        if not v:
            break
        sup = _item_support(rel)
        dropped = {}
        for p in v:
            s = int(coh.strat[p])
            W = res[p][1]
            x = min(W, key=lambda it: (sup[(s, it)], it))
            _suppress(rel, p, x)
            dropped.setdefault(s, set()).add(x)
        targets = set(v) | _holders(rel, {(s_, x) for s_, xs in dropped.items() for x in xs})
        rounds += 1
    rel.meta = dict(rounds=rounds, runtime_s=time.time() - t0)
    return rel


def _collapse(rel, eidx, s, node):
    """Entries under node (in stratum s, or everywhere if s is None) of released records whose
    shown node is strictly below node: set to node. Returns the number of entries changed."""
    e = eidx.entries(s, node)
    if len(e) == 0:
        return 0
    lvn = rel.comp.level_max[node]
    sh = rel.shown[e]
    ok = sh >= 0
    lv = np.where(ok, rel.comp.level_max[np.maximum(sh, 0)], 99)
    sel = ok & (lv < lvn) & rel.kept[eidx.own[e]]
    rel.shown[e[sel]] = node
    return int(sel.sum())


def _collapse_cost(rel, eidx, s, node):
    e = eidx.entries(s, node)
    if len(e) == 0:
        return 0.0
    sh = rel.shown[e]
    lvn = rel.comp.level_max[node]
    ok = (sh >= 0) & rel.kept[eidx.own[e]]
    lv = np.where(ok, rel.comp.level_max[np.maximum(sh, 0)], 99)
    sel = ok & (lv < lvn)
    return float((rel.comp.loss[node] - rel.comp.loss[sh[sel]]).sum())


def _lift_target(rel, eidx, scope, x):
    """The node a lift of item x collapses to: x itself while the subtree of x (in the scope)
    still holds released entries shown strictly below x (this happens when x is also a raw
    code, e.g. ICD-9 "042", or after suppression), else parent(x); None if x is complete and
    at the top. If every item of a violating itemset W has no target, every record of the
    stratum holding codes under the items of W shows exactly those items, so supp(W) equals
    its uniform-top support, which is >= K on F_L: a b3 violator always has a target."""
    e = eidx.entries(scope, x)
    if len(e):
        sh = rel.shown[e]
        ok = (sh >= 0) & rel.kept[eidx.own[e]]
        if (ok & (rel.comp.level_max[np.maximum(sh, 0)] < rel.comp.level_max[x])).any():
            return int(x)
    a = int(rel.comp.parent[x])
    return a if a >= 0 else None


def _group_lift(rel, eidx, s, p, W, x, a, K, post, inv, comp):
    """B3g: lift p and the cheapest co-holders to exactly node a, so that at least K released
    records of stratum s show exactly (W minus x) plus a. Returns (entries changed, set of
    patients lifted, set of nodes removed from them) or None when fewer than K records of s hold
    codes under a and show W minus x (the caller then collapses the stratum)."""
    coh = rel.coh
    rest = [int(y) for y in W if y != x]
    cand = None
    for y in rest:
        py = post.get(y)
        if py is None:
            return None
        cand = py if cand is None else np.intersect1d(cand, py, assume_unique=True)
    e = eidx.entries(s, a)
    ok = rel.kept[eidx.own[e]] & (rel.shown[e] >= 0)
    e = e[ok]
    owners = eidx.own[e]
    holders = np.unique(owners)
    if cand is not None:
        holders = np.intersect1d(holders, inv[s][cand], assume_unique=False)
    if len(holders) < K:
        return None
    # cost of lifting each holder: NCP increase over its entries under a
    sh = rel.shown[e]
    below = comp.level_max[sh] < comp.level_max[a]
    dl = np.where(below, comp.loss[a] - comp.loss[sh], 0.0)
    cost = pd_groupsum(owners, dl)
    n_below = pd_groupsum(owners, below.astype(float))
    n_at = pd_groupsum(owners, (sh == a).astype(float))
    hs = [int(h) for h in holders]
    # a record shows item a iff one of its entries is shown exactly as a; holders whose entries
    # under a are all shown ABOVE a cannot be brought to a (that would lower them) and do not
    # count (counting them made the lift a no-op and the loop never ended)
    done = [h for h in hs if n_at.get(h, 0) > 0]
    todo = sorted((h for h in hs if n_at.get(h, 0) == 0 and n_below.get(h, 0) > 0 and h != p),
                  key=lambda h: (cost.get(h, 0.0), h))
    lift = [p] if n_at.get(p, 0) == 0 and n_below.get(p, 0) > 0 else []
    need = K - len(done) - len(lift)
    if need > len(todo) or (not lift and need <= 0):
        return None       # cannot reach K exact matches, or p already matches: collapse instead
    lift += todo[:max(0, need)]
    sel = np.isin(owners, np.array(lift)) & below
    removed = set(np.unique(sh[sel]).tolist())
    rel.shown[e[sel]] = a
    return int(sel.sum()), set(lift), removed


def _stall_report(rel, res, dbg, K, n=6):
    """Diagnostics when a b3 round changes nothing: node strings, levels and counts only."""
    comp, coh = rel.comp, rel.coh
    nd = lambda x: f"{comp.nodes[x]}(lo{comp.level_of[x]},hi{comp.level_max[x]})"
    print(f"  STALL: {len(dbg)} actions, all without effect", flush=True)
    for p, s, W, tg, choice, kind, outc in dbg[:n]:
        a, b = coh.indptr[p], coh.indptr[p + 1]
        sh = rel.shown[a:b]
        print(f"   s={s} supp={res[p][0]} W={[nd(x) for x in W]} "
              f"targets={ {nd(k): (nd(t) if t is not None else None) for k, t in tg.items()} } "
              f"choice={choice[0]}:{nd(choice[1])} {kind}->{outc}", flush=True)
        for x in W:
            under = [nd(int(y)) for y, c in zip(sh, coh.cidx[a:b])
                     if y >= 0 and x in comp.anc[c].tolist()]
            print(f"     own entries with x on the code chain: {under}; "
                  f"shown == x: {int((sh == x).sum())}", flush=True)


def pd_groupsum(keys, vals):
    if len(keys) == 0:
        return {}
    u, inv = np.unique(keys, return_inverse=True)
    sums = np.bincount(inv, weights=vals)
    return dict(zip(u.tolist(), sums.tolist()))


def b3(coh, comp, L, m, K, kept, verbose=True, workers=None, hybrid=False, global_cut=False,
       apriori_order=False, name=None, group=False, supp_w=1.0):
    """b3 (hybrid=False, kept = F_L), b3s (hybrid=True, kept = everybody), apriori
    (global_cut=True, apriori_order=True). supp_w scales the cost of suppressing an entry in the
    hybrid choice (1 = NCP-equivalent; large = prefer lifting; small = prefer suppression)."""
    t0 = time.time()
    rel = Release(coh, comp, kept=kept, name=name or ("b3s" if hybrid else "b3") + f"{L}")
    eidx = EntryIndex(coh, comp)
    members = coh.stratum_members()
    sizes = list(range(1, m + 1)) if apriori_order else [m]
    rounds = 0
    n_coll = n_supp = n_stuck = n_grp = 0
    for mm in sizes:
        targets = None
        while True:
            res = audit(rel, mm, "strict", targets=targets, workers=workers)
            v = violators(res, K)
            _log(verbose, f"  [{rel.name} m={mm}/{m} K={K}] round {rounds}: audited {len(res)}, "
                          f"violators {len(v)}")
            if not v:
                break
            sup = _item_support(rel)
            supp_pairs = set()
            refused_now = []
            if group:
                from .core import build_postings
                pos_, post_all = build_postings(rel)
                inv = {}
                for st_ in range(len(coh.strata)):
                    mem = np.flatnonzero((coh.strat == st_) & rel.kept)
                    inv[st_] = mem[np.argsort(pos_[mem])]
            # how many current violators have an item of their worst itemset under each node
            # (a collapse repairs all of them at once: its cost is shared, hybrid only)
            under = {}
            if hybrid:
                for p in v:
                    s = int(coh.strat[p])
                    nodes = set()
                    for x in res[p][1]:
                        nodes.update(comp.ancestors_or_self(int(x)))
                    for n in nodes:
                        under[(s, n)] = under.get((s, n), 0) + 1
            v.sort(key=lambda p: (overall_min(res[p][0]), p))
            fp0 = (hash(rel.shown.tobytes()), hash(rel.kept.tobytes()))
            dbg = []
            touched = set()
            new_targets = set()
            for p in v:
                s = int(coh.strat[p])
                W = list(res[p][1])
                scope = None if global_cut else s
                if any((scope, x) in touched for x in W):
                    new_targets.add(p)
                    continue
                W.sort(key=lambda it: (sup[(s, it)], comp.level_max[it], it))
                targets_of = {x: _lift_target(rel, eidx, scope, x) for x in W}
                lift = [x for x in W if targets_of[x] is not None]
                choice = None
                if global_cut:
                    if lift:
                        costs = [(_collapse_cost(rel, eidx, None, targets_of[x]), i, x)
                                 for i, x in enumerate(lift)]
                        choice = ("lift", min(costs)[2])
                elif hybrid:
                    x = W[0]
                    if targets_of[x] is None:
                        choice = ("supp", x)
                    else:
                        cl = (_collapse_cost(rel, eidx, s, targets_of[x])
                              / max(1, under.get((s, targets_of[x]), 1)))
                        aa, bb = coh.indptr[p], coh.indptr[p + 1]
                        ne = int((rel.shown[aa:bb] == x).sum())
                        cs = ne * (1.0 - comp.loss[x]) * supp_w   # supp_w: price of suppression
                        choice = ("lift", x) if cl <= cs else ("supp", x)
                elif lift:
                    choice = ("lift", lift[0])
                if choice is None:
                    n_stuck += 1
                    rel.kept[p] = False          # impossible on F_L (see _lift_target); counted
                    refused_now.append(p)
                    continue
                kind, x = choice
                if kind == "lift" and group and not global_cut:
                    a = targets_of[x]
                    g = _group_lift(rel, eidx, s, p, W, x, a, K, post_all[s], inv, comp)
                    dbg.append((p, s, W, dict(targets_of), choice, "group",
                                None if g is None else (g[0], len(g[1]), len(g[2]))))
                    if g is not None:
                        n_grp += 1
                        _, lifted, removed = g
                        touched.update({(s, x), (s, a)} | {(s, int(y)) for y in removed})
                        new_targets.add(p)
                        new_targets.update(lifted)
                        supp_pairs.update((s, int(y)) for y in removed)   # their holders lost support
                        continue
                if kind == "lift":
                    a = targets_of[x]
                    changed = _collapse(rel, eidx, scope, a)
                    dbg.append((p, s, W, dict(targets_of), choice, "collapse", changed))
                    n_coll += 1
                    touched.update({(scope, x), (scope, a)})
                    # every node whose entries moved to a is touched
                    e = eidx.entries(scope, a)
                    for l in range(0, comp.level_max[a]):
                        touched.update((scope, int(n)) for n in np.unique(comp.anc[coh.cidx[e], l]))
                    new_targets.add(p)
                    del changed
                else:
                    _suppress(rel, p, x)
                    n_supp += 1
                    touched.add((s, x))
                    new_targets.add(p)
                    supp_pairs.add((s, x))
            # holders of a suppressed item lost support: re-audit them
            # a refusal lowers supports: re-audit the records sharing an item with the refused
            if refused_now:
                own = coh.owner()
                sel = np.isin(own, np.array(refused_now))
                supp_pairs |= set(zip(coh.strat[own[sel]].tolist(), rel.shown[sel].tolist()))
            targets = new_targets | _holders(rel, {(a, b) for a, b in supp_pairs if b >= 0})
            if (hash(rel.shown.tobytes()), hash(rel.kept.tobytes())) == fp0:
                _stall_report(rel, res, dbg, K)
                raise RuntimeError(f"b3 stalled at round {rounds}: no change, {len(v)} violators")
            rounds += 1
    rel.meta = dict(rounds=rounds, runtime_s=time.time() - t0, collapses=n_coll,
                    suppressions=n_supp, stuck_refused=n_stuck, group_lifts=n_grp)
    return rel


def apriori(coh, comp, m, K, kept=None, verbose=True, workers=None, name=None):
    kept = np.ones(coh.n, bool) if kept is None else kept
    return b3(coh, comp, comp.L, m, K, kept, verbose=verbose, workers=workers, global_cut=True,
              apriori_order=True, name=name or f"apriori{comp.L}")
