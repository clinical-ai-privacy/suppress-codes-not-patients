"""The most-specific-match attack.

The adversary knows the target's stratum A_p and a set Q of the target's TRUE codes
(|Q| = min(m, |U_p|), drawn uniformly). In the released table of the stratum he finds the
candidates C(Q): records that show, for every q in Q, q or an ancestor of q. He scores each
candidate by how specifically it matches, sum over q of the level of the deepest shown string
consistent with q (0 = q itself), and guesses uniformly among the candidates with the lowest
score. This is the reasoning of the minimality attack (Wong et al. 2007): a release that keeps
the violator's codes raw and coarsens its covers leaves the violator as the unique most specific
match.

Success for a target = 1/|best| if the target is among the best, else 0. Reported, over a seeded
sample of released targets: the mean success, the share of targets with success >= 1/2 and = 1,
and, for comparison, the loose guess 1/|C(Q)| (an adversary who does not weigh specificity).
Under a uniform level, best = C(Q) = E(Q), so success <= 1/K for every target. Under local
recoding (A, B2, B3) the most specific matches can be fewer than K even when the release passes
its audit; this is what the attack measures. Suppressed codes in Q can leave the target outside
C(Q) (success 0); "no candidate" counts the targets for which C(Q) is empty.
"""
from __future__ import annotations

import numpy as np

from .core import Release, build_postings


def attack(rel: Release, m, n_targets=20000, seed=20261106):
    coh, comp = rel.coh, rel.comp
    rng = np.random.default_rng(seed)
    released = np.flatnonzero(rel.kept & (coh.lengths() > 0))
    tg = rng.choice(released, size=min(n_targets, len(released)), replace=False)
    pos, posts = build_postings(rel)
    n_of = np.bincount(coh.strat[rel.kept], minlength=len(coh.strata))
    by_s = {}
    for p in tg:
        by_s.setdefault(int(coh.strat[p]), []).append(int(p))
    succ, succ_loose, empty = [], [], 0
    for s, ps in by_s.items():
        post = posts[s]
        n_s = int(n_of[s])
        for p in ps:
            row = coh.row(p)
            Q = rng.choice(row, size=min(m, len(row)), replace=False)
            score = np.zeros(n_s, dtype=np.int64)
            ok = np.ones(n_s, bool)
            for q in Q:
                spec = np.full(n_s, 10 ** 6, dtype=np.int64)
                for l in range(comp.L, -1, -1):
                    a = post.get(int(comp.anc[q, l]))
                    if a is not None:
                        spec[a] = l
                ok &= spec < 10 ** 6
                score += spec
            if not ok.any():
                empty += 1
                succ.append(0.0)
                succ_loose.append(0.0)
                continue
            me = int(pos[p])
            nC = int(ok.sum())
            succ_loose.append(1.0 / nC if ok[me] else 0.0)
            best = score[ok].min()
            nb = int((ok & (score == best)).sum())
            succ.append(1.0 / nb if ok[me] and score[me] == best else 0.0)
    succ, succ_loose = np.array(succ), np.array(succ_loose)
    return dict(attack_targets=int(len(succ)), attack_mean=float(succ.mean()),
                attack_ge_half=float((succ >= 0.5).mean()), attack_certain=float((succ == 1).mean()),
                attack_loose_mean=float(succ_loose.mean()), attack_no_candidate=empty / len(succ))
