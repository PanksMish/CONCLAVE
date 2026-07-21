"""Heuristic Mediator (HM): legacy four-agent mediator using confidence-weighted
percentile-rank fusion with simple compromise heuristics (predates CONCLAVE's
Nash-bargaining formulation)."""
import numpy as np
from ._harness import run_baseline


def _percentile_ranks(ranking, ids):
    n = len(ranking)
    pos = {u: p for p, u in enumerate(ranking)}
    return {u: 1.0 - pos.get(u, n - 1) / max(n - 1, 1) for u in ids}


def _hm_shortlist_fn(C_ids, C_F, rankings, confidences, psi_matrix, d, ks, rng):
    scores = {u: 0.0 for u in C_ids}
    total_c = sum(confidences) or 1.0
    for ranking, c in zip(rankings, confidences):
        if ranking is None:
            continue
        pr = _percentile_ranks(ranking, C_ids)
        for u in C_ids:
            scores[u] += (c / total_c) * pr[u]
    # simple compromise heuristic: mild penalty for candidates far from the
    # per-objective median (discourages one-sided picks) -- a hand-tuned
    # heuristic in place of principled bargaining.
    idx = {u: i for i, u in enumerate(C_ids)}
    med = np.median(C_F, axis=0)
    for u in C_ids:
        dist_penalty = 0.05 * np.mean(np.abs(C_F[idx[u]] - med))
        scores[u] -= dist_penalty
    ordered = sorted(C_ids, key=lambda u: -scores[u])
    return ordered[:ks]


def run(ids, F, **kwargs):
    return run_baseline(ids, F, _hm_shortlist_fn, use_pareto_filter=False, **kwargs)
