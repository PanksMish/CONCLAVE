"""Weighted Borda and Weighted Kemeny rank-aggregation baselines."""
import numpy as np
from ._harness import run_baseline
from .. import consensus as cons


def _borda_shortlist_fn(C_ids, C_F, rankings, confidences, psi_matrix, d, ks, rng):
    n = len(C_ids)
    scores = {u: 0.0 for u in C_ids}
    for ranking, c in zip(rankings, confidences):
        if ranking is None:
            continue
        for pos, u in enumerate(ranking):
            scores[u] += c * (n - 1 - pos)  # classical Borda points, confidence-weighted
    ordered = sorted(C_ids, key=lambda u: -scores[u])
    return ordered[:ks]


def _kemeny_shortlist_fn(C_ids, C_F, rankings, confidences, psi_matrix, d, ks, rng):
    # rank-only variant of CONCLAVE's ordinal-consensus module (Eq. 22-23),
    # with NO Pareto filtering, NO bargaining-aware selection, NO stability logic.
    consensus_ranking, _ = cons.weighted_kendall_consensus_ranking(C_ids, rankings, confidences)
    return consensus_ranking[:ks]


def run_borda(ids, F, **kwargs):
    return run_baseline(ids, F, _borda_shortlist_fn, use_pareto_filter=False, **kwargs)


def run_kemeny(ids, F, **kwargs):
    return run_baseline(ids, F, _kemeny_shortlist_fn, use_pareto_filter=False, **kwargs)
