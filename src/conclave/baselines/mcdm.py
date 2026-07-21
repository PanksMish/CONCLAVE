"""Classical multi-criteria decision-making baselines: TOPSIS and VIKOR.
Both operate directly on the normalized objective matrix with fixed weights,
without any agent negotiation."""
import numpy as np
from ._harness import run_baseline

MCDM_WEIGHTS = np.full(4, 0.25)
VIKOR_V = 0.5


def _topsis_shortlist_fn(C_ids, C_F, rankings, confidences, psi_matrix, d, ks, rng):
    # vector-normalize each column, then apply weights
    norm = C_F / np.sqrt((C_F ** 2).sum(axis=0, keepdims=True) + 1e-12)
    weighted = norm * MCDM_WEIGHTS
    ideal_best = weighted.max(axis=0)
    ideal_worst = weighted.min(axis=0)
    dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
    closeness = dist_worst / (dist_best + dist_worst + 1e-12)
    order = np.argsort(-closeness)
    return [C_ids[i] for i in order[:ks]]


def _vikor_shortlist_fn(C_ids, C_F, rankings, confidences, psi_matrix, d, ks, rng):
    f_best = C_F.max(axis=0)
    f_worst = C_F.min(axis=0)
    denom = np.clip(f_best - f_worst, 1e-9, None)
    weighted_gap = MCDM_WEIGHTS * (f_best - C_F) / denom
    S = weighted_gap.sum(axis=1)
    R = weighted_gap.max(axis=1)
    S_best, S_worst = S.min(), S.max()
    R_best, R_worst = R.min(), R.max()
    Q = (VIKOR_V * (S - S_best) / max(S_worst - S_best, 1e-9) +
         (1 - VIKOR_V) * (R - R_best) / max(R_worst - R_best, 1e-9))
    order = np.argsort(Q)  # lower Q is better
    return [C_ids[i] for i in order[:ks]]


def run_topsis(ids, F, **kwargs):
    return run_baseline(ids, F, _topsis_shortlist_fn, use_pareto_filter=False, **kwargs)


def run_vikor(ids, F, **kwargs):
    return run_baseline(ids, F, _vikor_shortlist_fn, use_pareto_filter=False, **kwargs)
