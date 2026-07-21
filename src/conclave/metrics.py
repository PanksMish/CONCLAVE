"""Evaluation metrics used throughout the paper's experiments."""
import itertools
import numpy as np

from . import pareto as pareto_mod
from . import bargaining as barg
from .utils import normalized_kendall_tau_distance


def pareto_purity(shortlist_ids, all_ids, all_F):
    return pareto_mod.pareto_purity(shortlist_ids, all_ids, all_F)


def hypervolume(points, ref=None):
    """Exact hypervolume via inclusion-exclusion over dominated boxes
    [ref, point]. Fine for small shortlists (ks <= ~10) as used here.
    Objectives are maximization, assumed already in [0, 1]^M."""
    points = np.asarray(points, dtype=float)
    n, m = points.shape
    if n == 0:
        return 0.0
    if ref is None:
        ref = np.zeros(m)
    total = 0.0
    for r in range(1, n + 1):
        for combo in itertools.combinations(range(n), r):
            lower = np.minimum.reduce(points[list(combo)])
            vol = np.prod(np.clip(lower - ref, 0, None))
            sign = (-1) ** (r + 1)
            total += sign * vol
    return float(max(total, 0.0))


def hypervolume_for_ids(shortlist_ids, ids, F):
    idx = {u: i for i, u in enumerate(ids)}
    pts = np.array([F[idx[u]] for u in shortlist_ids if u in idx])
    if len(pts) == 0:
        return 0.0
    return hypervolume(pts)


def spread(shortlist_ids, ids, F):
    idx = {u: i for i, u in enumerate(ids)}
    pts = np.array([F[idx[u]] for u in shortlist_ids if u in idx])
    if len(pts) < 2:
        return 0.0
    return float(np.mean(np.std(pts, axis=0)))


def weighted_kendall_agreement(shortlist_ranking, agent_rankings, confidences):
    """Confidence-weighted agreement between the shortlist's internal ranking
    and each agent's ranking, restricted to the shortlisted items. Returns a
    value in [0, 1], higher = stronger agreement (1 - normalized discordance)."""
    items = set(shortlist_ranking)
    total_w, agg = 0.0, 0.0
    for ranking, c in zip(agent_rankings, confidences):
        if ranking is None:
            continue
        restricted = [u for u in ranking if u in items]
        if len(restricted) < 2:
            continue
        dist = normalized_kendall_tau_distance(shortlist_ranking, restricted)
        agg += c * (1.0 - dist)
        total_w += c
    if total_w == 0:
        return 0.0
    return agg / total_w


def nash_social_welfare(utilities):
    return barg.nash_social_welfare(utilities)


def jain_fairness(utilities):
    return barg.jain_fairness(utilities)


def shortlist_stability_series(history_shortlists):
    """Mean Jaccard overlap between consecutive round-wise shortlists (SS)."""
    if len(history_shortlists) < 2:
        return 1.0
    overlaps = []
    for a, b in zip(history_shortlists[:-1], history_shortlists[1:]):
        sa, sb = set(a), set(b)
        union = sa | sb
        overlaps.append(len(sa & sb) / len(union) if union else 1.0)
    return float(np.mean(overlaps))


def compute_all_metrics(result, ids, F):
    """result: the dict returned by pipeline.run_conclave (or a compatible
    baseline output with the same keys). Returns a flat dict of metrics."""
    shortlist = result["shortlist"]
    consensus_ranking = result.get("consensus_ranking", shortlist)
    C_ids = result.get("C_ids", ids)
    psi_matrix = result.get("psi_matrix")
    d = result.get("d")
    confidences = result.get("confidences", [1.0] * psi_matrix.shape[0]) if psi_matrix is not None else []

    shortlist_internal_ranking = [u for u in consensus_ranking if u in shortlist]
    agent_rankings = result.get("agent_rankings")

    metrics = {}
    metrics["PP5"] = pareto_purity(shortlist, ids, F)
    metrics["HV5"] = hypervolume_for_ids(shortlist, ids, F)
    metrics["Spread"] = spread(shortlist, ids, F)

    if agent_rankings is not None:
        metrics["WKA"] = weighted_kendall_agreement(shortlist_internal_ranking, agent_rankings, confidences)
    else:
        metrics["WKA"] = np.nan

    if psi_matrix is not None and d is not None:
        idx_of = {u: i for i, u in enumerate(C_ids)}
        s_idx = [idx_of[u] for u in shortlist if u in idx_of]
        utilities = barg.agent_utility_vector(psi_matrix, s_idx)
        metrics["NSW"] = nash_social_welfare(utilities)
        metrics["JF"] = jain_fairness(utilities)
    else:
        metrics["NSW"] = np.nan
        metrics["JF"] = np.nan

    history = result.get("history")
    if history is not None:
        metrics["SS"] = shortlist_stability_series(history["shortlists"])
    else:
        metrics["SS"] = np.nan

    metrics["RTC"] = result.get("rounds_to_convergence", np.nan)
    return metrics
