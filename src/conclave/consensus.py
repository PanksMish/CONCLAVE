"""Weighted ordinal consensus ranking + bargaining-aware greedy shortlist construction."""
import numpy as np
from . import bargaining as barg


def weighted_preference_matrix(ids, rankings, confidences):
    """W_ij = sum_k c_k * 1[agent k ranks i above j]  (Eq. 23)."""
    n = len(ids)
    idx = {u: i for i, u in enumerate(ids)}
    W = np.zeros((n, n))
    for ranking, c in zip(rankings, confidences):
        if ranking is None:
            continue
        pos = {u: p for p, u in enumerate(ranking)}
        for u in ids:
            for v in ids:
                if u == v:
                    continue
                if pos.get(u, 1e9) < pos.get(v, 1e9):
                    W[idx[u], idx[v]] += c
    return W


def weighted_kendall_consensus_ranking(ids, rankings, confidences):
    """Approximate the weighted-Kemeny objective (Eq. 22) via weighted Copeland
    scoring on the pairwise preference matrix W (Eq. 23): a standard, tractable
    Kemeny approximation. Returns consensus ranking (best -> worst)."""
    W = weighted_preference_matrix(ids, rankings, confidences)
    net_score = W.sum(axis=1) - W.sum(axis=0)
    order = np.argsort(-net_score)
    return [ids[i] for i in order], W


def rank_scores(consensus_ranking):
    """rankscore_pi*(u), Eq. 25."""
    n = len(consensus_ranking)
    pos = {u: p for p, u in enumerate(consensus_ranking)}
    scores = {}
    for u in consensus_ranking:
        scores[u] = 1.0 - (pos[u]) / max(n - 1, 1)
    return scores


def _marginal_hv(F, idx_of, shortlist_idx, u):
    """Marginal hypervolume contribution of candidate u given the current
    partial shortlist -- operationalizes the coverage term L_cov in Eq. 10
    (discourages concentrating the shortlist in a narrow region of objective
    space) as a positive reward inside the greedy selection score."""
    from .metrics import hypervolume
    base_pts = F[shortlist_idx] if shortlist_idx else np.empty((0, F.shape[1]))
    base_hv = hypervolume(base_pts) if len(base_pts) else 0.0
    trial_pts = np.vstack([base_pts, F[idx_of[u]][None, :]]) if len(base_pts) else F[idx_of[u]][None, :]
    return hypervolume(trial_pts) - base_hv


def build_shortlist(ids, consensus_ranking, psi_matrix, d, confidences, ks,
                     F=None, alpha=0.5, coverage_weight=0.2, use_bargaining=True,
                     gain_fn=None, gain_fn_kwargs=None):
    """Greedy bargaining-aware shortlist construction (Eqs. 24-28), with an
    added coverage/hypervolume term reflecting the paper's L_cov component of
    the Pareto-fidelity loss (Eq. 10).

    If use_bargaining is False (ablation: CONCLAVE w/o Bargaining), the
    shortlist is simply the top-ks candidates by consensus rank score.

    gain_fn: callable(psi_matrix, idx_list, d, confidences, **kwargs) -> scalar.
    Defaults to the Nash bargaining gain (Eq. 8); pass an alternative (e.g.
    Kalai-Smorodinsky or Egalitarian) to compare bargaining mechanisms.
    """
    idx_of = {u: i for i, u in enumerate(ids)}
    rscore = rank_scores(consensus_ranking)
    gain_fn = gain_fn or barg.nash_gain
    gain_fn_kwargs = gain_fn_kwargs or {}

    if not use_bargaining:
        ordered = sorted(ids, key=lambda u: -rscore[u])
        return ordered[:ks]

    remaining = list(ids)
    shortlist = []
    shortlist_idx = []
    marg_history = []
    use_coverage = F is not None and coverage_weight > 0

    for _ in range(ks):
        current_gain = gain_fn(psi_matrix, shortlist_idx, d, confidences, **gain_fn_kwargs)
        candidate_deltas = {}
        hv_deltas = {}
        for u in remaining:
            trial_idx = shortlist_idx + [idx_of[u]]
            new_gain = gain_fn(psi_matrix, trial_idx, d, confidences, **gain_fn_kwargs)
            candidate_deltas[u] = new_gain - current_gain
            if use_coverage:
                hv_deltas[u] = _marginal_hv(F, idx_of, shortlist_idx, u)

        deltas = np.array(list(candidate_deltas.values()))
        if deltas.max() - deltas.min() > 1e-9:
            norm_deltas = {u: (v - deltas.min()) / (deltas.max() - deltas.min())
                           for u, v in candidate_deltas.items()}
        else:
            norm_deltas = {u: 0.5 for u in candidate_deltas}

        if use_coverage:
            hv_arr = np.array(list(hv_deltas.values()))
            if hv_arr.max() - hv_arr.min() > 1e-9:
                norm_hv = {u: (v - hv_arr.min()) / (hv_arr.max() - hv_arr.min())
                           for u, v in hv_deltas.items()}
            else:
                norm_hv = {u: 0.5 for u in hv_deltas}
        else:
            norm_hv = {u: 0.0 for u in remaining}

        best_u, best_score = None, -np.inf
        for u in remaining:
            bargaining_term = (1 - coverage_weight) * norm_deltas[u] + coverage_weight * norm_hv[u]
            s = alpha * rscore[u] + (1 - alpha) * bargaining_term
            if s > best_score:
                best_score, best_u = s, u

        shortlist.append(best_u)
        shortlist_idx.append(idx_of[best_u])
        remaining.remove(best_u)
        marg_history.append(candidate_deltas[best_u])

    return shortlist
