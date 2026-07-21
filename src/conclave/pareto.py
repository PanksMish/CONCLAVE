"""Pareto dominance and epsilon-archive candidate filtering."""
import numpy as np


def dominance_mask(F):
    """Return a boolean mask over rows of F indicating non-dominated (Pareto) candidates.

    u dominates v iff u >= v in all objectives and u > v in at least one (Eq. 4).
    All objectives are maximization objectives (higher is better).
    """
    n = F.shape[0]
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        ge = np.all(F >= F[i], axis=1)
        gt = np.any(F > F[i], axis=1)
        dominators = ge & gt
        dominators[i] = False
        if dominators.any():
            dominated[i] = True
    return ~dominated


def epsilon_archive(ids, F, eps=0.02):
    """Restrict the non-dominated set to one representative per occupied
    eps-grid cell to preserve frontier coverage (Eq. 16).

    Returns filtered (ids, F, mask_into_original_pareto_set).
    """
    mask = dominance_mask(F)
    p_ids = [ids[i] for i in range(len(ids)) if mask[i]]
    p_F = F[mask]

    if len(p_ids) == 0:
        return p_ids, p_F, mask

    cells = np.floor(p_F / eps).astype(int)
    cell_keys = [tuple(c) for c in cells]

    reps = {}
    for i, key in enumerate(cell_keys):
        cell_center = (cells[i] + 0.5) * eps
        dist = np.linalg.norm(p_F[i] - cell_center)
        if key not in reps or dist < reps[key][1]:
            reps[key] = (i, dist)

    keep_idx = sorted(v[0] for v in reps.values())
    out_ids = [p_ids[i] for i in keep_idx]
    out_F = p_F[keep_idx]
    return out_ids, out_F, mask


def pareto_purity(shortlist_ids, all_ids, all_F):
    """Fraction of shortlisted candidates that are Pareto-optimal in the full batch (PP_ks)."""
    mask = dominance_mask(all_F)
    pareto_ids = {all_ids[i] for i in range(len(all_ids)) if mask[i]}
    if len(shortlist_ids) == 0:
        return 0.0
    return sum(1 for u in shortlist_ids if u in pareto_ids) / len(shortlist_ids)
