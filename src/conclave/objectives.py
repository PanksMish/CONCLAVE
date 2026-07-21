"""Objective-vector construction and candidate-batch extraction (Eq. 1, 33)."""
import numpy as np

OBJECTIVE_COLUMNS = ["f_return", "f_risk", "f_cost", "f_consistency"]
OBJECTIVE_NAMES = ["Return", "Risk", "Cost", "Consistency"]


def get_candidate_batch(df, category):
    """Return (ids, F) for a given SEBI category.

    ids: list[str] fund ids
    F:   (N, 4) normalized objective matrix, columns = OBJECTIVE_COLUMNS
    """
    sub = df[df["category"] == category].reset_index(drop=True)
    ids = sub["fund_id"].tolist()
    F = sub[OBJECTIVE_COLUMNS].to_numpy(dtype=float)
    return ids, F


def sample_batch(ids, F, batch_size, rng):
    """Sub-sample a fixed-size candidate batch for one evaluation session."""
    n = len(ids)
    if batch_size >= n:
        idx = np.arange(n)
    else:
        idx = rng.choice(n, size=batch_size, replace=False)
    return [ids[i] for i in idx], F[idx].copy()
