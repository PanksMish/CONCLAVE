"""Shared numerical utilities."""
import numpy as np


def kendall_tau_distance(rank_a, rank_b):
    """Number of discordant pairs between two rankings over the same item set.

    rank_a, rank_b: dict item -> position (0 = best) or lists of items in order.
    Returns raw discordant-pair count (not normalized).
    """
    if isinstance(rank_a, (list, tuple)):
        rank_a = {item: i for i, item in enumerate(rank_a)}
    if isinstance(rank_b, (list, tuple)):
        rank_b = {item: i for i, item in enumerate(rank_b)}

    items = list(rank_a.keys())
    n = len(items)
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_order = rank_a[items[i]] - rank_a[items[j]]
            b_order = rank_b[items[i]] - rank_b[items[j]]
            if a_order * b_order < 0:
                discordant += 1
    return discordant


def normalized_kendall_tau_distance(rank_a, rank_b):
    n = len(rank_a)
    max_pairs = n * (n - 1) / 2.0
    if max_pairs == 0:
        return 0.0
    return kendall_tau_distance(rank_a, rank_b) / max_pairs


def minmax_scale(x, eps=1e-12):
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < eps:
        return np.full_like(x, 0.5)
    return (x - lo) / (hi - lo)


def rng_for_session(seed, session_idx):
    """Deterministic per-session RNG stream derived from a master seed."""
    return np.random.default_rng([seed, session_idx])
