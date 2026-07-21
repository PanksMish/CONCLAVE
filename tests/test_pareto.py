import numpy as np
from conclave.pareto import dominance_mask, epsilon_archive


def test_dominance_simple():
    F = np.array([
        [0.9, 0.9],
        [0.5, 0.5],
        [0.9, 0.1],
    ])
    mask = dominance_mask(F)
    assert mask[0] == True
    assert mask[1] == False  # dominated by row 0
    assert mask[2] == False  # dominated by row 0 (tie dim0, worse dim1)


def test_epsilon_archive_keeps_non_dominated():
    rng = np.random.default_rng(0)
    F = rng.uniform(0, 1, size=(50, 4))
    ids = [f"u{i}" for i in range(50)]
    out_ids, out_F, mask = epsilon_archive(ids, F, eps=0.05)
    assert len(out_ids) <= mask.sum()
    assert len(out_ids) > 0
    # every archived point must itself be non-dominated
    full_mask = dominance_mask(F)
    archived_idx = [ids.index(u) for u in out_ids]
    assert all(full_mask[i] for i in archived_idx)
