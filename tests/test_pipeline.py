import numpy as np
from conclave.pipeline import run_conclave


def _toy_batch(n=30, m=4, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.uniform(0, 1, size=(n, m))
    ids = [f"u{i}" for i in range(n)]
    return ids, F


def test_run_conclave_returns_valid_shortlist():
    ids, F = _toy_batch()
    res = run_conclave(ids, F, ks=5, max_rounds=4, rng=np.random.default_rng(1))
    assert len(res["shortlist"]) == 5
    assert len(set(res["shortlist"])) == 5
    assert all(u in ids for u in res["shortlist"])
    assert res["rounds_to_convergence"] >= 1


def test_ablate_pareto_uses_full_batch():
    ids, F = _toy_batch()
    res = run_conclave(ids, F, ks=5, max_rounds=3, ablate_pareto=True, rng=np.random.default_rng(1))
    assert set(res["C_ids"]) == set(ids)


def test_ablate_stability_runs_full_rounds():
    ids, F = _toy_batch()
    res = run_conclave(ids, F, ks=5, max_rounds=4, ablate_stability=True, rng=np.random.default_rng(1))
    assert res["rounds_to_convergence"] == 4
