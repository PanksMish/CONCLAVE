import numpy as np
from conclave.metrics import hypervolume, shortlist_stability_series, pareto_purity


def test_hypervolume_single_point():
    pts = np.array([[0.5, 0.5]])
    hv = hypervolume(pts)
    assert abs(hv - 0.25) < 1e-9


def test_hypervolume_dominated_point_no_extra_contribution():
    pts = np.array([[0.5, 0.5], [0.2, 0.2]])
    hv = hypervolume(pts)
    hv_single = hypervolume(np.array([[0.5, 0.5]]))
    assert abs(hv - hv_single) < 1e-9


def test_shortlist_stability_identical_rounds():
    hist = [["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c"]]
    assert shortlist_stability_series(hist) == 1.0


def test_pareto_purity_full():
    ids = ["a", "b", "c"]
    F = np.array([[0.9, 0.9], [0.1, 0.1], [0.5, 0.5]])
    assert pareto_purity(["a"], ids, F) == 1.0
    assert pareto_purity(["b"], ids, F) == 0.0
