import numpy as np
from conclave import bargaining as barg


def test_tchebycheff_utility_monotonic():
    F = np.array([[0.9, 0.9], [0.5, 0.5], [0.1, 0.1]])
    lam = np.array([0.5, 0.5])
    z_star = barg.ideal_point(F)
    psi = barg.agent_utilities(F, lam, z_star)
    assert psi[0] > psi[1] > psi[2]


def test_nash_gain_increases_with_better_shortlist():
    psi_matrix = np.array([
        [0.9, 0.5, 0.1],
        [0.2, 0.6, 0.9],
    ])
    d = np.array([0.1, 0.1])
    confidences = [0.8, 0.8]
    gain_good = barg.nash_gain(psi_matrix, [0, 2], d, confidences)
    gain_bad = barg.nash_gain(psi_matrix, [0], d, confidences)
    assert isinstance(gain_good, float)
    assert isinstance(gain_bad, float)


def test_jain_fairness_bounds():
    assert barg.jain_fairness([1, 1, 1]) == 1.0
    assert 0 < barg.jain_fairness([1, 0.01, 0.01]) < 1.0
