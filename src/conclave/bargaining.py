"""Agent-specific utility surrogates (augmented Tchebycheff) and bargaining rules."""
import numpy as np

RHO = 1e-4  # augmentation constant (paper's setting)
EPS_STAB = 1e-6


def ideal_point(F):
    return F.max(axis=0)


def tchebycheff_disutility(F, lam, z_star, rho=RHO):
    """g_k(u) for every candidate row of F (Eq. 18)."""
    gap = np.abs(z_star[None, :] - F)          # (N, M)
    weighted_gap = lam[None, :] * gap
    max_term = weighted_gap.max(axis=1)
    sum_term = weighted_gap.sum(axis=1)
    return max_term + rho * sum_term


def agent_utilities(F, lam, z_star, rho=RHO):
    """psi_k(u) = exp(-g_k(u)) for every candidate (Eq. 19)."""
    g = tchebycheff_disutility(F, lam, z_star, rho=rho)
    return np.exp(-g)


def shortlist_utility(psi_k, shortlist_idx):
    """U_k(S) (Eq. 6/20)."""
    if len(shortlist_idx) == 0:
        return 0.0
    return float(np.mean(psi_k[list(shortlist_idx)]))


def disagreement_point(psi_matrix, baseline_idx_per_agent):
    """d_k = U_k(S_base_k). baseline_idx_per_agent: list of index-arrays, one per agent,
    e.g. each agent's own top-ks self-preferred shortlist."""
    K = psi_matrix.shape[0]
    d = np.zeros(K)
    for k in range(K):
        d[k] = shortlist_utility(psi_matrix[k], baseline_idx_per_agent[k])
    return d


def nash_gain(psi_matrix, shortlist_idx, d, confidences, eps=EPS_STAB):
    """Confidence-weighted Nash-style bargaining gain G_barg(S) (Eq. 8)."""
    K = psi_matrix.shape[0]
    total = 0.0
    for k in range(K):
        Uk = shortlist_utility(psi_matrix[k], shortlist_idx)
        surplus = Uk - d[k] + eps
        surplus = max(surplus, 1e-9)  # numerical guard
        total += confidences[k] * np.log(surplus)
    return total


def agent_utility_vector(psi_matrix, shortlist_idx):
    K = psi_matrix.shape[0]
    return np.array([shortlist_utility(psi_matrix[k], shortlist_idx) for k in range(K)])


def kalai_smorodinsky_gain(psi_matrix, shortlist_idx, d, confidences, ideal_utils):
    """K-S bargaining score: rewards shortlists whose per-agent surplus ratios
    (relative to each agent's own achievable ideal utility) are balanced and high."""
    u = agent_utility_vector(psi_matrix, shortlist_idx)
    ratios = np.clip((u - d) / np.clip(ideal_utils - d, 1e-6, None), 0, None)
    weighted_mean = np.sum(confidences * ratios) / np.sum(confidences)
    balance_penalty = np.std(ratios)
    return weighted_mean - 0.5 * balance_penalty


def egalitarian_gain(psi_matrix, shortlist_idx, d, confidences):
    """Egalitarian (maximin) bargaining score: maximize the worst-off agent's surplus."""
    u = agent_utility_vector(psi_matrix, shortlist_idx)
    surplus = u - d
    return float(np.min(surplus))


def jain_fairness(utilities):
    utilities = np.asarray(utilities, dtype=float)
    utilities = np.clip(utilities, 1e-9, None)
    n = len(utilities)
    return float((utilities.sum() ** 2) / (n * np.sum(utilities ** 2)))


def nash_social_welfare(utilities):
    utilities = np.clip(np.asarray(utilities, dtype=float), 1e-9, None)
    return float(np.exp(np.mean(np.log(utilities))))
