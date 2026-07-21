"""Stability-aware stopping rule."""


def jaccard_stability(S_t, S_prev):
    if S_prev is None:
        return 0.0
    a, b = set(S_t), set(S_prev)
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def stability_loss(S_t, S_prev):
    return 1.0 - jaccard_stability(S_t, S_prev)


def check_convergence(S_t, S_prev, gain_t, gain_prev, tau_s=0.8, tau_g=0.03):
    """Eq. 32: converged iff stability loss small AND bargaining gain has
    stopped improving materially. tau_g is a *relative* tolerance on the
    bargaining gain (|Delta G| / |G_prev|), since the raw Nash-gain magnitude
    scales with team size and confidence-weighting and is not on a fixed scale."""
    if S_prev is None or gain_prev is None:
        return False
    stab = jaccard_stability(S_t, S_prev)
    rel_gain_delta = abs(gain_t - gain_prev) / max(abs(gain_prev), 1e-6)
    return (stab >= tau_s) and (rel_gain_delta <= tau_g)
