"""End-to-end CONCLAVE orchestrator (Eq. 12 decomposition)."""
import numpy as np

from . import pareto
from . import bargaining as barg
from . import consensus as cons
from . import stability as stab
from .agents import default_agent_team


def _own_baseline_shortlist(psi_k, ks):
    order = np.argsort(-psi_k)
    return list(order[:ks])


def run_conclave(ids, F, agents=None, ks=5, max_rounds=5, eps_archive=0.02, alpha=0.5,
                  tau_s=0.8, tau_g=0.03, ablate_pareto=False, ablate_bargaining=False,
                  ablate_stability=False, bargaining_rule="nash", rng=None,
                  agent_kwargs=None):
    """Run the full CONCLAVE negotiation loop over one candidate batch.

    Returns a dict with the final shortlist plus full per-round history, so
    downstream code can compute every metric in the paper (PP5, HV5, WKA,
    NSW, JF, SS, RTC) from a single call.
    """
    rng = rng or np.random.default_rng(0)
    agents = agents or default_agent_team()
    agent_kwargs = agent_kwargs or {}
    K = len(agents)

    # --- Module 1: Pareto-constrained candidate filtering ---
    if ablate_pareto:
        C_ids, C_F = list(ids), F.copy()
    else:
        C_ids, C_F, _ = pareto.epsilon_archive(ids, F, eps=eps_archive)
        if len(C_ids) < ks:  # guard: archive too small, fall back to full batch
            C_ids, C_F = list(ids), F.copy()

    z_star = barg.ideal_point(F)  # ideal point over the FULL batch, matching the baseline harness

    # Per-agent utilities computed once over the full batch (Eq. 17-19); CONCLAVE's
    # shortlist selection is still restricted to C, but "how good is a candidate"
    # is measured on the same yardstick used to evaluate every baseline.
    idx_of_full = {u: i for i, u in enumerate(ids)}
    C_mask = np.array([idx_of_full[u] for u in C_ids])

    gain_fn, gain_kwargs_base = _resolve_bargaining_rule(bargaining_rule, C_F, agents, z_star)

    history = {"shortlists": [], "gains": [], "stability": [], "confidences": []}
    S_prev, gain_prev = None, None
    rounds_used = 0
    final_state = None

    noise_decay = agent_kwargs.get("noise_decay", 0.6)

    for t in range(1, max_rounds + 1):
        rounds_used = t
        noise_scale = noise_decay ** (t - 1)
        rankings, confidences = [], []
        for agent in agents:
            ranking, conf = agent.propose(C_ids, C_F, rng, drop_prob=agent_kwargs.get("drop_prob", 0.0),
                                           noise_scale=noise_scale)
            if agent_kwargs.get("confidence_noise"):
                conf = float(np.clip(conf + rng.normal(0, agent_kwargs["confidence_noise"]), 0.02, 0.99))
            rankings.append(ranking)
            confidences.append(conf)

        # --- Module 2: Agent preference grounding ---
        psi_matrix_full = np.stack([
            barg.agent_utilities(F, agent.lam, z_star) for agent in agents
        ])
        psi_matrix = psi_matrix_full[:, C_mask]  # restricted to the Pareto-filtered candidate set
        baseline_idx_full = [_own_baseline_shortlist(psi_matrix_full[k], ks) for k in range(K)]
        d = barg.disagreement_point(psi_matrix_full, baseline_idx_full)

        # --- Module 3: ordinal consensus + bargaining-aware shortlist ---
        consensus_ranking, W = cons.weighted_kendall_consensus_ranking(C_ids, rankings, confidences)

        gain_kwargs = dict(gain_kwargs_base)
        if bargaining_rule == "kalai":
            ideal_utils = np.array([psi_matrix[k].max() for k in range(K)])
            gain_kwargs["ideal_utils"] = ideal_utils

        S_t = cons.build_shortlist(C_ids, consensus_ranking, psi_matrix, d, confidences, ks,
                                    F=C_F, alpha=alpha, use_bargaining=not ablate_bargaining,
                                    gain_fn=gain_fn, gain_fn_kwargs=gain_kwargs)

        idx_of = {u: i for i, u in enumerate(C_ids)}
        S_t_idx = [idx_of[u] for u in S_t]
        gain_t = barg.nash_gain(psi_matrix, S_t_idx, d, confidences)  # always report Nash gain for RTC comparability

        stability_now = stab.jaccard_stability(S_t, S_prev)
        history["shortlists"].append(S_t)
        history["gains"].append(gain_t)
        history["stability"].append(stability_now)
        history["confidences"].append(confidences)

        final_state = dict(
            shortlist=S_t, consensus_ranking=consensus_ranking, psi_matrix=psi_matrix,
            d=d, confidences=confidences, C_ids=C_ids, C_F=C_F, agent_rankings=rankings,
        )

        converged = (not ablate_stability) and stab.check_convergence(
            S_t, S_prev, gain_t, gain_prev, tau_s=tau_s, tau_g=tau_g)
        if ablate_stability:
            # always run to max_rounds when stability-aware stopping is disabled
            S_prev, gain_prev = S_t, gain_t
            continue
        if converged:
            S_prev, gain_prev = S_t, gain_t
            break
        S_prev, gain_prev = S_t, gain_t

    final_state["rounds_to_convergence"] = rounds_used
    final_state["history"] = history
    return final_state


def _resolve_bargaining_rule(rule, C_F, agents, z_star):
    if rule == "nash":
        return barg.nash_gain, {}
    if rule == "kalai":
        return barg.kalai_smorodinsky_gain, {}
    if rule == "egalitarian":
        return barg.egalitarian_gain, {}
    raise ValueError(f"unknown bargaining rule {rule}")
