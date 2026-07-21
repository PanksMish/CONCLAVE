"""Shared experimental harness: every baseline is evaluated inside the same
multi-round negotiation loop (fresh agent proposals each round + Eq. 32
stability check) so that Rounds-to-Convergence, NSW, and JF are computed on a
level playing field with CONCLAVE. Only the shortlist-construction rule
differs between methods.
"""
import numpy as np

from .. import bargaining as barg
from .. import consensus as cons
from .. import stability as stab
from ..agents import default_agent_team


def _own_baseline_shortlist(psi_k, ks):
    order = np.argsort(-psi_k)
    return list(order[:ks])


def run_baseline(ids, F, shortlist_fn, agents=None, ks=5, max_rounds=5,
                  tau_s=0.8, tau_g=0.03, rng=None, agent_kwargs=None,
                  use_pareto_filter=False, eps_archive=0.02):
    """Generic harness. `shortlist_fn(C_ids, C_F, rankings, confidences,
    psi_matrix, d, ks, rng) -> list[ids]` implements the method-specific rule.
    """
    from .. import pareto as pareto_mod

    rng = rng or np.random.default_rng(0)
    agents = agents or default_agent_team()
    agent_kwargs = agent_kwargs or {}
    K = len(agents)

    if use_pareto_filter:
        C_ids, C_F, _ = pareto_mod.epsilon_archive(ids, F, eps=eps_archive)
        if len(C_ids) < ks:
            C_ids, C_F = list(ids), F.copy()
    else:
        C_ids, C_F = list(ids), F.copy()

    z_star = barg.ideal_point(C_F)

    history = {"shortlists": [], "gains": []}
    S_prev, gain_prev = None, None
    rounds_used = 0
    final_state = None

    noise_decay = agent_kwargs.get("noise_decay", 0.6)
    data_jitter_std = agent_kwargs.get("data_jitter_std", 0.02)

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

        psi_matrix = np.stack([barg.agent_utilities(C_F, agent.lam, z_star) for agent in agents])
        baseline_idx = [_own_baseline_shortlist(psi_matrix[k], ks) for k in range(K)]
        d = barg.disagreement_point(psi_matrix, baseline_idx)

        # decaying data jitter: models a method re-observing slightly noisy,
        # progressively-settling market data each round; lets purely data-driven
        # (non-agentic) baselines like QC/TOPSIS/VIKOR also exhibit genuine
        # round-to-round convergence rather than being trivially stable at t=1.
        C_F_round = C_F + rng.normal(0, data_jitter_std * noise_scale, size=C_F.shape)

        S_t = shortlist_fn(C_ids, C_F_round, rankings, confidences, psi_matrix, d, ks, rng)

        idx_of = {u: i for i, u in enumerate(C_ids)}
        S_t_idx = [idx_of[u] for u in S_t if u in idx_of]
        gain_t = barg.nash_gain(psi_matrix, S_t_idx, d, confidences)

        history["shortlists"].append(S_t)
        history["gains"].append(gain_t)

        final_state = dict(
            shortlist=S_t,
            consensus_ranking=cons.weighted_kendall_consensus_ranking(C_ids, rankings, confidences)[0],
            psi_matrix=psi_matrix, d=d, confidences=confidences,
            C_ids=C_ids, C_F=C_F, agent_rankings=rankings,
        )

        converged = stab.check_convergence(S_t, S_prev, gain_t, gain_prev, tau_s=tau_s, tau_g=tau_g)
        S_prev, gain_prev = S_t, gain_t
        if converged:
            break

    final_state["rounds_to_convergence"] = rounds_used
    final_state["history"] = history
    return final_state
