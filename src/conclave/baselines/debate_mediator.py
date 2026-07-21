"""Debate-style multi-agent LLM consensus (DM): agents iteratively exchange
"arguments" by nudging their score vectors toward the group's confidence-
weighted mean (opinion pooling / persuasion dynamics), then a final
confidence-weighted vote selects the shortlist. This is the strongest
non-CONCLAVE baseline family described in the abstract ("debate-style LLM
consensus"), but -- unlike CONCLAVE -- it has no explicit Pareto constraint,
no bargaining-fairness objective, and no stability-aware stopping rule of its
own (it is still evaluated inside the shared harness's stopping check for a
fair RTC comparison).
"""
import numpy as np
from ._harness import run_baseline
from ..agents import default_agent_team

DEBATE_ROUNDS = 3
PERSUASION_RATE = 0.35


def _debate_shortlist_fn(C_ids, C_F, rankings, confidences, psi_matrix, d, ks, rng):
    K, n = psi_matrix.shape
    scores = psi_matrix.copy()  # each agent's private utility estimate per candidate
    w = np.array(confidences)
    w = w / (w.sum() + 1e-12)

    for _ in range(DEBATE_ROUNDS):
        group_mean = (w[:, None] * scores).sum(axis=0, keepdims=True)  # (1, n)
        scores = (1 - PERSUASION_RATE) * scores + PERSUASION_RATE * group_mean
        scores = scores + rng.normal(0, 0.01, size=scores.shape)  # residual disagreement

    final_score = (w[:, None] * scores).sum(axis=0)
    order = np.argsort(-final_score)
    return [C_ids[i] for i in order[:ks]]


def run(ids, F, **kwargs):
    agents = default_agent_team()
    return run_baseline(ids, F, _debate_shortlist_fn, agents=agents, use_pareto_filter=False, **kwargs)
