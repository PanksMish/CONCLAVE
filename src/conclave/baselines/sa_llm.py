"""Single-Agent LLM (SA-LLM): one generalist financial-advisor agent produces
the shortlist directly, without multi-agent negotiation. Its shortlist is still
scored against the standard 4-specialized-agent utility framework (psi_matrix,
d, confidences) so that NSW/JF/WKA remain comparable across all methods in
Table 1 -- exactly as the paper does for every baseline.
"""
import numpy as np
from ._harness import run_baseline
from ..agents import LLMAgent


def generalist_agent(reasoning_noise=0.06):
    agent = LLMAgent("Generalist", primary_idx=0, reasoning_noise=reasoning_noise)
    agent.lam = np.full(4, 0.25)  # balanced weighting, no specialization
    return agent


def make_shortlist_fn():
    agent = generalist_agent()

    def _sa_shortlist_fn(C_ids, C_F, rankings, confidences, psi_matrix, d, ks, rng):
        ranking, _ = agent.propose(C_ids, C_F, rng)
        return ranking[:ks]

    return _sa_shortlist_fn


def run(ids, F, **kwargs):
    return run_baseline(ids, F, make_shortlist_fn(), use_pareto_filter=False, **kwargs)
