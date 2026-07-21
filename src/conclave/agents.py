"""Specialized LLM-agent simulator.

Each agent ak has an objective emphasis vector lambda_k (Eq. 5): weight 0.70 on
its primary objective, 0.10 uniformly on the other three (paper's setting).
`propose()` returns a ranking over the candidate batch plus a confidence score,
mimicking what an LLM advisor would return after being prompted with the batch
and its role. Reasoning noise + occasional rank perturbations + an agent-quality
factor stand in for LLM sampling variance and imperfect reasoning, while keeping
everything deterministic given a seed (no external API calls).
"""
import numpy as np
from .objectives import OBJECTIVE_NAMES


PRIMARY_WEIGHT = 0.70
SECONDARY_WEIGHT = 0.30 / 3.0


class LLMAgent:
    def __init__(self, name, primary_idx, n_objectives=4,
                 reasoning_noise=0.06, quality=1.0, contrarian=False):
        self.name = name
        self.primary_idx = primary_idx
        self.lam = np.full(n_objectives, SECONDARY_WEIGHT)
        self.lam[primary_idx] = PRIMARY_WEIGHT
        self.reasoning_noise = reasoning_noise
        self.quality = quality          # 1.0 = nominal; <1 = "weak model" perturbation
        self.contrarian = contrarian    # if True, roughly inverts its own ranking

    def score(self, F, rng, noise_scale=1.0):
        """Weighted linear score used to derive this agent's proposed ranking."""
        base = F @ self.lam
        noise = rng.normal(0, noise_scale * self.reasoning_noise / max(self.quality, 1e-3), size=F.shape[0])
        s = base + noise
        if self.contrarian:
            s = -s + rng.normal(0, 0.02 * noise_scale, size=F.shape[0])
        return s

    def propose(self, ids, F, rng, drop_prob=0.0, noise_scale=1.0):
        """Return (ranking, confidence). ranking = list of ids best->worst.
        drop_prob simulates the 'dropped agent' perturbation: with that
        probability the agent abstains (returns None ranking, confidence 0).
        noise_scale < 1 models an agent that has refined its view across
        negotiation rounds (used to produce genuine round-to-round convergence)."""
        if drop_prob > 0 and rng.random() < drop_prob:
            return None, 0.0
        s = self.score(F, rng, noise_scale=noise_scale)
        order = np.argsort(-s)
        ranking = [ids[i] for i in order]
        # confidence: higher when this agent's score spread is large (self-consistent),
        # scaled by quality, with mild stochasticity that also decays with noise_scale
        spread = (s.max() - s.min()) if len(s) > 1 else 1.0
        conf = np.clip(0.5 + 0.4 * np.tanh(spread) * self.quality + rng.normal(0, 0.05 * noise_scale), 0.05, 0.99)
        return ranking, float(conf)


def default_agent_team(reasoning_noise=0.06):
    return [
        LLMAgent("Return", 0, reasoning_noise=reasoning_noise),
        LLMAgent("Risk", 1, reasoning_noise=reasoning_noise),
        LLMAgent("Cost", 2, reasoning_noise=reasoning_noise),
        LLMAgent("Consistency", 3, reasoning_noise=reasoning_noise),
    ]


def perturbed_agent_team(scenario, reasoning_noise=0.06):
    """Build an agent team under one of the robustness perturbation scenarios."""
    team = default_agent_team(reasoning_noise=reasoning_noise)
    if scenario == "nominal":
        return team, dict(drop_prob=0.0)
    if scenario == "dropped_agent":
        return team, dict(drop_prob=0.35)
    if scenario == "noisy_ranking":
        for a in team:
            a.reasoning_noise *= 4.0
        return team, dict(drop_prob=0.0)
    if scenario == "confidence_noise":
        # handled by caller adding extra noise to confidence; agents unchanged
        return team, dict(drop_prob=0.0, confidence_noise=0.25)
    if scenario == "weak_model":
        for a in team:
            a.quality = 0.4
        return team, dict(drop_prob=0.0)
    if scenario == "contrarian_agent":
        team[-1].contrarian = True
        return team, dict(drop_prob=0.0)
    raise ValueError(f"unknown scenario {scenario}")
