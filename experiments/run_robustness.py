"""Robustness study: compare CONCLAVE vs the strongest baseline (DM, debate-
style multi-agent consensus) under four perturbation scenarios: dropped agent,
noisy ranking, confidence noise, weak model, and a contrarian agent.
Produces results/robustness.csv with per-scenario HV5/NSW degradation
relative to the nominal (unperturbed) setting.

Usage:
    python experiments/run_robustness.py --data data/funds.csv --sessions 30 --out results/robustness.csv
"""
import argparse
import sys
import os
import zlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from conclave.objectives import get_candidate_batch, sample_batch  # noqa: E402
from conclave.pipeline import run_conclave  # noqa: E402
from conclave.metrics import compute_all_metrics  # noqa: E402
from conclave.agents import perturbed_agent_team  # noqa: E402
from conclave.baselines.debate_mediator import _debate_shortlist_fn  # noqa: E402
from conclave.baselines._harness import run_baseline  # noqa: E402

CATEGORIES = ["Large Cap", "Mid Cap", "Small Cap", "Flexi Cap", "ELSS", "Multi Cap"]
SCENARIOS = ["nominal", "dropped_agent", "noisy_ranking", "confidence_noise", "weak_model", "contrarian_agent"]


def stable_hash(s):
    return zlib.crc32(s.encode("utf-8")) % 100000


def run_method_under_scenario(method, ids, F, scenario, ks, max_rounds, rng):
    team, agent_kwargs = perturbed_agent_team(scenario)
    if method == "CONCLAVE":
        res = run_conclave(ids, F, agents=team, ks=ks, max_rounds=max_rounds, rng=rng, agent_kwargs=agent_kwargs)
    elif method == "DM":
        res = run_baseline(ids, F, _debate_shortlist_fn, agents=team, ks=ks, max_rounds=max_rounds,
                            rng=rng, agent_kwargs=agent_kwargs)
    else:
        raise ValueError(method)
    return compute_all_metrics(res, ids, F)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/funds.csv")
    ap.add_argument("--sessions", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--ks", type=int, default=5)
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/robustness.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    for category in CATEGORIES:
        ids, F = get_candidate_batch(df, category)
        for session in range(args.sessions):
            session_seed = args.seed * 10000 + stable_hash(category) + session
            rng = np.random.default_rng(session_seed)
            b_ids, b_F = sample_batch(ids, F, min(args.batch_size, len(ids)), rng)
            for method in ["CONCLAVE", "DM"]:
                for scenario in SCENARIOS:
                    m = run_method_under_scenario(
                        method, b_ids, b_F, scenario, args.ks, args.max_rounds,
                        np.random.default_rng([session_seed, stable_hash(method), stable_hash(scenario)]))
                    m["method"] = method
                    m["scenario"] = scenario
                    m["category"] = category
                    m["session"] = session
                    rows.append(m)
        print(f"[{category}] done", flush=True)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"\nWrote {len(out_df)} rows to {args.out}")

    nominal = out_df[out_df.scenario == "nominal"].groupby("method")[["HV5", "NSW"]].mean()
    degrad_rows = []
    for method in ["CONCLAVE", "DM"]:
        for scenario in SCENARIOS:
            if scenario == "nominal":
                continue
            sub = out_df[(out_df.method == method) & (out_df.scenario == scenario)]
            hv_deg = 1 - (sub["HV5"].mean() / nominal.loc[method, "HV5"])
            nsw_deg = 1 - (sub["NSW"].mean() / nominal.loc[method, "NSW"])
            degrad_rows.append(dict(method=method, scenario=scenario,
                                     hv_degradation_pct=100 * hv_deg, nsw_degradation_pct=100 * nsw_deg))
    degrad_df = pd.DataFrame(degrad_rows)
    degrad_df.to_csv(args.out.replace(".csv", "_degradation.csv"), index=False)
    print("\n=== Robustness degradation (%) relative to nominal ===")
    print(degrad_df.round(2))


if __name__ == "__main__":
    main()
