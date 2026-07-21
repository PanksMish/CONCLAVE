"""Compare Nash, Kalai-Smorodinsky, and Egalitarian bargaining rules inside
the CONCLAVE pipeline. Produces results/bargaining_rules.csv with per-objective
agent utility profiles and overall per-session utility.

Usage:
    python experiments/run_bargaining_comparison.py --data data/funds.csv --sessions 30 --out results/bargaining_rules.csv
"""
import argparse
import sys
import os
import zlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from conclave.objectives import get_candidate_batch, sample_batch, OBJECTIVE_NAMES  # noqa: E402
from conclave.pipeline import run_conclave  # noqa: E402
from conclave import bargaining as barg  # noqa: E402

CATEGORIES = ["Large Cap", "Mid Cap", "Small Cap", "Flexi Cap", "ELSS", "Multi Cap"]
RULES = ["nash", "kalai", "egalitarian"]


def stable_hash(s):
    return zlib.crc32(s.encode("utf-8")) % 100000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/funds.csv")
    ap.add_argument("--sessions", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--ks", type=int, default=5)
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/bargaining_rules.csv")
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
            for rule in RULES:
                res = run_conclave(b_ids, b_F, ks=args.ks, max_rounds=args.max_rounds,
                                    bargaining_rule=rule,
                                    rng=np.random.default_rng([session_seed, stable_hash(rule)]))
                idx_of = {u: i for i, u in enumerate(res["C_ids"])}
                s_idx = [idx_of[u] for u in res["shortlist"] if u in idx_of]
                utilities = barg.agent_utility_vector(res["psi_matrix"], s_idx)
                overall = float(np.mean(utilities))
                row = dict(rule=rule, category=category, session=session, overall_utility=overall)
                for name, u in zip(OBJECTIVE_NAMES, utilities):
                    row[f"utility_{name}"] = float(u)
                rows.append(row)
        print(f"[{category}] done", flush=True)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"\nWrote {len(out_df)} rows to {args.out}")

    print("\n=== Bargaining rule comparison (mean over sessions) ===")
    cols = ["overall_utility"] + [f"utility_{n}" for n in OBJECTIVE_NAMES]
    print(out_df.groupby("rule")[cols].mean().round(3))


if __name__ == "__main__":
    main()
