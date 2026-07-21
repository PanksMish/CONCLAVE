"""Run the main experiment: every method (CONCLAVE + 8 baselines) evaluated
across N independent sessions per SEBI category. Produces results/table1.csv
with one row per (method, category, session) plus the aggregated Table 1.

Usage:
    python experiments/run_main_experiment.py --data data/funds.csv --sessions 30 --out results/table1.csv
"""
import argparse
import sys
import os
import zlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from conclave.objectives import get_candidate_batch, sample_batch, OBJECTIVE_COLUMNS  # noqa: E402
from conclave.pipeline import run_conclave  # noqa: E402
from conclave.metrics import compute_all_metrics  # noqa: E402
from conclave.baselines import METHOD_RUNNERS  # noqa: E402

CATEGORIES = ["Large Cap", "Mid Cap", "Small Cap", "Flexi Cap", "ELSS", "Multi Cap"]


def stable_hash(s):
    """Deterministic string hash (Python's built-in hash() is randomized per
    process via PYTHONHASHSEED, which would break reproducibility)."""
    return zlib.crc32(s.encode("utf-8")) % 100000


def run_all_methods(ids, F, ks, max_rounds, session_seed):
    rows = []
    conclave_res = run_conclave(ids, F, ks=ks, max_rounds=max_rounds,
                                 rng=np.random.default_rng([session_seed, 1]))
    m = compute_all_metrics(conclave_res, ids, F)
    m["RTC"] = conclave_res["rounds_to_convergence"]
    m["method"] = "CONCLAVE"
    rows.append(m)

    for name, fn in METHOD_RUNNERS.items():
        res = fn(ids, F, ks=ks, max_rounds=max_rounds, rng=np.random.default_rng([session_seed, 2, stable_hash(name)]))
        m = compute_all_metrics(res, ids, F)
        m["RTC"] = res["rounds_to_convergence"]
        m["method"] = name
        rows.append(m)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/funds.csv")
    ap.add_argument("--sessions", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--ks", type=int, default=5)
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/table1.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    all_rows = []
    for category in CATEGORIES:
        ids, F = get_candidate_batch(df, category)
        for session in range(args.sessions):
            session_seed = args.seed * 10000 + stable_hash(category) + session
            rng = np.random.default_rng(session_seed)
            b_ids, b_F = sample_batch(ids, F, min(args.batch_size, len(ids)), rng)
            rows = run_all_methods(b_ids, b_F, args.ks, args.max_rounds, session_seed)
            for r in rows:
                r["category"] = category
                r["session"] = session
            all_rows.extend(rows)
            print(f"[{category}] session {session+1}/{args.sessions} done", flush=True)

    out_df = pd.DataFrame(all_rows)
    cols = ["method", "category", "session", "PP5", "HV5", "Spread", "WKA", "NSW", "JF", "SS", "RTC"]
    out_df = out_df[cols]
    out_df.to_csv(args.out, index=False)
    print(f"\nWrote {len(out_df)} rows to {args.out}")

    print("\n=== Table 1 (mean ± std over all sessions/categories) ===")
    summary = out_df.groupby("method")[["PP5", "HV5", "WKA", "NSW", "JF", "SS", "RTC"]].agg(["mean", "std"])
    print(summary.round(3))
    summary.to_csv(args.out.replace(".csv", "_summary.csv"))


if __name__ == "__main__":
    main()
