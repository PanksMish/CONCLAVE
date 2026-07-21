"""Ablation study: CON-FULL vs CON-NP (no Pareto), CON-NB (no bargaining),
CON-NS (no stability). Produces results/ablation.csv.

Usage:
    python experiments/run_ablation.py --data data/funds.csv --sessions 30 --out results/ablation.csv
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

CATEGORIES = ["Large Cap", "Mid Cap", "Small Cap", "Flexi Cap", "ELSS", "Multi Cap"]

VARIANTS = {
    "CON-FULL": dict(ablate_pareto=False, ablate_bargaining=False, ablate_stability=False),
    "CON-NP": dict(ablate_pareto=True, ablate_bargaining=False, ablate_stability=False),
    "CON-NB": dict(ablate_pareto=False, ablate_bargaining=True, ablate_stability=False),
    "CON-NS": dict(ablate_pareto=False, ablate_bargaining=False, ablate_stability=True),
}


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
    ap.add_argument("--out", default="results/ablation.csv")
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
            for variant_name, flags in VARIANTS.items():
                res = run_conclave(b_ids, b_F, ks=args.ks, max_rounds=args.max_rounds,
                                    rng=np.random.default_rng([session_seed, stable_hash(variant_name)]),
                                    **flags)
                m = compute_all_metrics(res, b_ids, b_F)
                m["RTC"] = res["rounds_to_convergence"]
                m["variant"] = variant_name
                m["category"] = category
                m["session"] = session
                rows.append(m)
        print(f"[{category}] done", flush=True)

    out_df = pd.DataFrame(rows)
    cols = ["variant", "category", "session", "PP5", "HV5", "Spread", "WKA", "NSW", "JF", "SS", "RTC"]
    out_df = out_df[cols]
    out_df.to_csv(args.out, index=False)
    print(f"\nWrote {len(out_df)} rows to {args.out}")

    print("\n=== Ablation summary ===")
    summary = out_df.groupby("variant")[["PP5", "HV5", "NSW", "JF", "RTC"]].mean()
    print(summary.round(3))


if __name__ == "__main__":
    main()
