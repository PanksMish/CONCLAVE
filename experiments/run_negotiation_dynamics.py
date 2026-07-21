"""Track hypervolume and shortlist stability round-by-round for CONCLAVE vs
SA-LLM, Heuristic Mediator, and Weighted Kemeny, to visualize negotiation
dynamics (paper's Figure 4).

Usage:
    python experiments/run_negotiation_dynamics.py --data data/funds.csv --sessions 30 --out results/dynamics.csv
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
from conclave.metrics import hypervolume_for_ids  # noqa: E402
from conclave.baselines import sa_llm, heuristic_mediator, rank_aggregation  # noqa: E402
from conclave.baselines._harness import run_baseline  # noqa: E402

CATEGORIES = ["Large Cap", "Mid Cap", "Small Cap", "Flexi Cap", "ELSS", "Multi Cap"]
MAX_ROUNDS = 5


def stable_hash(s):
    return zlib.crc32(s.encode("utf-8")) % 100000


def _stability_series(shortlists):
    return [1.0] + [
        len(set(a) & set(b)) / len(set(a) | set(b))
        for a, b in zip(shortlists[:-1], shortlists[1:])
    ]


def track_conclave(ids, F, ks, rng):
    res = run_conclave(ids, F, ks=ks, max_rounds=MAX_ROUNDS, rng=rng)
    hist = res["history"]
    hv_series = [hypervolume_for_ids(s, ids, F) for s in hist["shortlists"]]
    return hv_series, _stability_series(hist["shortlists"])


def track_baseline(shortlist_fn, ids, F, ks, rng):
    res = run_baseline(ids, F, shortlist_fn, ks=ks, max_rounds=MAX_ROUNDS, rng=rng)
    hist = res["history"]
    hv_series = [hypervolume_for_ids(s, ids, F) for s in hist["shortlists"]]
    return hv_series, _stability_series(hist["shortlists"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/funds.csv")
    ap.add_argument("--sessions", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--ks", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/dynamics.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    methods = {
        "CONCLAVE": None,  # special-cased
        "SA": sa_llm.make_shortlist_fn(),
        "HM": heuristic_mediator._hm_shortlist_fn,
        "WK": rank_aggregation._kemeny_shortlist_fn,
    }

    rows = []
    for category in CATEGORIES:
        ids, F = get_candidate_batch(df, category)
        for session in range(args.sessions):
            session_seed = args.seed * 10000 + stable_hash(category) + session
            rng = np.random.default_rng(session_seed)
            b_ids, b_F = sample_batch(ids, F, min(args.batch_size, len(ids)), rng)
            for method, fn in methods.items():
                mrng = np.random.default_rng([session_seed, stable_hash(method)])
                if method == "CONCLAVE":
                    hv_series, stab_series = track_conclave(b_ids, b_F, args.ks, mrng)
                else:
                    hv_series, stab_series = track_baseline(fn, b_ids, b_F, args.ks, mrng)
                for round_idx, (hv, stab) in enumerate(zip(hv_series, stab_series), start=1):
                    rows.append(dict(method=method, category=category, session=session,
                                      round=round_idx, HV5=hv, stability=stab))
        print(f"[{category}] done", flush=True)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"\nWrote {len(out_df)} rows to {args.out}")

    print("\n=== Mean HV5 and stability by round ===")
    print(out_df.groupby(["method", "round"])[["HV5", "stability"]].mean().round(3))


if __name__ == "__main__":
    main()
