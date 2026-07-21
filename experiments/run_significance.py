"""Paired significance testing between CONCLAVE and the strongest baselines,
with Holm-Bonferroni correction for multiple comparisons (paper's Table 3).

Usage:
    python experiments/run_significance.py --table1 results/table1.csv --out results/significance.csv
"""
import argparse
import sys
import os
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

METRICS = ["PP5", "HV5", "WKA", "NSW", "JF"]
COMPARISON_BASELINES = ["WK", "TOPSIS", "DM"]


def holm_bonferroni(pvalues):
    """Return Holm-Bonferroni-corrected p-values, same order as input."""
    pvalues = np.asarray(pvalues, dtype=float)
    m = len(pvalues)
    order = np.argsort(pvalues)
    corrected = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvalues[idx]
        running_max = max(running_max, adj)
        corrected[idx] = min(running_max, 1.0)
    return corrected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table1", default="results/table1.csv")
    ap.add_argument("--out", default="results/significance.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.table1)
    # pair sessions by (category, session) so the paired t-test compares like-for-like batches
    df = df.set_index(["category", "session", "method"])

    rows = []
    raw_pvals = []
    for baseline in COMPARISON_BASELINES:
        for metric in METRICS:
            conclave_vals = df.xs("CONCLAVE", level="method")[metric]
            baseline_vals = df.xs(baseline, level="method")[metric]
            common_idx = conclave_vals.index.intersection(baseline_vals.index)
            a = conclave_vals.loc[common_idx].values
            b = baseline_vals.loc[common_idx].values
            t_stat, p_val = stats.ttest_rel(a, b)
            rows.append(dict(comparison=f"CONCLAVE vs {baseline}", metric=metric,
                              mean_diff=float(np.mean(a - b)), t_stat=float(t_stat), p_raw=float(p_val)))
            raw_pvals.append(p_val)

    corrected = holm_bonferroni(raw_pvals)
    for row, p_corr in zip(rows, corrected):
        row["p_holm_bonferroni"] = p_corr
        row["significant_at_0.05"] = p_corr < 0.05

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} rows to {args.out}\n")

    pivot = out_df.pivot(index="comparison", columns="metric", values="p_holm_bonferroni")
    pivot = pivot[METRICS]
    print("=== Table 3: Holm-Bonferroni corrected p-values ===")
    print(pivot.round(4))


if __name__ == "__main__":
    main()
