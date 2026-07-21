"""
Generate a synthetic-but-realistic universe of Indian open-ended equity mutual
funds across 6 SEBI categories, following the filtering protocol in the paper:
AUM >= 500 crore, >= 5 years of daily NAV history.

Output: a CSV with raw financial indicators AND the normalized 4-D objective
vector f(u) = [f_return, f_risk, f_cost, f_consistency] (Eq. 33).

Usage:
    python data/generate_data.py --out data/funds.csv --seed 42 --n-per-category 60
"""
import argparse
import numpy as np
import pandas as pd

CATEGORIES = ["Large Cap", "Mid Cap", "Small Cap", "Flexi Cap", "ELSS", "Multi Cap"]

# category-level generative priors: (return_mu, return_sigma, vol_mu, vol_sigma)
CATEGORY_PRIORS = {
    "Large Cap":  dict(cagr_mu=12.5, cagr_sd=2.5, vol_mu=14.0, vol_sd=2.5, exp_mu=1.1, exp_sd=0.35),
    "Mid Cap":    dict(cagr_mu=16.0, cagr_sd=4.0, vol_mu=19.0, vol_sd=3.5, exp_mu=1.4, exp_sd=0.35),
    "Small Cap":  dict(cagr_mu=18.5, cagr_sd=5.5, vol_mu=23.0, vol_sd=4.5, exp_mu=1.6, exp_sd=0.35),
    "Flexi Cap":  dict(cagr_mu=14.0, cagr_sd=3.0, vol_mu=16.0, vol_sd=3.0, exp_mu=1.2, exp_sd=0.35),
    "ELSS":       dict(cagr_mu=14.5, cagr_sd=3.2, vol_mu=16.5, vol_sd=3.0, exp_mu=1.1, exp_sd=0.30),
    "Multi Cap":  dict(cagr_mu=15.0, cagr_sd=3.5, vol_mu=17.5, vol_sd=3.2, exp_mu=1.3, exp_sd=0.35),
}


def _minmax(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < 1e-12:
        return np.ones_like(x) * 0.5
    return (x - lo) / (hi - lo)


def generate_category(category, n, rng):
    p = CATEGORY_PRIORS[category]
    aum = rng.lognormal(mean=np.log(1500), sigma=0.9, size=n) + 300  # crore
    history_years = rng.uniform(4.0, 15.0, size=n)

    cagr_3y = rng.normal(p["cagr_mu"], p["cagr_sd"], size=n)
    cagr_5y = cagr_3y + rng.normal(-0.5, 1.2, size=n)
    annual_vol = np.clip(rng.normal(p["vol_mu"], p["vol_sd"], size=n), 5, None)
    max_drawdown = np.clip(annual_vol * rng.uniform(1.4, 2.2, size=n), 5, 80)
    var_95 = annual_vol * rng.uniform(0.6, 0.9, size=n)

    rf = 6.5  # risk-free rate proxy, %
    sharpe = (cagr_5y - rf) / annual_vol
    downside_vol = annual_vol * rng.uniform(0.55, 0.85, size=n)
    sortino = (cagr_5y - rf) / downside_vol

    expense_ratio = np.clip(rng.normal(p["exp_mu"], p["exp_sd"], size=n), 0.1, 2.75)
    exit_load = rng.choice([0.0, 0.5, 1.0], size=n, p=[0.2, 0.55, 0.25])

    rolling_alpha_std = np.clip(rng.normal(4.5, 1.8, size=n) + annual_vol * 0.08, 0.5, None)

    fund_ids = [f"{category.replace(' ', '')[:2].upper()}{i:04d}" for i in range(n)]

    df = pd.DataFrame({
        "fund_id": fund_ids,
        "category": category,
        "aum_cr": aum.round(1),
        "history_years": history_years.round(2),
        "cagr_3y": cagr_3y.round(2),
        "cagr_5y": cagr_5y.round(2),
        "sharpe": sharpe.round(3),
        "sortino": sortino.round(3),
        "annual_vol": annual_vol.round(2),
        "max_drawdown": max_drawdown.round(2),
        "var_95": var_95.round(2),
        "expense_ratio": expense_ratio.round(2),
        "exit_load": exit_load,
        "rolling_alpha_std": rolling_alpha_std.round(3),
    })
    return df


def apply_screen(df):
    """Paper's filtering protocol: AUM >= 500cr and >= 5y history."""
    return df[(df["aum_cr"] >= 500) & (df["history_years"] >= 5.0)].reset_index(drop=True)


def build_objectives(df):
    """Construct the normalized 4-D objective vector per category (Eq. 33)."""
    out = []
    for cat, g in df.groupby("category"):
        g = g.copy()
        return_raw = 0.6 * _minmax(g["cagr_5y"]) + 0.4 * _minmax(g["sharpe"] + g["sortino"])
        risk_raw = 1.0 - (0.45 * _minmax(g["annual_vol"]) + 0.35 * _minmax(g["max_drawdown"]) + 0.20 * _minmax(g["var_95"]))
        cost_raw = 1.0 - (0.75 * _minmax(g["expense_ratio"]) + 0.25 * _minmax(g["exit_load"]))
        consistency_raw = 1.0 - _minmax(g["rolling_alpha_std"])

        g["f_return"] = return_raw
        g["f_risk"] = risk_raw
        g["f_cost"] = cost_raw
        g["f_consistency"] = consistency_raw
        out.append(g)
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/funds.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-per-category", type=int, default=60)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    frames = [generate_category(c, args.n_per_category, rng) for c in CATEGORIES]
    df = pd.concat(frames, ignore_index=True)
    df = apply_screen(df)
    df = build_objectives(df)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} funds across {df['category'].nunique()} categories to {args.out}")


if __name__ == "__main__":
    main()
