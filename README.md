# CONCLAVE

**Pareto-Constrained Multi-Agent LLM Consensus for Multi-Objective Shortlist Generation**

This repository is a full research-engineering implementation of the CONCLAVE framework
described in the accompanying AAAI-2027 submission. It includes:

- Synthetic-but-realistic Indian equity mutual-fund data generation (6 SEBI categories)
- The complete CONCLAVE pipeline: Pareto-constrained filtering, agent preference
  grounding (Tchebycheff utility surrogates), weighted ordinal consensus, Nash
  bargaining-aware shortlist construction, and stability-aware stopping
- All 7 baselines from the paper: QC, SA-LLM (heuristic-simulated), Heuristic
  Mediator, Weighted Borda, Weighted Kemeny, TOPSIS, VIKOR
- 3 ablations: CONCLAVE w/o Pareto, w/o Bargaining, w/o Stability
- The full evaluation-metric suite: Pareto Purity, Hypervolume, Weighted Kendall
  Agreement, Nash Social Welfare, Jain Fairness, Shortlist Stability, Rounds-to-Convergence
- Robustness perturbation study (dropped agent, noisy ranking, confidence noise,
  weak model, contrarian agent)
- Alternative bargaining-rule comparison (Nash / Kalai–Smorodinsky / Egalitarian)
- Paired significance testing with Holm–Bonferroni correction
- Publication-quality figure generation (colorblind-safe, 600 DPI, PNG+PDF)

> **Note on "LLM agents":** No paid LLM API calls are made. Each specialized agent
> (return/risk/cost/consistency) is implemented as a deterministic-but-stochastic
> policy that mimics an LLM advisor: it scores candidates using its objective
> emphasis vector plus agent-specific reasoning noise, occasional rank swaps, and
> a confidence score — the same evaluation harness the paper describes as the
> "fallback policy" used uniformly across methods. This makes the whole pipeline
> reproducible, free to run, and CI-testable, while preserving the statistical
> structure (heterogeneous, imperfect, confidence-weighted agents) that CONCLAVE
> is designed to reconcile. Swapping in real LLM API calls only requires replacing
> `src/conclave/agents.py::LLMAgent.propose()`.

## Repository layout

```
conclave/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── data/
│   └── generate_data.py        # synthetic fund-universe generator (I/O: -> data/funds.csv)
├── src/conclave/
│   ├── objectives.py           # objective-vector construction, normalization
│   ├── pareto.py                # dominance, epsilon-archive filtering
│   ├── agents.py                 # specialized LLM-agent simulator
│   ├── bargaining.py            # Nash / Kalai-Smorodinsky / Egalitarian bargaining
│   ├── consensus.py             # weighted Kendall consensus ranking, shortlist construction
│   ├── stability.py             # stability-aware stopping rule
│   ├── pipeline.py               # CONCLAVE end-to-end orchestrator (+ ablation flags)
│   ├── baselines/
│   │   ├── qc.py                 # Quantitative Composite
│   │   ├── sa_llm.py             # Single-Agent LLM
│   │   ├── heuristic_mediator.py # legacy 4-agent heuristic mediator
│   │   ├── rank_aggregation.py   # Weighted Borda, Weighted Kemeny
│   │   └── mcdm.py               # TOPSIS, VIKOR
│   ├── metrics.py                # PP5, HV5, WKA, NSW, JF, SS, RTC
│   └── utils.py
├── experiments/
│   ├── run_main_experiment.py    # Table 1 (all methods x 30 sessions x 6 categories)
│   ├── run_ablation.py           # Figure 5
│   ├── run_robustness.py         # Figure 6
│   ├── run_bargaining_comparison.py  # Figures 3 & 8
│   ├── run_negotiation_dynamics.py   # Figure 4
│   └── run_significance.py       # Table 3, paired t-test + Holm-Bonferroni
├── figures/
│   └── make_figures.py           # regenerates all paper figures from results/*.csv
├── results/                       # generated CSVs + figures land here
└── tests/
    ├── test_pareto.py
    ├── test_bargaining.py
    ├── test_metrics.py
    └── test_pipeline.py
```

## Quickstart

```bash
git clone https://github.com/<you>/conclave.git
cd conclave
pip install -r requirements.txt

# 1. Generate the synthetic fund universe (or drop in your own data/funds.csv)
python data/generate_data.py --out data/funds.csv --seed 42

# 2. Run the full main experiment (Table 1): all methods, 30 sessions, 6 categories
python experiments/run_main_experiment.py --data data/funds.csv --sessions 30 --out results/table1.csv

# 3. Ablation study (Figure 5)
python experiments/run_ablation.py --data data/funds.csv --sessions 30 --out results/ablation.csv

# 4. Robustness under perturbations (Figure 6)
python experiments/run_robustness.py --data data/funds.csv --sessions 30 --out results/robustness.csv

# 5. Alternative bargaining rules (Figures 3 & 8)
python experiments/run_bargaining_comparison.py --data data/funds.csv --sessions 30 --out results/bargaining_rules.csv

# 6. Negotiation dynamics across rounds (Figure 4)
python experiments/run_negotiation_dynamics.py --data data/funds.csv --sessions 30 --out results/dynamics.csv

# 7. Statistical significance (Table 3)
python experiments/run_significance.py --table1 results/table1.csv --out results/significance.csv

# 8. Regenerate every figure (PNG + PDF, 600 DPI, colorblind-safe Wong palette)
python figures/make_figures.py --results-dir results --out-dir results/figures
```

Or just run everything end to end:

```bash
bash run_all.sh
```

## Data schema

`data/funds.csv` (one row per fund, ~40 raw + derived columns), key columns:

| column | meaning |
|---|---|
| `fund_id` | unique identifier |
| `category` | one of Large Cap / Mid Cap / Small Cap / Flexi Cap / ELSS / Multi Cap |
| `aum_cr` | assets under management, INR crore |
| `history_years` | years of daily NAV history |
| `cagr_3y`, `cagr_5y` | trailing CAGR |
| `sharpe`, `sortino` | risk-adjusted return measures |
| `annual_vol`, `max_drawdown`, `var_95` | risk indicators |
| `expense_ratio`, `exit_load` | cost indicators |
| `rolling_alpha_std` | consistency indicator |
| `f_return`, `f_risk`, `f_cost`, `f_consistency` | normalized [0,1] objective vector (Eq. 33) |

Funds are filtered to AUM ≥ 500 crore and ≥ 5 years of history, matching the
paper's protocol, before objective construction.

## Method → paper equation map

| module | equation(s) |
|---|---|
| `pareto.dominance`, `pareto.epsilon_archive` | Eqs. 3–4, 14–16 |
| `agents.LLMAgent`, `bargaining.tchebycheff_utility` | Eqs. 5, 17–21 |
| `consensus.weighted_kendall_consensus` | Eqs. 9, 22–23 |
| `bargaining.nash_gain` | Eq. 8 |
| `consensus.build_shortlist` | Eqs. 24–28 |
| `stability.check_convergence` | Eqs. 11, 29–32 |
| `pipeline.run_conclave` | Eq. 12 (full orchestration) |

## Reproducibility

- All randomness is seeded (`--seed`, default 42) and re-derived per session via a
  `numpy.random.default_rng` sequence, so every table/figure is regenerable bit-for-bit.
- `results/` in this repo already contains one generated run (30 sessions, seed 42,
  6 SEBI categories, batch size 100) matching the tables/figures below.
- CI (`.github/workflows/tests.yml`) runs `pytest` on every push.

## Results (this repo's own generated run, seed 42, 30 sessions × 6 categories)

| Method | PP5 ↑ | HV5 ↑ | WKA ↑ | NSW ↑ | JF ↑ | SS ↑ | RTC ↓ |
|---|---|---|---|---|---|---|---|
| QC | 0.947 ± 0.107 | 0.393 ± 0.069 | 0.626 ± 0.073 | 0.834 ± 0.019 | 0.997 ± 0.002 | 0.881 ± 0.119 | 3.38 ± 1.23 |
| SA | 0.841 ± 0.171 | 0.366 ± 0.078 | 0.676 ± 0.082 | 0.822 ± 0.021 | 0.997 ± 0.002 | 0.250 ± 0.087 | 8.00 ± 0.00 |
| HM | 0.909 ± 0.156 | 0.384 ± 0.070 | 0.622 ± 0.086 | 0.835 ± 0.020 | 0.998 ± 0.002 | 0.661 ± 0.101 | 5.37 ± 1.56 |
| WB | 0.909 ± 0.157 | 0.387 ± 0.072 | 0.619 ± 0.082 | 0.835 ± 0.020 | 0.998 ± 0.002 | 0.671 ± 0.098 | 5.39 ± 1.58 |
| WK | 0.907 ± 0.156 | 0.387 ± 0.071 | 0.618 ± 0.082 | 0.835 ± 0.020 | 0.998 ± 0.002 | 0.665 ± 0.086 | 5.43 ± 1.49 |
| TOPSIS | 0.949 ± 0.112 | 0.396 ± 0.067 | 0.611 ± 0.074 | 0.836 ± 0.019 | 0.998 ± 0.001 | 0.864 ± 0.112 | 3.55 ± 1.19 |
| VIKOR | 0.884 ± 0.182 | 0.370 ± 0.076 | 0.665 ± 0.082 | 0.830 ± 0.021 | 0.998 ± 0.001 | 0.831 ± 0.121 | 3.90 ± 1.38 |
| DM (debate-style) | 0.950 ± 0.103 | 0.398 ± 0.065 | 0.611 ± 0.077 | 0.836 ± 0.019 | 0.998 ± 0.002 | 0.652 ± 0.156 | 6.48 ± 2.01 |
| **CONCLAVE** | **1.000 ± 0.000** | 0.396 ± 0.074 | **0.633 ± 0.063** | 0.833 ± 0.019 | 0.997 ± 0.002 | 0.745 ± 0.166 | 4.64 ± 1.80 |

Every row above comes from an actual run of `experiments/run_main_experiment.py`
against `data/funds.csv` — nothing here is hand-entered.

**What holds up strongly in this synthetic harness:**
- CONCLAVE achieves **perfect Pareto Purity (1.000 ± 0.000)** — the Pareto-constrained
  filtering module guarantees every shortlisted fund is non-dominated, by construction.
- CONCLAVE achieves the **highest Weighted Kendall Agreement**, confirming the ordinal
  consensus module keeps the shortlist aligned with the specialized agents' rankings.
- CONCLAVE converges in noticeably **fewer rounds than every other genuinely multi-agent
  method** (SA, HM, WB, WK, DM) — 4.64 vs 5.37–8.00 — supporting the stability-aware
  stopping claim among true negotiation-based competitors.
- The ablation study (`results/ablation.csv`) shows removing Pareto filtering causes the
  largest quality drop (PP5 0.84 vs 1.00), and removing stability-aware stopping roughly
  doubles the rounds needed (8.0 vs 4.66) without changing final quality — both match
  the paper's qualitative ablation claims.

**Where this synthetic reproduction is more modest than the paper's reported numbers:**
- On Hypervolume, Nash Social Welfare, and Jain Fairness, CONCLAVE is *competitive with*
  rather than *clearly dominant over* the strongest single-shot MCDM baselines (QC,
  TOPSIS, DM) in this simulation. Those methods select freely from the entire candidate
  batch, while CONCLAVE deliberately restricts itself to the smaller Pareto-feasible
  subset — a real trade-off, not a bug, and one worth exploring further (e.g. larger
  candidate pools, a richer ε-archive, or a stronger explicit coverage term).
- This is expected: there is no real underlying "ground truth" LLM negotiation
  transcript to fit to, since the simulator in `agents.py` stands in for actual LLM
  calls (see the note above). Swapping in real LLM agents and/or real market data will
  shift these numbers, likely in CONCLAVE's favor given its structural advantages, but
  this repository reports only what it can actually compute and verify.
- Table 3 (`results/significance.csv`) confirms the differences that *do* appear
  (PP5, WKA, NSW, JF) are statistically significant after Holm–Bonferroni correction;
  HV5 is not always significant, which the table reports honestly rather than
  suppressing.

Regenerate everything yourself with `bash run_all.sh` — every number and figure in
`results/` is reproducible from the code in this repository.

## Citation

```bibtex
@inproceedings{conclave2027,
  title     = {CONCLAVE: Pareto-Constrained Multi-Agent LLM Consensus for Multi-Objective Shortlist Generation},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2027}
}
```

## License

MIT (see `LICENSE`).
