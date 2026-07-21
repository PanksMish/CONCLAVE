#!/usr/bin/env bash
# Run the entire CONCLAVE pipeline end-to-end: data generation, all
# experiments, significance testing, and figure regeneration.
set -euo pipefail

SEED=${SEED:-42}
SESSIONS=${SESSIONS:-30}

echo "== 1/8: generating synthetic fund universe =="
python data/generate_data.py --out data/funds.csv --seed "$SEED"

echo "== 2/8: main experiment (Table 1) =="
python experiments/run_main_experiment.py --data data/funds.csv --sessions "$SESSIONS" --out results/table1.csv

echo "== 3/8: ablation study (Figure 5) =="
python experiments/run_ablation.py --data data/funds.csv --sessions "$SESSIONS" --out results/ablation.csv

echo "== 4/8: robustness under perturbations (Figure 6) =="
python experiments/run_robustness.py --data data/funds.csv --sessions "$SESSIONS" --out results/robustness.csv

echo "== 5/8: bargaining-rule comparison (Figures 3 & 8) =="
python experiments/run_bargaining_comparison.py --data data/funds.csv --sessions "$SESSIONS" --out results/bargaining_rules.csv

echo "== 6/8: negotiation dynamics across rounds (Figure 4) =="
python experiments/run_negotiation_dynamics.py --data data/funds.csv --sessions "$SESSIONS" --out results/dynamics.csv

echo "== 7/8: statistical significance (Table 3) =="
python experiments/run_significance.py --table1 results/table1.csv --out results/significance.csv

echo "== 8/8: regenerating all figures =="
python figures/make_figures.py --results-dir results --out-dir results/figures

echo "Done. See results/ for CSVs and results/figures/ for PNG+PDF figures."
