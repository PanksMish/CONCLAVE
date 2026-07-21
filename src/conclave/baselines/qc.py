"""Quantitative Composite (QC): non-LLM baseline, single fixed-weight composite score."""
import numpy as np
from ._harness import run_baseline

QC_WEIGHTS = np.array([0.30, 0.30, 0.20, 0.20])  # return, risk, cost, consistency


def _qc_shortlist_fn(C_ids, C_F, rankings, confidences, psi_matrix, d, ks, rng):
    composite = C_F @ QC_WEIGHTS
    order = np.argsort(-composite)
    return [C_ids[i] for i in order[:ks]]


def run(ids, F, **kwargs):
    return run_baseline(ids, F, _qc_shortlist_fn, use_pareto_filter=False, **kwargs)
