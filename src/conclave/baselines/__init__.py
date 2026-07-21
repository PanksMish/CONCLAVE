from . import qc, sa_llm, heuristic_mediator, rank_aggregation, mcdm, debate_mediator

METHOD_RUNNERS = {
    "QC": qc.run,
    "SA": sa_llm.run,
    "HM": heuristic_mediator.run,
    "WB": rank_aggregation.run_borda,
    "WK": rank_aggregation.run_kemeny,
    "TOPSIS": mcdm.run_topsis,
    "VIKOR": mcdm.run_vikor,
    "DM": debate_mediator.run,
}
