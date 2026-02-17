"""
Out-of-Distribution evaluation stub for BiasUnlearn debiased models.

OOD evaluation uses real earnings call data and consensus beat/miss scenarios.
Implementation pending: requires earnings call transcript collection via Manticore SQL
and consensus/actual EPS data from FactSet.

See new_npo/PLAN.md Section 6 for full OOD evaluation design.
"""

# TODO: Implement when OOD data (earnings calls, consensus data) is collected.
#
# Planned components:
#   - collect_earnings_data(): Manticore SQL query for earnings call transcripts
#   - build_ood_prompts(): construct investment decision prompts from real earnings data
#   - run_ood_evaluation(): measure bias on real financial data
#   - compute_ood_metrics(): sector gap, size gap, decision accuracy, calibration
