# Aggregate data dictionary

These CSVs contain only aggregate summary statistics reported in the
manuscript. They do not contain licensed daily data or security-level
observations.

- `primary_outcomes.csv`: fixed 2,115-day training / 529-day test split;
  volatilities are percentages per trading day and Sharpe is annualized.
- `market_regimes.csv`: daily portfolio volatility in percent per day,
  conditioned on a fixed training-period 90th-percentile threshold for each
  VOM-labelled proxy; only 506 matched holdout days are available per proxy.
- `highdim_conditioning.csv`: condition numbers for the 25-stock,
  100-return panel ($q=0.25$), using gross total returns and, as a
  sensitivity, unadjusted `PX_LAST`.
- `q_gt_1.csv`: mean annualized realized volatility in percent across ten
  short training-window experiments ($N=38$, $T=20$, test windows of 60 days).
  Only the RIE window-to-window standard deviation was preserved in the
  reported output.
- `distributional_robustness.csv`: holdout volatility after raw versus
  rank-Gaussianized training marginals.
- `inference_summary.csv`: inferential quantities and their reported design.

The 25-stock high-dimensional results and VOM-conditioned summaries are
derived from licensed Bloomberg exports. Before using or redistributing even
these aggregates, users must independently verify that their data-provider
terms permit it. Summary-only charts do not make the underlying analyses
reproducible.
