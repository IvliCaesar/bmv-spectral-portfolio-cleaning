# Figures

Each chart is generated from the aggregate CSVs in `../data/` by
`../scripts/generate_figures.py`.

- `figure1_primary_holdout`: four-estimator holdout-volatility dot plot;
  axis truncated to resolve the small differences.
- `figure2_allocation_breadth`: inverse-Herfindahl and Shannon allocation
  summaries on separate scales.
- `figure3_market_regimes`: VOM-proxy high/low descriptive comparisons;
  field meaning unverified and no confidence intervals available.
- `figure4_conditioning`: sample/clipped condition numbers on logarithmic
  scales for covariance and correlation matrices.
- `figure5_q_gt_1`: annualized risk in ten short-window $q=1.9$ trials, on a
  logarithmic scale; the RIE window standard deviation is 320.17%.
- `figure6_distributional_robustness`: holdout risk under raw and
  rank-Gaussianized training marginals; holdout data remain untransformed.

PNG files are used by the typeset manuscript. HTML files retain interactive
Altair tooltips and the chart data.
