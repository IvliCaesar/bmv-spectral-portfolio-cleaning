# Spectral covariance cleaning and minimum-variance portfolios in Mexican equities

Research materials for the working paper by Julio César Galindo López and
Laura Jiménez Casillas, affiliated with Universidad Panamericana, Ciudad UP.

**Research question.** Does nonlinear random-matrix-theory (RMT) covariance
cleaning lower out-of-sample minimum-variance portfolio risk without reducing
allocation breadth, and how do the observed results vary with market regimes
and the asset-to-observation ratio?

## Main findings

- On a fixed 38-stock, 2,644-return panel, the 529-day chronological holdout
  volatility is 0.6794% per day for RIE and 0.6884% for the sample covariance:
  a 1.31% relative reduction. This is a descriptive difference, not a
  statistically established volatility improvement.
- The paired 20-day moving-block bootstrap estimates an annualized Sharpe
  difference (RIE minus sample) of +0.011 (95% interval [-0.021, 0.057];
  one-sided p=0.186). It does not establish a Sharpe improvement.
- RIE's inverse-Herfindahl effective holdings count is 9.86 versus 9.36 for
  the sample estimator, while Shannon effective breadth is 26.92 for both.
  Conclusions about "diversity" therefore depend on the chosen allocation
  metric and do not describe investor diversity.
- In the two provisionally labelled Bloomberg VOM series, high-state daily
  volatility is higher than low-state volatility. The increase is around
  30--32% for VOMXCUS and 3--6% for VOMXGUS across the four estimators.
  Bloomberg field metadata are missing, so these are descriptive associations
  with provisional proxies, not causal market-regime effects.
- On a matched 25-stock, 100-return panel ($q=0.25$), clipping lowers the
  correlation-matrix condition number from 24.34 to 5.33. This is a
  conditioning result, not evidence of better future portfolio returns.
- In the deliberately short-window $q=1.9$ exercise, unmodified RIE is
  unstable: mean annualized realized volatility is 144.43%, versus 11.78% for
  clipping, 12.12% for Ledoit--Wolf, and 16.57% for the sample estimator.
  This is an implementation boundary, not evidence that all nonlinear
  shrinkage methods fail at $q>1$.

The paper and the results files provide the complete set of reported
estimands, caveats, and sample definitions. All charts in `figures/` are
generated from the aggregate tables in `data/` with Altair. Their tooltips
preserve the exact tabulated values.

## Reproduce the public artifacts

Requirements: Python 3.10 or newer, Altair, `vl-convert-python`, and
pdflatex with the LaTeX packages used by the manuscript.

```powershell
python -m pip install -r requirements.txt
python scripts/validate_summaries.py
python scripts/generate_figures.py
powershell -ExecutionPolicy Bypass -File scripts/compile_paper.ps1
```

`generate_figures.py` writes interactive Vega-Lite HTML and high-resolution PNG
figures. The LaTeX source uses the PNGs for portable PDF compilation.

## Data and reproducibility

The original Economática and Bloomberg price exports are licensed and are not
included. The CSVs in `data/` are **aggregate summary statistics only**,
transcribed from the reported analyses; they contain no daily prices, returns,
constituent-level time series, weights, or Bloomberg exports. They are enough
to recreate the displayed summary charts and to verify arithmetic and
Marchenko--Pastur edge calculations. They are **not enough to rerun the
portfolio estimation, bootstrap, factor regressions, regime classification,
or eigendecompositions**. Reproducing those analyses requires lawful access to
the licensed inputs and the complete computational workflow. This public
repository contains summary checks and figure-generation code, not the
original licensed-data estimation scripts; it is artifact-reproducible, not a
full computational replication package.

The recent 25-stock panel used `TOT_RETURN_INDEX_GROSS_DVDS` from the
September 2026 Bloomberg workbook. The 25-ticker set was fixed by a prior
diagnostic, not sampled randomly from the 275 instrument tabs. Its results
must not be generalized to the whole universe. VOM series labels and their
economic meaning remain unverified.

## Repository layout

```text
data/       Aggregate reported results and provenance notes
figures/    Altair interactive HTML and static PNG charts
paper/      LaTeX manuscript and compiled PDF
scripts/    Chart generation, summary validation, and paper build
```

## Citation and reuse

Please cite the manuscript and the references listed there when using the
research. The manuscript is a working draft, not a peer-reviewed article.
No license granting reuse of the manuscript or data summaries is provided in
this repository; obtain the authors' permission before redistributing them.
Third-party data remain subject to their providers' terms.

## Limitations

The primary estimator comparison is one static-weight split, not a repeated
walk-forward deployment study. The portfolios are unconstrained and exclude
transaction costs, turnover, liquidity, short-sale frictions, and market
impact. The primary universe's construction may also create selection or
survivorship bias. The U.S. Fama--French factor regression is an exploratory
cross-market diagnostic, not a Mexican-equity pricing test. See the manuscript
for the complete discussion.
