# Covariance estimation under portfolio constraints in Mexican equities

Research materials for the working paper by Julio César Galindo López and
Laura Jiménez Casillas, Universidad Panamericana, Ciudad UP.

**Research question.** How sensitive are minimum-variance portfolio outcomes
to covariance estimation when Mexican equities are evaluated under long-only
constraints and periodic rebalancing?

## Current evidence

The validated panel contains 37 equities mapped from a prior 38-security list
to Bloomberg workbook tabs with `TOT_RETURN_INDEX_GROSS_DVDS`. It covers
2011-07-18 to 2025-09-29, with 3,706 daily log returns and no gaps greater
than seven calendar days. FUNO11 was absent from the workbook and was not
replaced. The fixed chronological split uses 2,965 training and 741 holdout
observations; the test interval is 2022-11-28 to 2025-09-29.

- In the unconstrained fixed holdout, RIE has daily volatility of 0.6952%,
  compared with 0.6998% for sample covariance. Its annualized Sharpe is
  -0.232 versus -0.207; the RIE-minus-sample difference is -0.0248 (20-day
  moving-block 95% interval [-0.0451, -0.0038]; HAC p=0.015).
- For long-only portfolios, fixed-holdout Sharpe estimates are -0.035 for
  RIE and 0.450 for equal-weight 1/N. The difference is imprecise in this
  single holdout (95% interval [-1.068, 0.053]; HAC p=0.099).
- In 25 rolling 21-observation buy-and-hold blocks with a 504-observation
  estimation window, long-only Sharpe is -0.646 for RIE and 0.767 for 1/N.
  The RIE-minus-1/N difference is -1.413 (21-day moving-block 95% interval
  [-2.333, -0.641]; HAC p=0.0013). This rolling analysis uses the same
  historical trajectory as the fixed holdout and is not an independent
  replication.
- Mean one-way turnover per actual rolling rebalance is 5.05% for long-only
  RIE and 2.81% for 1/N. At an assumed 50 basis points per unit of turnover,
  annualized mean returns are 0.57% and 18.03%, respectively. These are
  hypothetical sensitivities, not observed trading costs.

The findings do not show that RIE is generally inferior or that 1/N will
outperform in future samples. They show that a modest volatility reduction
does not imply a Sharpe improvement and that constraints, rebalance protocol,
and the naive benchmark materially affect the comparison.

## Reproduce the public artifacts

Requirements: Python 3.10 or newer, packages in `requirements.txt`, and
pdflatex with the LaTeX packages used by the manuscript.

```powershell
python -m pip install -r requirements.txt
python scripts/validate_summaries.py
python scripts/generate_figures.py
powershell -ExecutionPolicy Bypass -File scripts/compile_paper.ps1
```

The figures are generated from aggregate CSVs with Altair and exported as
interactive Vega-Lite HTML and high-resolution PNG. The PDF uses the PNG
figures. The public aggregate checks validate stored outputs; they do not
re-estimate portfolios.

To rerun estimation, lawful access to a compatible Bloomberg workbook is
required:

```powershell
python scripts/run_licensed_panel_validation.py --input "C:\path\to\licensed-workbook.xlsx" --output "C:\path\to\aggregate-output"
```

The estimation script writes aggregate CSV summaries only. Do not save a
licensed workbook, raw prices or returns, or security-level portfolio weights
inside this public repository.

## Data, scope, and limitations

Bloomberg data are licensed and are not redistributed. The repository
contains aggregate results, a ticker-to-workbook-tab map, code, and figures;
it contains no daily licensed prices, returns, or security weights.
Reproduction requires lawful access to a compatible workbook with the
documented tabs and total-return field.

The 37-security universe is retrospective, not reconstructed from
point-in-time BMV/BIVA membership. Balanced coverage does not remove
survivorship, ticker-mapping, or ex-post selection bias. The analysis uses
one fixed holdout and a rolling sensitivity on the same historical path;
inference conditions on the chosen universe and fitted weights. Transaction
costs are scenarios, not observed Mexican execution costs, and exclude
liquidity constraints, market impact, taxes, short-sale frictions, and initial
deployment. The risk-free rate is a fixed 8% assumption, not a local
yield-series estimate. Thus, the paper does not establish a fully investable
strategy, causal institutional effects, or results generalizable to other
emerging economies.

For a stronger submission to an emerging-markets finance journal, the
highest-value extensions are a point-in-time universe including delisted
securities, observed local trading-cost and liquidity measures, and a
comparable multi-market design. This draft should not be described as
submission-ready until those design limits and the journal fit are addressed.

## Repository layout

```text
data/       Aggregate outcomes, inference, and universe mapping
figures/    Altair interactive HTML and static PNG figures
paper/      LaTeX manuscript and compiled PDF
scripts/    Estimation, validation, figure generation, and PDF build
```

## Citation and reuse

Please cite the working paper and references listed in it when using the
research. The manuscript is not peer reviewed. No license granting reuse of
the manuscript or aggregate research artifacts is provided; obtain the
authors' permission before redistribution. Bloomberg data remain subject to
their provider's terms.
