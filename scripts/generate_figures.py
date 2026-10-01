from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#B279A2",
]
alt.data_transformers.disable_max_rows()


def table(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


def export(chart: alt.TopLevelMixin, name: str) -> None:
    chart.save(OUT / f"{name}.html", embed_options={"renderer": "svg"})
    chart.save(OUT / f"{name}.png", scale_factor=2)


def primary_holdout() -> alt.Chart:
    frame = table("primary_outcomes.csv")
    frame["estimator"] = pd.Categorical(
        frame["estimator"],
        ["Sample", "Eigenvalue clipping", "Ledoit-Wolf", "RIE"],
        ordered=True,
    )
    points = (
        alt.Chart(frame)
        .mark_point(filled=True, size=190)
        .encode(
            x=alt.X("estimator:N", title=None, sort=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "holdout_volatility_pct_per_day:Q",
                title="Realized volatility (% per trading day)",
                scale=alt.Scale(domain=[0.69, 0.715], zero=False),
            ),
            color=alt.Color(
                "estimator:N",
                scale=alt.Scale(domain=["Sample", "Eigenvalue clipping", "Ledoit-Wolf", "RIE"], range=COLORS),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("estimator:N", title="Estimator"),
                alt.Tooltip("holdout_volatility_pct_per_day:Q", title="Holdout vol. (%/day)", format=".4f"),
                alt.Tooltip("annualized_sharpe:Q", title="Annualized Sharpe", format=".3f"),
                alt.Tooltip(
                    "holdout_annualized_mean_return_pct:Q",
                    title="Annualized arithmetic mean return (%)",
                    format=".2f",
                ),
            ],
        )
    )
    sample = float(
        frame.loc[frame["estimator"] == "Sample", "holdout_volatility_pct_per_day"].iloc[0]
    )
    baseline = alt.Chart(pd.DataFrame({"baseline": [sample]})).mark_rule(
        color="#303030", strokeDash=[5, 4]
    ).encode(y=alt.Y("baseline:Q", scale=alt.Scale(domain=[0.69, 0.715])))
    labels = points.mark_text(dy=-12, color="#222").encode(
        text=alt.Text("holdout_volatility_pct_per_day:Q", format=".4f")
    )
    return (
        (points + baseline + labels)
        .properties(
            title="All estimators are close on the common 741-day holdout",
            width=650,
            height=340,
            description=(
                "Daily realized volatility from simple asset returns at daily target weights. "
                "All four covariance estimators have similar holdout risk."
            ),
        )
        .configure_axis(grid=True, gridColor="#E8E8E8", labelFontSize=12, titleFontSize=13)
        .configure_title(anchor="start", fontSize=17, subtitleFontSize=12)
        .configure_view(stroke=None)
    )


def allocation_breadth() -> alt.VConcatChart:
    frame = table("primary_outcomes.csv").melt(
        id_vars=["estimator"],
        value_vars=["effective_number_assets", "shannon_effective_breadth"],
        var_name="measure",
        value_name="effective_breadth",
    )
    labels = {
        "effective_number_assets": "Inverse-Herfindahl effective holdings",
        "shannon_effective_breadth": "Shannon effective breadth",
    }
    charts = []
    for measure in ("effective_number_assets", "shannon_effective_breadth"):
        subset = frame.loc[frame["measure"] == measure].copy()
        subset["estimator"] = pd.Categorical(
            subset["estimator"],
            ["Sample", "Eigenvalue clipping", "Ledoit-Wolf", "RIE"],
            ordered=True,
        )
        chart = (
            alt.Chart(subset)
            .mark_line(point=alt.OverlayMarkDef(size=110), strokeWidth=2)
            .encode(
                x=alt.X("estimator:N", title=None, sort=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("effective_breadth:Q", title=labels[measure], scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "estimator:N",
                    scale=alt.Scale(domain=["Sample", "Eigenvalue clipping", "Ledoit-Wolf", "RIE"], range=COLORS),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("estimator:N", title="Estimator"),
                    alt.Tooltip("effective_breadth:Q", title=labels[measure], format=".2f"),
                ],
            )
            .properties(width=650, height=175)
        )
        charts.append(chart)
    return (
        alt.vconcat(*charts, spacing=16)
        .properties(
            title="Portfolio breadth depends on its definition",
            description=(
                "RIE has 10.85 inverse-Herfindahl effective holdings versus 10.58 "
                "for sample covariance; Shannon effective breadth is 25.91 versus 25.37."
            ),
        )
        .configure_axis(grid=True, gridColor="#E8E8E8", labelFontSize=11, titleFontSize=12)
        .configure_title(anchor="start", fontSize=17)
        .configure_view(stroke=None)
    )


def market_regimes() -> alt.FacetChart:
    frame = table("market_regimes.csv").melt(
        id_vars=["proxy", "state", "matched_days"],
        var_name="estimator",
        value_name="volatility",
    )
    frame["proxy"] = frame["proxy"].map({"VOMXCUS": "VOMXCUS*", "VOMXGUS": "VOMXGUS*"})
    return (
        alt.Chart(frame)
        .mark_line(point=alt.OverlayMarkDef(size=95), strokeWidth=2)
        .encode(
            x=alt.X(
                "state:N",
                title=None,
                sort=["Below threshold", "Above threshold"],
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "volatility:Q",
                title="Portfolio volatility (% per day)",
                scale=alt.Scale(domain=[0.56, 0.82], zero=False),
            ),
            color=alt.Color(
                "estimator:N",
                scale=alt.Scale(
                    domain=["Sample", "Eigenvalue clipping", "Ledoit-Wolf", "RIE"],
                    range=COLORS,
                ),
                title="Estimator",
            ),
            detail="estimator:N",
            column=alt.Column("proxy:N", title=None, header=alt.Header(labelFontSize=12)),
            tooltip=[
                alt.Tooltip("proxy:N", title="Proxy"),
                alt.Tooltip("estimator:N", title="Estimator"),
                alt.Tooltip("state:N", title="State"),
                alt.Tooltip("matched_days:Q", title="Matched days"),
                alt.Tooltip("volatility:Q", title="Volatility (%/day)", format=".4f"),
            ],
        )
        .properties(
            title="High-threshold days coincide with higher realized risk",
            width=360,
            height=270,
            description=(
                "Descriptive portfolio volatility above and below a fixed training-period "
                "90th-percentile threshold for two VOM-labelled series. The economic meaning "
                "of these fields has not been verified."
            ),
        )
        .configure_axis(grid=True, gridColor="#E8E8E8", labelFontSize=11, titleFontSize=12)
        .configure_title(anchor="start", fontSize=17)
        .configure_legend(orient="bottom", labelFontSize=11)
        .configure_view(stroke=None)
    )


def conditioning() -> alt.FacetChart:
    frame = table("highdim_conditioning.csv")
    frame = frame.loc[frame["field"] == "Gross total-return index"].copy()
    frame["estimator"] = pd.Categorical(
        frame["estimator"], ["Sample", "Eigenvalue clipping"], ordered=True
    )
    return (
        alt.Chart(frame)
        .mark_line(point=alt.OverlayMarkDef(size=150), strokeWidth=2)
        .encode(
            x=alt.X("estimator:N", title=None, sort=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("condition_number:Q", title="Condition number", scale=alt.Scale(type="log")),
            color=alt.Color(
                "estimator:N",
                scale=alt.Scale(domain=["Sample", "Eigenvalue clipping"], range=COLORS[:2]),
                legend=None,
            ),
            detail="matrix:N",
            column=alt.Column("matrix:N", title=None),
            tooltip=[
                alt.Tooltip("matrix:N", title="Matrix"),
                alt.Tooltip("estimator:N", title="Estimator"),
                alt.Tooltip("condition_number:Q", title="Condition number", format=".2f"),
                alt.Tooltip("q:Q", title="N/T", format=".2f"),
            ],
        )
        .properties(
            title="Eigenvalue clipping improves conditioning at q = 0.25",
            width=265,
            height=290,
            description=(
                "For the 25-stock total-return panel, clipping reduces correlation "
                "condition number from 24.34 to 5.33 and covariance condition number "
                "from 279.55 to 128.49."
            ),
        )
        .configure_axis(grid=True, gridColor="#E8E8E8", labelFontSize=11, titleFontSize=12)
        .configure_title(anchor="start", fontSize=17)
        .configure_view(stroke=None)
    )


def static_benchmarks() -> alt.Chart:
    frame = table("static_constraint_baselines.csv")
    frame = frame.loc[frame["constraint"] == "Long-only"].copy()
    frame["portfolio"] = frame["estimator"].replace(
        {"Equal-weight 1/N": "Equal-weight 1/N"}
    )
    frame["portfolio"] = pd.Categorical(
        frame["portfolio"],
        ["Sample", "Eigenvalue clipping", "Ledoit-Wolf", "RIE", "Equal-weight 1/N"],
        ordered=True,
    )
    return (
        alt.Chart(frame)
        .mark_circle(size=180, opacity=0.9)
        .encode(
            x=alt.X(
                "holdout_annualized_volatility_pct:Q",
                title="Annualized realized volatility (%)",
                scale=alt.Scale(domain=[9, 14], zero=False),
            ),
            y=alt.Y(
                "holdout_annualized_mean_return_pct:Q",
                title="Annualized arithmetic mean return (%)",
                scale=alt.Scale(domain=[0, 16], zero=False),
            ),
            color=alt.Color(
                "portfolio:N",
                scale=alt.Scale(
                    domain=[
                        "Sample",
                        "Eigenvalue clipping",
                        "Ledoit-Wolf",
                        "RIE",
                        "Equal-weight 1/N",
                    ],
                    range=COLORS + ["#E45756"],
                ),
                title="Portfolio",
            ),
            tooltip=[
                alt.Tooltip("portfolio:N", title="Portfolio"),
                alt.Tooltip(
                    "holdout_annualized_mean_return_pct:Q",
                    title="Annualized mean return (%)",
                    format=".2f",
                ),
                alt.Tooltip(
                    "holdout_annualized_volatility_pct:Q",
                    title="Annualized volatility (%)",
                    format=".2f",
                ),
                alt.Tooltip(
                    "holdout_annualized_sharpe_rf_8pct:Q",
                    title="Sharpe (8% risk-free rate)",
                    format=".3f",
                ),
                alt.Tooltip(
                    "mean_one_way_turnover_per_rebalance:Q",
                    title="Mean one-way turnover per day",
                    format=".3%",
                ),
            ],
        )
        .properties(
            title="The long-only 1/N benchmark has the highest holdout Sharpe",
            width=650,
            height=340,
            description=(
                "Five long-only portfolios on the same 741-day holdout. "
                "The equally weighted benchmark has higher volatility but a larger "
                "annualized arithmetic return and Sharpe than the GMV portfolios."
            ),
        )
        .configure_axis(grid=True, gridColor="#E8E8E8", labelFontSize=12, titleFontSize=13)
        .configure_title(anchor="start", fontSize=17)
        .configure_legend(orient="bottom", labelFontSize=10)
        .configure_view(stroke=None)
    )


def rolling_walkforward() -> alt.FacetChart:
    frame = table("rolling_walkforward_outcomes.csv")
    frame["portfolio"] = frame["estimator"]
    frame["portfolio"] = pd.Categorical(
        frame["portfolio"],
        ["Sample", "Eigenvalue clipping", "Ledoit-Wolf", "RIE", "Equal-weight 1/N"],
        ordered=True,
    )
    return (
        alt.Chart(frame)
        .mark_circle(size=145, opacity=0.9)
        .encode(
            x=alt.X(
                "holdout_annualized_volatility_pct:Q",
                title="Annualized realized volatility (%)",
                scale=alt.Scale(domain=[8, 14], zero=False),
            ),
            y=alt.Y(
                "holdout_annualized_mean_return_pct:Q",
                title="Annualized arithmetic mean return (%)",
                scale=alt.Scale(domain=[-5, 20], zero=False),
            ),
            color=alt.Color(
                "portfolio:N",
                scale=alt.Scale(
                    domain=[
                        "Sample",
                        "Eigenvalue clipping",
                        "Ledoit-Wolf",
                        "RIE",
                        "Equal-weight 1/N",
                    ],
                    range=COLORS + ["#E45756"],
                ),
                title="Estimator",
            ),
            column=alt.Column(
                "constraint:N",
                title=None,
                sort=["Unconstrained", "Long-only"],
                header=alt.Header(labelFontSize=12),
            ),
            tooltip=[
                alt.Tooltip("portfolio:N", title="Estimator"),
                alt.Tooltip("constraint:N", title="Constraint"),
                alt.Tooltip(
                    "holdout_annualized_mean_return_pct:Q",
                    title="Annualized mean return (%)",
                    format=".2f",
                ),
                alt.Tooltip(
                    "holdout_annualized_volatility_pct:Q",
                    title="Annualized volatility (%)",
                    format=".2f",
                ),
                alt.Tooltip(
                    "holdout_annualized_sharpe_rf_8pct:Q",
                    title="Sharpe (8% risk-free rate)",
                    format=".3f",
                ),
                alt.Tooltip(
                    "mean_one_way_turnover_per_rebalance:Q",
                    title="Mean one-way turnover per rebalance",
                    format=".3%",
                ),
            ],
        )
        .properties(
            title="Rolling results are exploratory and depend on constraints",
            width=330,
            height=300,
            description=(
                "Twenty-five monthly rebalances using a 504-trading-day lookback "
                "and 21-day buy-and-hold periods. No transaction costs are included."
            ),
        )
        .configure_axis(grid=True, gridColor="#E8E8E8", labelFontSize=11, titleFontSize=12)
        .configure_title(anchor="start", fontSize=17)
        .configure_legend(orient="bottom", labelFontSize=10)
        .configure_view(stroke=None)
    )


def transaction_cost_sensitivity() -> alt.Chart:
    frame = table("transaction_cost_sensitivity.csv")
    frame = frame.loc[
        (frame["design"] == "Rolling; 21-day buy-and-hold")
        & (frame["constraint"] == "Long-only")
    ].copy()
    frame["portfolio"] = frame["estimator"]
    frame["portfolio"] = pd.Categorical(
        frame["portfolio"],
        ["Sample", "Eigenvalue clipping", "Ledoit-Wolf", "RIE", "Equal-weight 1/N"],
        ordered=True,
    )
    return (
        alt.Chart(frame)
        .mark_line(point=alt.OverlayMarkDef(size=80), strokeWidth=2)
        .encode(
            x=alt.X(
                "one_way_cost_basis_points:Q",
                title="Assumed cost (basis points per one-way turnover)",
                axis=alt.Axis(tickMinStep=5),
            ),
            y=alt.Y(
                "net_annualized_mean_return_pct:Q",
                title="Net annualized arithmetic mean return (%)",
                scale=alt.Scale(domain=[-2, 20], zero=False),
            ),
            color=alt.Color(
                "portfolio:N",
                scale=alt.Scale(
                    domain=[
                        "Sample",
                        "Eigenvalue clipping",
                        "Ledoit-Wolf",
                        "RIE",
                        "Equal-weight 1/N",
                    ],
                    range=COLORS + ["#E45756"],
                ),
                title="Portfolio",
            ),
            detail="portfolio:N",
            tooltip=[
                alt.Tooltip("portfolio:N", title="Portfolio"),
                alt.Tooltip("one_way_cost_basis_points:Q", title="Assumed cost (bps)"),
                alt.Tooltip(
                    "net_annualized_mean_return_pct:Q",
                    title="Net annualized return (%)",
                    format=".2f",
                ),
                alt.Tooltip(
                    "mean_one_way_turnover_per_rebalance:Q",
                    title="Mean one-way turnover per rebalance",
                    format=".1%",
                ),
            ],
        )
        .properties(
            title="The equal-weight benchmark remains ahead under assumed trading costs",
            width=700,
            height=340,
            description=(
                "Net annualized arithmetic returns over monthly rolling portfolios, "
                "after subtracting assumed costs of 0 to 50 basis points per one-way "
                "turnover. These are scenarios, not measured Mexican trading costs."
            ),
        )
        .configure_axis(grid=True, gridColor="#E8E8E8", labelFontSize=12, titleFontSize=13)
        .configure_title(anchor="start", fontSize=17)
        .configure_legend(orient="bottom", labelFontSize=10)
        .configure_view(stroke=None)
    )


def main() -> None:
    charts = {
        "figure1_primary_holdout": primary_holdout(),
        "figure2_allocation_breadth": allocation_breadth(),
        "figure3_market_regimes": market_regimes(),
        "figure4_conditioning": conditioning(),
        "figure5_static_benchmarks": static_benchmarks(),
        "figure6_rolling_walkforward": rolling_walkforward(),
        "figure7_transaction_cost_sensitivity": transaction_cost_sensitivity(),
    }
    for name, chart in charts.items():
        export(chart, name)
        print(f"Wrote {name}.html and {name}.png")


if __name__ == "__main__":
    main()
