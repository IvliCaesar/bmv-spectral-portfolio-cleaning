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
                scale=alt.Scale(domain=[0.675, 0.692], zero=False),
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
            ],
        )
    )
    sample = float(
        frame.loc[frame["estimator"] == "Sample", "holdout_volatility_pct_per_day"].iloc[0]
    )
    baseline = alt.Chart(pd.DataFrame({"baseline": [sample]})).mark_rule(
        color="#303030", strokeDash=[5, 4]
    ).encode(y=alt.Y("baseline:Q", scale=alt.Scale(domain=[0.675, 0.692])))
    labels = points.mark_text(dy=-12, color="#222").encode(
        text=alt.Text("holdout_volatility_pct_per_day:Q", format=".4f")
    )
    return (
        (points + baseline + labels)
        .properties(
            title="All estimators are close on the common 529-day holdout",
            width=650,
            height=340,
            description=(
                "Daily realized volatility for four minimum-variance estimators. "
                "RIE is 0.6794 percent versus 0.6884 percent for the sample estimator."
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
                "RIE has 9.86 effective holdings versus 9.36 for sample covariance, "
                "but both have Shannon effective breadth 26.92."
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


def q_gt_1() -> alt.Chart:
    frame = table("q_gt_1.csv")
    frame["estimator"] = pd.Categorical(
        frame["estimator"],
        ["Sample", "Eigenvalue clipping", "Ledoit-Wolf", "RIE"],
        ordered=True,
    )
    points = (
        alt.Chart(frame)
        .mark_point(filled=True, size=180)
        .encode(
            x=alt.X("estimator:N", title=None, sort=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "mean_annualized_realized_volatility_pct:Q",
                title="Mean annualized realized volatility (%)",
                scale=alt.Scale(type="log", domain=[8, 200]),
            ),
            color=alt.Color(
                "estimator:N",
                scale=alt.Scale(domain=["Sample", "Eigenvalue clipping", "Ledoit-Wolf", "RIE"], range=COLORS),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("estimator:N", title="Estimator"),
                alt.Tooltip("mean_annualized_realized_volatility_pct:Q", title="Mean annualized vol. (%)", format=".2f"),
                alt.Tooltip("window_std_annualized_volatility_pct:Q", title="RIE window SD (%)", format=".2f"),
            ],
        )
    )
    labels = points.mark_text(dy=-12, color="#222").encode(
        text=alt.Text("mean_annualized_realized_volatility_pct:Q", format=".2f")
    )
    return (
        (points + labels)
        .properties(
            title="At q = 1.9, this unmodified RIE implementation is unstable",
            width=650,
            height=300,
            description=(
                "Mean annualized realized volatility over ten short-window trials. "
                "The RIE window-to-window standard deviation is 320.17 percent."
            ),
        )
        .configure_axis(grid=True, gridColor="#E8E8E8", labelFontSize=12, titleFontSize=13)
        .configure_title(anchor="start", fontSize=17)
        .configure_view(stroke=None)
    )


def distributional_robustness() -> alt.Chart:
    frame = table("distributional_robustness.csv").melt(
        id_vars="estimator",
        var_name="training_transform",
        value_name="volatility",
    )
    frame["training_transform"] = frame["training_transform"].map(
        {
            "raw_training_holdout_volatility_pct_per_day": "Raw",
            "rank_gaussianized_training_holdout_volatility_pct_per_day": "Rank-Gaussianized",
        }
    )
    frame["estimator"] = pd.Categorical(
        frame["estimator"],
        ["Sample", "Eigenvalue clipping", "Ledoit-Wolf", "RIE"],
        ordered=True,
    )
    return (
        alt.Chart(frame)
        .mark_line(point=alt.OverlayMarkDef(size=90), strokeWidth=2)
        .encode(
            x=alt.X(
                "training_transform:N",
                title=None,
                sort=["Raw", "Rank-Gaussianized"],
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "volatility:Q",
                title="Holdout volatility (% per day)",
                scale=alt.Scale(domain=[0.675, 0.70], zero=False),
            ),
            color=alt.Color(
                "estimator:N",
                scale=alt.Scale(domain=list(frame["estimator"].cat.categories), range=COLORS),
                title="Estimator",
            ),
            detail="estimator:N",
            tooltip=[
                alt.Tooltip("estimator:N", title="Estimator"),
                alt.Tooltip("training_transform:N", title="Training transformation"),
                alt.Tooltip("volatility:Q", title="Holdout vol. (%/day)", format=".4f"),
            ],
        )
        .properties(
            title="Rank-Gaussianizing training data weakens, but preserves, RIE's edge over sample",
            width=650,
            height=280,
            description=(
                "Holdout volatility for four estimators after fitting on raw or "
                "rank-Gaussianized training marginals. The holdout remains untransformed."
            ),
        )
        .configure_axis(grid=True, gridColor="#E8E8E8", labelFontSize=12, titleFontSize=13)
        .configure_title(anchor="start", fontSize=17)
        .configure_legend(orient="bottom", labelFontSize=11)
        .configure_view(stroke=None)
    )


def main() -> None:
    charts = {
        "figure1_primary_holdout": primary_holdout(),
        "figure2_allocation_breadth": allocation_breadth(),
        "figure3_market_regimes": market_regimes(),
        "figure4_conditioning": conditioning(),
        "figure5_q_gt_1": q_gt_1(),
        "figure6_distributional_robustness": distributional_robustness(),
    }
    for name, chart in charts.items():
        export(chart, name)
        print(f"Wrote {name}.html and {name}.png")


if __name__ == "__main__":
    main()
