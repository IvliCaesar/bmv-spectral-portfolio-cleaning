from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def close(actual: float, expected: float, *, tolerance: float = 1e-6) -> None:
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance):
        raise AssertionError(f"{actual} differs from expected {expected}")


def main() -> None:
    audit = read_csv("panel_audit.csv")
    assert len(audit) == 1
    assert int(audit[0]["asset_count"]) == 37
    assert int(audit[0]["daily_gap_count_over_7_calendar_days"]) == 0
    assert audit[0]["return_source"] == "Bloomberg TOT_RETURN_INDEX_GROSS_DVDS"

    primary = read_csv("primary_outcomes.csv")
    assert len(primary) == 4
    by_estimator = {row["estimator"]: row for row in primary}
    sample, rie = by_estimator["Sample"], by_estimator["RIE"]
    close(float(sample["holdout_volatility_pct_per_day"]), 0.6997940861)
    close(float(rie["holdout_volatility_pct_per_day"]), 0.6951748809)
    close(
        float(rie["annualized_sharpe"]) - float(sample["annualized_sharpe"]),
        -0.0248319881,
    )
    close(
        float(rie["effective_number_assets"])
        - float(sample["effective_number_assets"]),
        0.2732228,
    )
    close(
        float(rie["shannon_effective_breadth"])
        - float(sample["shannon_effective_breadth"]),
        0.5418326,
    )

    static = read_csv("static_constraint_baselines.csv")
    assert len(static) == 9
    static_rows = {
        (row["estimator"], row["constraint"]): row for row in static
    }
    equal_weight = static_rows[("Equal-weight 1/N", "Long-only")]
    assert float(equal_weight["holdout_annualized_sharpe_rf_8pct"]) == max(
        float(row["holdout_annualized_sharpe_rf_8pct"])
        for row in static if row["constraint"] == "Long-only"
    )
    assert all(
        float(row["mean_one_way_turnover_per_rebalance"]) > 0
        for row in static
    )

    rolling = read_csv("rolling_walkforward_outcomes.csv")
    assert len(rolling) == 9
    rolling_rows = {
        (row["estimator"], row["constraint"]): row for row in rolling
    }
    rolling_equal = rolling_rows[("Equal-weight 1/N", "Long-only")]
    assert float(rolling_equal["holdout_annualized_sharpe_rf_8pct"]) > float(
        rolling_rows[("RIE", "Long-only")]["holdout_annualized_sharpe_rf_8pct"]
    )
    assert all(
        float(row["mean_one_way_turnover_per_rebalance"]) >= 0
        for row in rolling
    )

    inference = read_csv("inference_summary.csv")
    assert len(inference) == 12
    static_sharpe = next(
        row for row in inference
        if row["quantity"]
        == "Fixed-holdout_Unconstrained_RIE_minus_Sample_annualized_Sharpe"
    )
    assert float(static_sharpe["upper_95"]) < 0
    assert float(static_sharpe["p_value"]) < 0.05
    static_volatility = next(
        row for row in inference
        if row["quantity"]
        == "Fixed-holdout_Unconstrained_RIE_minus_Sample_daily_volatility_percentage_points"
    )
    assert float(static_volatility["upper_95"]) < 0
    assert float(static_volatility["p_value"]) < 0.01
    benchmark_sharpe = next(
        row for row in inference
        if row["quantity"]
        == "Fixed-holdout_Long-only_RIE_minus_Equal-weight_1N_annualized_Sharpe"
    )
    assert float(benchmark_sharpe["estimate"]) < 0
    assert float(benchmark_sharpe["upper_95"]) > 0

    sensitivity = read_csv("inference_reanalysis.csv")
    assert len(sensitivity) == 36
    rolling_long_sharpe = [
        row for row in sensitivity
        if row["constraint"] == "Long-only"
        and row["outcome"] == "RIE_minus_Sample_annualized_Sharpe"
        and row["design"].startswith("Rolling")
    ]
    assert len(rolling_long_sharpe) == 3
    assert any(float(row["bootstrap_lower_95"]) < 0 for row in rolling_long_sharpe)
    assert any(float(row["bootstrap_upper_95"]) > 0 for row in rolling_long_sharpe)

    costs = read_csv("transaction_cost_sensitivity.csv")
    assert len(costs) == 90
    zero_cost = next(
        row for row in costs
        if row["design"] == "Rolling; 21-day buy-and-hold"
        and row["estimator"] == "RIE"
        and row["constraint"] == "Long-only"
        and int(row["one_way_cost_basis_points"]) == 0
    )
    close(
        float(zero_cost["net_annualized_mean_return_pct"]),
        float(zero_cost["gross_annualized_mean_return_pct"]),
    )
    close(
        float(zero_cost["mean_one_way_turnover_per_rebalance"]),
        float(rolling_rows[("RIE", "Long-only")]["mean_one_way_turnover_per_rebalance"]),
    )
    equal_weight_cost = next(
        row for row in costs
        if row["design"] == "Rolling; 21-day buy-and-hold"
        and row["estimator"] == "Equal-weight 1/N"
        and row["constraint"] == "Long-only"
        and int(row["one_way_cost_basis_points"]) == 0
    )
    close(
        float(equal_weight_cost["mean_one_way_turnover_per_rebalance"]),
        float(
            rolling_rows[("Equal-weight 1/N", "Long-only")][
                "mean_one_way_turnover_per_rebalance"
            ]
        ),
    )
    assert all(
        float(row["net_annualized_mean_return_pct"])
        <= float(row["gross_annualized_mean_return_pct"])
        for row in costs
    )

    regimes = read_csv("market_regimes.csv")
    assert len(regimes) == 4
    matched_counts = {
        proxy: sum(int(row["matched_days"]) for row in regimes if row["proxy"] == proxy)
        for proxy in {row["proxy"] for row in regimes}
    }
    assert matched_counts == {"VOMXCUS": 690, "VOMXGUS": 690}
    for proxy in {row["proxy"] for row in regimes}:
        rows = {row["state"]: row for row in regimes if row["proxy"] == proxy}
        assert set(rows) == {"Below threshold", "Above threshold"}
        assert sum(int(row["matched_days"]) for row in rows.values()) == 690
        for estimator in ("Sample", "Eigenvalue clipping", "Ledoit-Wolf", "RIE"):
            assert float(rows["Above threshold"][estimator]) > float(
                rows["Below threshold"][estimator]
            )

    highdim = read_csv("highdim_conditioning.csv")
    highdim_rows = [
        row for row in highdim if row["field"] == "Gross total-return index"
    ]
    assert all(
        int(row["assets"]) == 25 and int(row["returns"]) == 100
        for row in highdim_rows
    )
    for matrix in ("Covariance", "Correlation"):
        rows = {
            row["estimator"]: float(row["condition_number"])
            for row in highdim_rows
            if row["matrix"] == matrix
        }
        assert rows["Eigenvalue clipping"] < rows["Sample"]
    q = 25 / 100
    close((1 + math.sqrt(q)) ** 2, 2.25)

    print("Aggregate summary checks passed.")
    print("  Bloomberg gross total-return panel is continuous: 37 assets, no >7-day gaps.")
    print(
        "  Simple-return static holdout: RIE vol "
        f"{float(rie['holdout_volatility_pct_per_day']):.4f}% vs sample "
        f"{float(sample['holdout_volatility_pct_per_day']):.4f}% per day."
    )
    print(
        "  Fixed-holdout RIE-minus-sample Sharpe interval excludes zero; "
        "volatility interval excludes zero."
    )
    print("  Equal-weight baseline, long-only portfolios, and walk-forward outputs checked.")
    print("  Market-proxy aggregates and q=0.25 conditioning checks passed.")


if __name__ == "__main__":
    main()
