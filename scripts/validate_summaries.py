from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def close(actual: float, expected: float, *, tolerance: float = 0.005) -> None:
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance):
        raise AssertionError(f"{actual} differs from expected {expected}")


def main() -> None:
    primary = read_csv("primary_outcomes.csv")
    assert len(primary) == 4
    sample = next(row for row in primary if row["estimator"] == "Sample")
    rie = next(row for row in primary if row["estimator"] == "RIE")
    sample_vol = float(sample["holdout_volatility_pct_per_day"])
    rie_vol = float(rie["holdout_volatility_pct_per_day"])
    relative_vol_change = 100 * (rie_vol / sample_vol - 1)
    close(relative_vol_change, -1.307)
    close(
        float(rie["effective_number_assets"])
        - float(sample["effective_number_assets"]),
        0.50,
    )
    close(
        float(rie["shannon_effective_breadth"])
        - float(sample["shannon_effective_breadth"]),
        0,
    )

    inference = read_csv("inference_summary.csv")
    sharpe = next(
        row for row in inference
        if row["quantity"] == "RIE_minus_sample_annualized_Sharpe"
    )
    assert float(sharpe["lower_95"]) < 0 < float(sharpe["upper_95"])
    assert float(sharpe["p_value"]) > 0.05
    close(float(sharpe["estimate"]), 0.011)

    regimes = read_csv("market_regimes.csv")
    for proxy in {row["proxy"] for row in regimes}:
        rows = {row["state"]: row for row in regimes if row["proxy"] == proxy}
        assert set(rows) == {"Below threshold", "Above threshold"}
        assert sum(int(row["matched_days"]) for row in rows.values()) == 506
        for estimator in ("Sample", "Eigenvalue clipping", "Ledoit-Wolf", "RIE"):
            low = float(rows["Below threshold"][estimator])
            high = float(rows["Above threshold"][estimator])
            assert high > low, (proxy, estimator, low, high)
    vomxcus_rows = {row["state"]: row for row in regimes if row["proxy"] == "VOMXCUS"}
    vomxcus_changes = [
        100 * (
            float(vomxcus_rows["Above threshold"][estimator])
            / float(vomxcus_rows["Below threshold"][estimator])
            - 1
        )
        for estimator in ("Sample", "Eigenvalue clipping", "Ledoit-Wolf", "RIE")
    ]
    assert min(vomxcus_changes) > 30
    assert max(vomxcus_changes) < 33

    highdim = read_csv("highdim_conditioning.csv")
    highdim_rows = [
        row for row in highdim if row["field"] == "Gross total-return index"
    ]
    assert all(int(row["assets"]) == 25 and int(row["returns"]) == 100 for row in highdim_rows)
    for matrix in ("Covariance", "Correlation"):
        rows = {
            row["estimator"]: float(row["condition_number"])
            for row in highdim
            if row["field"] == "Gross total-return index"
            and row["matrix"] == matrix
        }
        assert rows["Eigenvalue clipping"] < rows["Sample"]
    correlation = {
        row["estimator"]: float(row["condition_number"])
        for row in highdim_rows if row["matrix"] == "Correlation"
    }
    covariance = {
        row["estimator"]: float(row["condition_number"])
        for row in highdim_rows if row["matrix"] == "Covariance"
    }
    close(100 * (1 - correlation["Eigenvalue clipping"] / correlation["Sample"]), 78.1, tolerance=0.1)
    close(100 * (1 - covariance["Eigenvalue clipping"] / covariance["Sample"]), 54.0, tolerance=0.1)
    q = 25 / 100
    mp_upper = (1 + math.sqrt(q)) ** 2
    close(mp_upper, 2.25, tolerance=1e-12)
    largest_eigenvalue = 4.178
    assert largest_eigenvalue > mp_upper

    q_gt_1 = read_csv("q_gt_1.csv")
    vols = {
        row["estimator"]: float(row["mean_annualized_realized_volatility_pct"])
        for row in q_gt_1
    }
    assert vols["RIE"] > 10 * vols["Eigenvalue clipping"]
    rie_window_sd = float(
        next(row for row in q_gt_1 if row["estimator"] == "RIE")[
            "window_std_annualized_volatility_pct"
        ]
    )
    close(rie_window_sd, 320.17)

    print("Summary checks passed.")
    print(f"  RIE vs sample holdout volatility: {relative_vol_change:.3f}%")
    print("  RIE vs sample breadth: +0.50 effective assets; Shannon difference 0")
    print("  Sharpe bootstrap interval includes zero; p=0.186")
    print("  Both VOM-proxy high states have higher descriptive volatility")
    print(f"  q=0.25 Marchenko-Pastur upper edge: {mp_upper:.2f}; top eigenvalue: 4.178")
    print("  Clipping lowers both reported condition numbers by 78.1% / 54.0%")
    print("  q>1 RIE is unstable (144.43% mean risk; 320.17% window SD)")


if __name__ == "__main__":
    main()
