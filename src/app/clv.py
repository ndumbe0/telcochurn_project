"""Customer Lifetime Value (CLV) and Revenue-at-Risk calculations."""
import pandas as pd

# Expected total tenure (months) by contract type
EXPECTED_TOTAL_MONTHS = {
    "Month-to-month": 24,
    "One year": 36,
    "Two year": 60,
}


def compute_clv_single(
    monthly_charges: float,
    tenure: float,
    churn_prob: float,
    contract: str = "Month-to-month",
) -> dict:
    """
    Estimate CLV and revenue at risk for a single customer.

    CLV            = MonthlyCharges × remaining_months × (1 − churn_prob)
    Revenue at Risk = MonthlyCharges × churn_prob × remaining_months
    """
    expected = EXPECTED_TOTAL_MONTHS.get(contract, 24)
    remaining = max(0.0, expected - tenure)
    clv = monthly_charges * remaining * (1 - churn_prob)
    rar = monthly_charges * churn_prob * remaining
    return {
        "CLV Estimate ($)": round(clv, 2),
        "Revenue at Risk ($)": round(rar, 2),
        "Est. Remaining Months": round(remaining, 1),
    }


def add_clv_to_batch(results: pd.DataFrame) -> pd.DataFrame:
    """Append CLV_Estimate and Revenue_at_Risk columns to batch prediction results."""
    df = results.copy()
    monthly = pd.to_numeric(
        df.get("MonthlyCharges", pd.Series([65.0] * len(df))), errors="coerce"
    ).fillna(65.0)
    tenure = pd.to_numeric(
        df.get("tenure", pd.Series([12.0] * len(df))), errors="coerce"
    ).fillna(12.0)
    contract_col = df.get("Contract", pd.Series(["Month-to-month"] * len(df)))
    churn_prob = (
        pd.to_numeric(df["Churn_Probability"], errors="coerce").fillna(50) / 100
    )

    expected = contract_col.map(EXPECTED_TOTAL_MONTHS).fillna(24)
    remaining = (expected - tenure).clip(lower=0)

    df["CLV_Estimate ($)"] = (monthly * remaining * (1 - churn_prob)).round(2)
    df["Revenue_at_Risk ($)"] = (monthly * churn_prob * remaining).round(2)
    return df
