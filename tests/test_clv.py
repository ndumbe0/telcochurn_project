"""Unit tests for CLV and revenue-at-risk calculations."""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app.clv import compute_clv_single, add_clv_to_batch, EXPECTED_TOTAL_MONTHS
import pandas as pd


class TestComputeClvSingle:
    def test_standard_case(self):
        result = compute_clv_single(
            monthly_charges=65.0, tenure=12, churn_prob=0.4, contract="Month-to-month"
        )
        assert result["CLV Estimate ($)"] >= 0
        assert result["Revenue at Risk ($)"] >= 0
        assert result["Est. Remaining Months"] == pytest.approx(12.0)

    def test_zero_churn_prob(self):
        """When churn probability is 0, CLV = full remaining value, revenue at risk = 0."""
        result = compute_clv_single(
            monthly_charges=50.0, tenure=0, churn_prob=0.0, contract="Month-to-month"
        )
        expected_remaining = EXPECTED_TOTAL_MONTHS["Month-to-month"]
        assert result["CLV Estimate ($)"] == pytest.approx(50.0 * expected_remaining, abs=0.01)
        assert result["Revenue at Risk ($)"] == pytest.approx(0.0, abs=0.01)

    def test_full_churn_prob(self):
        """When churn probability is 1, CLV = 0, all revenue is at risk."""
        result = compute_clv_single(
            monthly_charges=80.0, tenure=0, churn_prob=1.0, contract="Two year"
        )
        assert result["CLV Estimate ($)"] == pytest.approx(0.0, abs=0.01)
        expected_remaining = EXPECTED_TOTAL_MONTHS["Two year"]
        assert result["Revenue at Risk ($)"] == pytest.approx(80.0 * expected_remaining, abs=0.01)

    def test_tenure_exceeds_expected(self):
        """If tenure exceeds contract length, remaining months should not be negative."""
        result = compute_clv_single(
            monthly_charges=70.0, tenure=100, churn_prob=0.5, contract="Month-to-month"
        )
        assert result["Est. Remaining Months"] == pytest.approx(0.0)
        assert result["CLV Estimate ($)"] == pytest.approx(0.0, abs=0.01)
        assert result["Revenue at Risk ($)"] == pytest.approx(0.0, abs=0.01)

    def test_zero_monthly_charges(self):
        """Zero monthly charges should produce zero CLV and zero revenue at risk."""
        result = compute_clv_single(
            monthly_charges=0.0, tenure=6, churn_prob=0.5, contract="One year"
        )
        assert result["CLV Estimate ($)"] == pytest.approx(0.0, abs=0.01)
        assert result["Revenue at Risk ($)"] == pytest.approx(0.0, abs=0.01)

    def test_unknown_contract_type(self):
        """Unknown contract type should fall back to 24-month default."""
        result = compute_clv_single(
            monthly_charges=60.0, tenure=0, churn_prob=0.0, contract="Unknown contract"
        )
        assert result["Est. Remaining Months"] == pytest.approx(24.0)

    def test_all_contract_types(self):
        """All known contract types should produce non-negative values."""
        for contract in ["Month-to-month", "One year", "Two year"]:
            result = compute_clv_single(50.0, 6, 0.3, contract)
            assert result["CLV Estimate ($)"] >= 0
            assert result["Revenue at Risk ($)"] >= 0


class TestAddClvToBatch:
    def _make_df(self):
        return pd.DataFrame({
            "customerID": ["001", "002", "003"],
            "MonthlyCharges": [65.0, 80.0, 45.0],
            "tenure": [12, 24, 6],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "Churn_Probability": [70.0, 20.0, 50.0],
            "Prediction": ["Churn", "Stay", "Churn"],
        })

    def test_columns_added(self):
        df = add_clv_to_batch(self._make_df())
        assert "CLV_Estimate ($)" in df.columns
        assert "Revenue_at_Risk ($)" in df.columns

    def test_no_negative_values(self):
        df = add_clv_to_batch(self._make_df())
        assert (df["CLV_Estimate ($)"] >= 0).all()
        assert (df["Revenue_at_Risk ($)"] >= 0).all()

    def test_original_df_unchanged(self):
        original = self._make_df()
        _ = add_clv_to_batch(original)
        assert "CLV_Estimate ($)" not in original.columns

    def test_missing_monthly_charges_defaults(self):
        df = pd.DataFrame({
            "Churn_Probability": [50.0],
            "Prediction": ["Churn"],
        })
        result = add_clv_to_batch(df)
        assert "CLV_Estimate ($)" in result.columns
        assert result["CLV_Estimate ($)"].iloc[0] >= 0
