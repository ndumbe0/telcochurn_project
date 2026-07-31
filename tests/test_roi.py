"""Unit tests for the retention ROI calculator."""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app.roi import compute_roi


class TestComputeRoi:
    BASE = dict(
        baseline_prob=0.7,
        new_prob=0.3,
        monthly_charges=65.0,
        tenure=6,
        contract="Month-to-month",
        cost_per_customer=50.0,
    )

    def test_revenue_saved_positive(self):
        result = compute_roi(**self.BASE)
        assert result["Revenue Saved ($)"] > 0

    def test_net_roi_calculated(self):
        result = compute_roi(**self.BASE)
        assert result["Net ROI ($)"] == pytest.approx(
            result["Revenue Saved ($)"] - self.BASE["cost_per_customer"], abs=0.01
        )

    def test_roi_pct_calculated(self):
        result = compute_roi(**self.BASE)
        expected = (result["Net ROI ($)"] / self.BASE["cost_per_customer"]) * 100
        assert result["ROI (%)"] == pytest.approx(expected, abs=0.1)

    def test_zero_cost(self):
        """Zero cost should return 0% ROI (no division error)."""
        result = compute_roi(
            baseline_prob=0.6, new_prob=0.2,
            monthly_charges=65.0, tenure=6,
            contract="Month-to-month", cost_per_customer=0.0,
        )
        assert result["ROI (%)"] == pytest.approx(0.0)

    def test_no_improvement(self):
        """When new_prob >= baseline_prob, revenue saved should be 0."""
        result = compute_roi(
            baseline_prob=0.3, new_prob=0.5,
            monthly_charges=65.0, tenure=6,
            contract="Month-to-month", cost_per_customer=20.0,
        )
        assert result["Revenue Saved ($)"] == pytest.approx(0.0, abs=0.01)
        assert result["Net ROI ($)"] < 0

    def test_breakeven_infinite_when_no_reduction(self):
        """If prob_reduction is 0, break-even should be N/A."""
        result = compute_roi(
            baseline_prob=0.4, new_prob=0.4,
            monthly_charges=65.0, tenure=6,
            contract="Month-to-month", cost_per_customer=20.0,
        )
        assert result["Break-even (months)"] == "N/A"

    def test_breakeven_reasonable(self):
        """Break-even should be a finite number when there's a real improvement."""
        result = compute_roi(**self.BASE)
        assert result["Break-even (months)"] != "N/A"
        be = float(result["Break-even (months)"])
        assert be > 0

    def test_two_year_contract_higher_revenue(self):
        """Two-year contract should produce higher revenue saved than month-to-month."""
        r_mtm = compute_roi(baseline_prob=0.6, new_prob=0.2, monthly_charges=65.0,
                             tenure=0, contract="Month-to-month", cost_per_customer=0.0)
        r_2yr = compute_roi(baseline_prob=0.6, new_prob=0.2, monthly_charges=65.0,
                             tenure=0, contract="Two year", cost_per_customer=0.0)
        assert r_2yr["Revenue Saved ($)"] > r_mtm["Revenue Saved ($)"]

    def test_all_keys_present(self):
        result = compute_roi(**self.BASE)
        for key in ["Probability Reduction", "Revenue Saved ($)", "Intervention Cost ($)",
                    "Net ROI ($)", "ROI (%)", "Break-even (months)", "Monthly Revenue at Risk ($)"]:
            assert key in result
