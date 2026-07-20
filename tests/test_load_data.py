"""Unit tests for data loading and cleaning pipeline."""
import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.load_data import clean_data


def _make_raw_df():
    """Create a minimal raw DataFrame that mirrors the telco CSV structure."""
    return pd.DataFrame({
        "customerID": ["001", "002", "003", "004"],
        "gender": ["Male", "Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0, 0],
        "Partner": ["Yes", "No", "No", "Yes"],
        "Dependents": ["No", "No", "Yes", "No"],
        "tenure": [12, 24, 3, 60],
        "PhoneService": ["Yes", "Yes", "No", "Yes"],
        "MultipleLines": ["No", "Yes", "No phone service", "No"],
        "InternetService": ["Fiber optic", "DSL", "No", "DSL"],
        "OnlineSecurity": ["No", "Yes", "No internet service", "Yes"],
        "OnlineBackup": ["No", "No", "No internet service", "Yes"],
        "DeviceProtection": ["No", "Yes", "No internet service", "No"],
        "TechSupport": ["No", "No", "No internet service", "Yes"],
        "StreamingTV": ["No", "No", "No internet service", "No"],
        "StreamingMovies": ["No", "No", "No internet service", "No"],
        "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "No", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check",
                          "Bank transfer (automatic)", "Credit card (automatic)"],
        "MonthlyCharges": [75.0, 55.0, 20.0, 50.0],
        "TotalCharges": ["900.0", "1320.0", "60.0", "3000.0"],
        "Churn": ["Yes", "No", "Yes", "No"],
    })


class TestCleanData:
    def test_customer_id_dropped(self):
        df = clean_data(_make_raw_df())
        assert "customerID" not in df.columns

    def test_churn_is_binary_int(self):
        df = clean_data(_make_raw_df())
        assert df["Churn"].dtype in (int, np.int64, np.int32)
        assert set(df["Churn"].unique()).issubset({0, 1})

    def test_binary_cols_are_numeric(self):
        df = clean_data(_make_raw_df())
        for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
            if col in df.columns:
                assert df[col].dtype in (int, np.int64, np.int32), f"{col} should be int"

    def test_total_charges_is_float(self):
        df = clean_data(_make_raw_df())
        assert pd.api.types.is_float_dtype(df["TotalCharges"])

    def test_total_charges_nan_filled(self):
        raw = _make_raw_df()
        raw.loc[0, "TotalCharges"] = " "  # invalid
        df = clean_data(raw)
        assert not df["TotalCharges"].isna().any()

    def test_no_nulls_in_output(self):
        df = clean_data(_make_raw_df())
        assert df.isna().sum().sum() == 0

    def test_row_count_preserved_when_clean(self):
        raw = _make_raw_df()
        df = clean_data(raw)
        assert len(df) == len(raw)

    def test_invalid_churn_rows_dropped(self):
        raw = _make_raw_df()
        raw.loc[0, "Churn"] = "Unknown"  # invalid — should be dropped
        df = clean_data(raw)
        assert len(df) == len(raw) - 1

    def test_internet_service_nulls_filled(self):
        raw = _make_raw_df()
        raw.loc[0, "OnlineSecurity"] = None
        df = clean_data(raw)
        assert df["OnlineSecurity"].iloc[0] == "No internet service"

    def test_senior_citizen_numeric(self):
        df = clean_data(_make_raw_df())
        assert df["SeniorCitizen"].dtype in (int, np.int64, np.int32)

    def test_does_not_mutate_input(self):
        raw = _make_raw_df()
        original_churn = raw["Churn"].copy()
        clean_data(raw)
        # Original should still have string "Yes"/"No"
        assert raw["Churn"].iloc[0] == original_churn.iloc[0]
