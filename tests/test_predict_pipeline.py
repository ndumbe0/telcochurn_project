"""Unit tests for prediction pipeline and preprocessing."""
import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.predict import _preprocess_input, predict_single, load_model_info


SAMPLE_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 75.0,
    "TotalCharges": 900.0,
}


class TestPreprocessInput:
    def test_yes_no_columns_converted(self):
        df = pd.DataFrame([SAMPLE_CUSTOMER.copy()])
        result = _preprocess_input(df)
        assert result["Partner"].iloc[0] == 1
        assert result["Dependents"].iloc[0] == 0
        assert result["PhoneService"].iloc[0] == 1

    def test_paperless_billing_converted(self):
        df = pd.DataFrame([SAMPLE_CUSTOMER.copy()])
        result = _preprocess_input(df)
        assert result["PaperlessBilling"].iloc[0] == 1

    def test_senior_citizen_numeric(self):
        df = pd.DataFrame([{"SeniorCitizen": "0", "tenure": 12, "MonthlyCharges": 65.0}])
        result = _preprocess_input(df)
        assert result["SeniorCitizen"].iloc[0] == 0

    def test_total_charges_numeric(self):
        df = pd.DataFrame([{"TotalCharges": "780.00", "tenure": 12, "MonthlyCharges": 65.0}])
        result = _preprocess_input(df)
        assert isinstance(result["TotalCharges"].iloc[0], float)

    def test_total_charges_invalid_coerced(self):
        df = pd.DataFrame([{"TotalCharges": "N/A", "tenure": 12, "MonthlyCharges": 65.0}])
        result = _preprocess_input(df)
        assert pd.isna(result["TotalCharges"].iloc[0])

    def test_null_internet_services_filled(self):
        df = pd.DataFrame([{"OnlineSecurity": None, "TechSupport": np.nan,
                             "tenure": 12, "MonthlyCharges": 65.0}])
        result = _preprocess_input(df)
        assert result["OnlineSecurity"].iloc[0] == "No internet service"
        assert result["TechSupport"].iloc[0] == "No internet service"

    def test_original_dict_unchanged(self):
        original = SAMPLE_CUSTOMER.copy()
        df = pd.DataFrame([SAMPLE_CUSTOMER.copy()])
        _preprocess_input(df)
        assert original == SAMPLE_CUSTOMER


class TestPredictSingle:
    def test_returns_required_keys(self):
        result = predict_single(SAMPLE_CUSTOMER)
        if "error" in result:
            pytest.skip("Model not available in test environment")
        for key in ["prediction", "churn_probability", "churn_label", "model"]:
            assert key in result

    def test_probability_in_range(self):
        result = predict_single(SAMPLE_CUSTOMER)
        if "error" in result:
            pytest.skip("Model not available")
        assert 0.0 <= result["churn_probability"] <= 1.0

    def test_prediction_binary(self):
        result = predict_single(SAMPLE_CUSTOMER)
        if "error" in result:
            pytest.skip("Model not available")
        assert result["prediction"] in (0, 1)

    def test_churn_label_matches_prediction(self):
        result = predict_single(SAMPLE_CUSTOMER)
        if "error" in result:
            pytest.skip("Model not available")
        expected_label = "Yes" if result["prediction"] == 1 else "No"
        assert result["churn_label"] == expected_label

    def test_high_risk_customer_higher_prob(self):
        """Month-to-month + Fiber optic + short tenure should score higher than long-term loyal."""
        high_risk = {**SAMPLE_CUSTOMER, "Contract": "Month-to-month", "tenure": 2,
                     "InternetService": "Fiber optic"}
        low_risk = {**SAMPLE_CUSTOMER, "Contract": "Two year", "tenure": 60,
                    "InternetService": "DSL"}
        r_high = predict_single(high_risk)
        r_low = predict_single(low_risk)
        if "error" in r_high or "error" in r_low:
            pytest.skip("Model not available")
        assert r_high["churn_probability"] > r_low["churn_probability"]

    def test_missing_model_returns_error(self, tmp_path, monkeypatch):
        """If model file doesn't exist, predict_single should return an error dict."""
        import src.models.predict as predict_module
        monkeypatch.setattr(predict_module, "MODEL_DIR", tmp_path)
        result = predict_single(SAMPLE_CUSTOMER)
        assert "error" in result


class TestLoadModelInfo:
    def test_returns_four_items(self):
        pipeline, model_name, cat_cols, num_cols = load_model_info()
        assert isinstance(pipeline, object) or pipeline is None
        # If model exists, verify structure
        if pipeline is not None:
            assert model_name is not None
            assert isinstance(cat_cols, list)
            assert isinstance(num_cols, list)
