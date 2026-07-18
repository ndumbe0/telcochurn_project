import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
import shap

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"


def _preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw form inputs to numeric form expected by the pipeline."""
    df = df.copy()
    yes_no_map = {"Yes": 1, "No": 0, "True": 1, "False": 0}
    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().map(yes_no_map).fillna(0).astype(int)
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        if col in df.columns:
            df[col] = df[col].fillna("No internet service")
    if "MultipleLines" in df.columns:
        df["MultipleLines"] = df["MultipleLines"].fillna("No")
    return df


def predict_single(customer_data: dict) -> dict:
    pipeline, model_name, cat_cols, num_cols = load_model_info()
    if pipeline is None:
        return {"error": "Model not found"}
    df = pd.DataFrame([customer_data])
    df = _preprocess_input(df)
    proba = pipeline.predict_proba(df)[0, 1]
    pred = int(proba >= 0.5)
    return {
        "prediction": pred,
        "churn_probability": round(float(proba), 4),
        "churn_label": "Yes" if pred == 1 else "No",
        "model": model_name,
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    pipeline, model_name, cat_cols, num_cols = load_model_info()
    if pipeline is None:
        raise FileNotFoundError("Model not found")

    # Keep original identifiers if present
    id_col = None
    if "customerID" in df.columns:
        id_col = df["customerID"].reset_index(drop=True)

    # Apply full cleaning (handles Yes/No, TotalCharges, etc.) then _preprocess_input
    try:
        from src.data.load_data import clean_data
        # clean_data drops customerID and Churn automatically
        df_clean = clean_data(df)
        # Drop Churn column if present (it's the target, not a feature)
        df_clean = df_clean.drop(columns=["Churn"], errors="ignore")
    except Exception:
        df_clean = df.copy()
        df_clean = df_clean.drop(columns=["customerID", "Churn"], errors="ignore")

    df_clean = _preprocess_input(df_clean)

    # Ensure all expected columns exist
    expected_cols = (num_cols or []) + (cat_cols or [])
    missing = [c for c in expected_cols if c not in df_clean.columns]
    if missing:
        raise ValueError(
            f"Uploaded file is missing required columns: {missing}. "
            f"Please include all customer feature columns."
        )

    proba = pipeline.predict_proba(df_clean)[:, 1]
    preds = (proba >= 0.5).astype(int)

    result = df_clean.copy()
    if id_col is not None:
        result.insert(0, "customerID", id_col.values)
    result["Churn_Probability"] = (proba * 100).round(1)
    result["Prediction"] = ["Churn" if p == 1 else "Stay" for p in preds]
    result["Risk_Level"] = pd.cut(
        proba,
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
    return result


def get_shap_values(customer_data: dict, pipeline=None, cat_cols=None, num_cols=None):
    if pipeline is None:
        pipeline, model_name, cat_cols, num_cols = load_model_info()
    if pipeline is None:
        return None, None
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["classifier"]
    df = pd.DataFrame([customer_data])
    df = _preprocess_input(df)
    X_processed = preprocessor.transform(df)
    cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)
    feature_names = num_cols + list(cat_feature_names)
    X_df = pd.DataFrame(X_processed, columns=feature_names)
    try:
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_df)
        shap_values = explainer.shap_values(X_df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        return shap_values[0], feature_names
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")
        coef = getattr(model, "coef_", None)
        if coef is not None:
            return coef * X_df.values[0], feature_names
        return None, None


def load_model_info():
    model_path = MODEL_DIR / "best_model.pkl"
    if not model_path.exists():
        return None, None, None, None
    info = joblib.load(model_path)
    return info["pipeline"], info["model_name"], info["cat_cols"], info["num_cols"]
