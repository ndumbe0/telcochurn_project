import pandas as pd
import numpy as np
import logging
import joblib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def main():
    logger.info("Loading data...")
    df = pd.read_csv(DATA_DIR / "telco.csv", index_col=0)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df.drop(columns=["customerID"], errors="ignore")
    yes_no_map = {"Yes": 1, "No": 0, "True": 1, "False": 0, True: 1, False: 0}
    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().map(yes_no_map).fillna(0).astype(int)
    df["Churn"] = df["Churn"].astype(str).str.strip().map(yes_no_map)
    df = df.dropna(subset=["Churn"])
    df["Churn"] = df["Churn"].astype(int)
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        df[col] = df[col].fillna("No internet service")
    df["MultipleLines"] = df["MultipleLines"].fillna("No")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
    df = df.dropna()

    logger.info(f"Data cleaned: {df.shape[0]} rows")
    y = df["Churn"].values
    X = df.drop(columns=["Churn"])
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"Categorical: {cat_cols}")
    logger.info(f"Numerical: {num_cols}")

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols),
    ])

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=3, random_state=42, eval_metric="logloss"
    )

    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", model),
    ])

    logger.info("Training XGBoost...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    logger.info(f"Test AUC: {auc:.4f}, F1: {f1:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    joblib.dump({
        "pipeline": pipeline,
        "model_name": "XGBoost",
        "cat_cols": cat_cols,
        "num_cols": num_cols,
    }, MODEL_DIR / "best_model.pkl")
    logger.info("Model saved!")

    logger.info("Generating SHAP...")
    X_train_p = preprocessor.transform(X_train)
    X_test_p = preprocessor.transform(X_test)
    cat_feat = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)
    feature_names = num_cols + list(cat_feat)
    X_test_df = pd.DataFrame(X_test_p, columns=feature_names)
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test_df.iloc[:100])
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, X_test_df.iloc[:100], show=False)
        plt.tight_layout()
        plt.savefig(MODEL_DIR / "shap_summary.png", bbox_inches="tight", dpi=150)
        plt.close()
        logger.info("SHAP summary saved!")
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
