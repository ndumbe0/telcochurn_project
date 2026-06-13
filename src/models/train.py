import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def build_preprocessor(cat_cols: list, num_cols: list) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols),
    ])


def build_models(random_state: int = 42) -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=random_state, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state, class_weight="balanced", max_depth=10),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=random_state, class_weight="balanced", n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=50, random_state=random_state, scale_pos_weight=3, eval_metric="logloss"),
        "LightGBM": lgb.LGBMClassifier(n_estimators=50, random_state=random_state, class_weight="balanced", verbose=-1),
    }


def evaluate_model(pipeline, X_train, y_train, X_test, y_test, name: str, cv: int = 2) -> dict:
    logger.info(f"Evaluating {name}...")
    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        logger.error(f"Error fitting {name}: {e}")
        return {"name": name, "roc_auc": 0, "f1": 0, "error": str(e)}
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred
    try:
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=StratifiedKFold(n_splits=cv), scoring="roc_auc")
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
    except Exception:
        cv_mean, cv_std = 0, 0
    return {
        "name": name,
        "accuracy": pipeline.score(X_test, y_test),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "cv_auc_mean": cv_mean,
        "cv_auc_std": cv_std,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def train_models_pipeline(X_train, y_train, X_test, y_test, cat_cols, num_cols) -> tuple:
    logger.info("Starting model training pipeline...")
    preprocessor = build_preprocessor(cat_cols, num_cols)
    results = []
    trained_models = {}
    models = build_models()
    for name, model in models.items():
        pipeline = ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42, sampling_strategy=0.8)),
            ("classifier", model),
        ])
        result = evaluate_model(pipeline, X_train, y_train, X_test, y_test, name)
        results.append(result)
        trained_models[name] = pipeline
        logger.info(f"{name}: AUC={result.get('roc_auc', 0):.4f}, F1={result.get('f1', 0):.4f}")

    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)
    sm = SMOTE(random_state=42, sampling_strategy=0.8)
    X_train_r, y_train_r = sm.fit_resample(X_train_p, y_train)
    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train_r, y_train_r)
        fitted_models[name] = model
    ensemble = VotingClassifier(
        estimators=[(n, m) for n, m in fitted_models.items()],
        voting="soft", n_jobs=-1
    )
    ensemble.fit(X_train_r, y_train_r)
    y_pred_ens = ensemble.predict(X_test_p)
    y_proba_ens = ensemble.predict_proba(X_test_p)[:, 1]
    result_ens = {
        "name": "Voting Ensemble",
        "accuracy": ensemble.score(X_test_p, y_test),
        "precision": precision_score(y_test, y_pred_ens, zero_division=0),
        "recall": recall_score(y_test, y_pred_ens, zero_division=0),
        "f1": f1_score(y_test, y_pred_ens, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba_ens),
        "cv_auc_mean": 0,
        "cv_auc_std": 0,
        "confusion_matrix": confusion_matrix(y_test, y_pred_ens).tolist(),
        "y_pred": y_pred_ens,
        "y_proba": y_proba_ens,
    }
    results.append(result_ens)
    trained_models["Voting Ensemble"] = ensemble
    logger.info(f"Voting Ensemble: AUC={result_ens.get('roc_auc', 0):.4f}, F1={result_ens.get('f1', 0):.4f}")

    metrics_df = pd.DataFrame(results).drop(columns=["y_pred", "y_proba", "confusion_matrix"], errors="ignore")
    metrics_df.to_csv(MODEL_DIR / "model_comparison.csv", index=False)
    return results, trained_models


def tune_best_model(results, trained_models, X_train, y_train, X_test, y_test, cat_cols, num_cols):
    valid = [r for r in results if r.get("roc_auc", 0) > 0]
    if not valid:
        logger.warning("No valid models, using XGBoost as default")
        best_name = "XGBoost"
    else:
        best_idx = np.argmax([r.get("roc_auc", 0) for r in valid])
        best_name = valid[best_idx]["name"]
    if best_name == "Voting Ensemble":
        best_name = "XGBoost"
    logger.info(f"Best model: {best_name}. Tuning...")
    model = build_models()[best_name]
    preprocessor = build_preprocessor(cat_cols, num_cols)
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)
    param_grids = {
        "XGBoost": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
        },
        "LightGBM": {
            "n_estimators": [100, 200],
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.01, 0.05, 0.1],
        },
        "Random Forest": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, 15],
        },
        "Logistic Regression": {
            "C": [0.1, 1.0, 10.0],
            "solver": ["liblinear", "lbfgs"],
        },
    }
    param_grid = param_grids.get(best_name, {"n_estimators": [100, 200]})
    smote = SMOTE(random_state=42, sampling_strategy=0.8)
    X_train_r, y_train_r = smote.fit_resample(X_train_p, y_train)
    n_iter = min(3, len(param_grid.get(list(param_grid.keys())[0], [100])))
    search = RandomizedSearchCV(model, param_grid, n_iter=n_iter, cv=StratifiedKFold(2),
                                scoring="roc_auc", n_jobs=-1, random_state=42)
    search.fit(X_train_r, y_train_r)
    logger.info(f"Best params for {best_name}: {search.best_params_}")
    logger.info(f"Best CV AUC: {search.best_score_:.4f}")
    final_pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", search.best_estimator_),
    ])
    final_pipeline.fit(X_train, y_train)
    y_proba = final_pipeline.predict_proba(X_test)[:, 1]
    y_pred = final_pipeline.predict(X_test)
    test_auc = roc_auc_score(y_test, y_proba)
    test_f1 = f1_score(y_test, y_pred)
    logger.info(f"Tuned {best_name} Test AUC: {test_auc:.4f}, F1: {test_f1:.4f}")
    return final_pipeline, best_name


def generate_shap_explanation(pipeline, X_train, X_test, cat_cols, num_cols, sample_size: int = 100):
    logger.info("Generating SHAP explanations...")
    preprocessor = pipeline.named_steps["preprocessor"]
    X_train_p = preprocessor.transform(X_train)
    X_test_p = preprocessor.transform(X_test)
    model = pipeline.named_steps["classifier"]
    cat_feat = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)
    feature_names = num_cols + list(cat_feat)
    X_test_df = pd.DataFrame(X_test_p, columns=feature_names)
    X_train_df = pd.DataFrame(X_train_p, columns=feature_names)
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test_df.iloc[:sample_size])
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, X_test_df.iloc[:sample_size], show=False)
        plt.tight_layout()
        plt.savefig(MODEL_DIR / "shap_summary.png", bbox_inches="tight", dpi=150)
        plt.close()
        logger.info("SHAP summary saved.")
        return shap_vals, feature_names
    except Exception as e:
        logger.warning(f"TreeExplainer failed: {e}")
        try:
            explainer = shap.LinearExplainer(model, X_train_df.iloc[:100])
            shap_vals = explainer.shap_values(X_test_df.iloc[:sample_size])
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_vals, X_test_df.iloc[:sample_size], show=False)
            plt.tight_layout()
            plt.savefig(MODEL_DIR / "shap_summary.png", bbox_inches="tight", dpi=150)
            plt.close()
            return shap_vals, feature_names
        except Exception as e2:
            logger.error(f"SHAP failed: {e2}")
            return None, None


def save_best_model(pipeline, name: str, cat_cols: list, num_cols: list):
    joblib.dump({
        "pipeline": pipeline,
        "model_name": name,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
    }, MODEL_DIR / "best_model.pkl")
    logger.info(f"Best model saved ({name}).")


def load_best_model():
    path = MODEL_DIR / "best_model.pkl"
    if not path.exists():
        return None, None, None, None
    info = joblib.load(path)
    return info["pipeline"], info["model_name"], info["cat_cols"], info["num_cols"]


def run_training_pipeline(X_train, y_train, X_test, y_test, cat_cols, num_cols):
    results, trained_models = train_models_pipeline(X_train, y_train, X_test, y_test, cat_cols, num_cols)
    best_pipeline, best_name = tune_best_model(results, trained_models, X_train, y_train, X_test, y_test, cat_cols, num_cols)
    save_best_model(best_pipeline, best_name, cat_cols, num_cols)
    generate_shap_explanation(best_pipeline, X_train, X_test, cat_cols, num_cols)
    return best_pipeline, results
