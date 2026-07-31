"""
Data Pipeline & Model Training tab.
Five sequential stages:
  1. Load   — upload a CSV/Excel or pick an existing file from data/
  2. Clean  — validate, clean, preview before/after, save cleaned CSV
  3. Config — test-size, random-state, class-balance, feature summary
  4. Train  — Quick (XGBoost, ~30 s) or Full (5 models + tuning, ~5–10 min)
  5. Results— metrics, model comparison chart, SHAP image, download buttons
"""
from __future__ import annotations

import io
import logging
import queue
import sys
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.load_data import clean_data, preprocess_features, split_data, save_processed_data
from src.models.train import (
    build_preprocessor,
    build_models,
    generate_shap_explanation,
    save_best_model,
    run_training_pipeline,
)

DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
PROC_DIR  = ROOT / "data" / "processed"

# ── Session-state keys used by this tab ────────────────────────────────────────
_S_RAW      = "pipe_raw_df"
_S_CLEAN    = "pipe_clean_df"
_S_FILENAME = "pipe_filename"
_S_CLEAN_OK = "pipe_clean_ok"
_S_SPLIT    = "pipe_split_cfg"
_S_TRAINED  = "pipe_train_results"   # dict produced after training
_S_STEP     = "pipe_active_step"


def _init_state():
    defaults = {
        _S_RAW:      None,
        _S_CLEAN:    None,
        _S_FILENAME: None,
        _S_CLEAN_OK: False,
        _S_SPLIT:    {"test_size": 0.20, "random_state": 42},
        _S_TRAINED:  None,
        _S_STEP:     1,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_file(uploaded) -> pd.DataFrame | None:
    """Read an uploaded Streamlit file object → DataFrame."""
    name = uploaded.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded)
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded)
        st.error(f"Unsupported file type: {uploaded.name}")
    except Exception as e:
        st.error(f"Could not read file: {e}")
    return None


def _existing_data_files() -> list[Path]:
    """Return CSV + Excel files already present in data/."""
    exts = {".csv", ".xlsx", ".xls"}
    files = sorted(
        p for p in DATA_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )
    return files


def _null_report(df: pd.DataFrame) -> pd.DataFrame:
    nulls = df.isnull().sum()
    pct   = (nulls / len(df) * 100).round(2)
    return pd.DataFrame({"Missing": nulls, "Missing %": pct})[nulls > 0]


def _class_balance_fig(df: pd.DataFrame) -> go.Figure:
    counts = df["Churn"].value_counts().reset_index()
    counts.columns = ["Churn", "Count"]
    counts["Label"] = counts["Churn"].map({0: "Stay", 1: "Churn"})
    fig = px.bar(
        counts, x="Label", y="Count",
        color="Label",
        color_discrete_map={"Churn": "#d62728", "Stay": "#1f77b4"},
        text="Count",
        title="Class Balance",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=280, showlegend=False)
    return fig


def _metrics_from_arrays(y_test, y_pred, y_proba) -> dict:
    return {
        "ROC-AUC":  round(roc_auc_score(y_test, y_proba), 4),
        "F1":       round(f1_score(y_test, y_pred, zero_division=0), 4),
        "Precision":round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":   round(recall_score(y_test, y_pred, zero_division=0), 4),
        "Accuracy": round(float(np.mean(y_pred == y_test)), 4),
    }


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


# ── Stage renderers ────────────────────────────────────────────────────────────

def _stage_load():
    st.subheader("📥 Step 1 — Load Raw Data")
    st.markdown(
        "Upload a **CSV or Excel** file containing raw Telco churn data, "
        "or pick one of the files already in the project."
    )

    src_choice = st.radio(
        "Data source",
        ["⬆️ Upload a file", "📂 Use an existing project file"],
        horizontal=True,
        key="pipe_src_choice",
    )

    raw_df    = None
    file_name = None

    if src_choice.startswith("⬆️"):
        uploaded = st.file_uploader(
            "Upload CSV or Excel (max 50 MB)",
            type=["csv", "xlsx", "xls"],
            key="pipe_upload",
        )
        if uploaded:
            if uploaded.size > 50 * 1024 * 1024:
                st.error("File exceeds 50 MB — please upload a smaller file.")
                return
            with st.spinner("Reading file…"):
                raw_df    = _load_file(uploaded)
                file_name = uploaded.name
    else:
        existing = _existing_data_files()
        if not existing:
            st.warning("No CSV/Excel files found in data/.")
            return
        choice = st.selectbox(
            "Select file",
            existing,
            format_func=lambda p: p.name,
            key="pipe_existing_choice",
        )
        if st.button("Load selected file", key="pipe_load_existing"):
            with st.spinner(f"Loading {choice.name}…"):
                try:
                    if choice.suffix.lower() == ".csv":
                        raw_df = pd.read_csv(choice)
                    else:
                        raw_df = pd.read_excel(choice)
                    file_name = choice.name
                except Exception as e:
                    st.error(f"Could not load {choice.name}: {e}")
                    return
        elif st.session_state[_S_RAW] is not None:
            # keep whatever was previously loaded
            raw_df    = st.session_state[_S_RAW]
            file_name = st.session_state[_S_FILENAME]

    if raw_df is None:
        # Nothing loaded yet — show required columns hint
        with st.expander("ℹ️ Required columns"):
            st.markdown("""
The pipeline expects these **21 standard Telco-churn columns** (case-sensitive):

| Column | Type | Notes |
|---|---|---|
| `customerID` | str | Optional — dropped during cleaning |
| `gender` | str | Male / Female |
| `SeniorCitizen` | int/bool | 0 / 1 |
| `Partner` | str/bool | Yes / No / True / False |
| `Dependents` | str/bool | Yes / No |
| `tenure` | int | Months |
| `PhoneService` | str/bool | Yes / No |
| `MultipleLines` | str | Yes / No / No phone service |
| `InternetService` | str | DSL / Fiber optic / No |
| `OnlineSecurity` | str | Yes / No / No internet service |
| `OnlineBackup` | str | Yes / No / No internet service |
| `DeviceProtection` | str | Yes / No / No internet service |
| `TechSupport` | str | Yes / No / No internet service |
| `StreamingTV` | str | Yes / No / No internet service |
| `StreamingMovies` | str | Yes / No / No internet service |
| `Contract` | str | Month-to-month / One year / Two year |
| `PaperlessBilling` | str/bool | Yes / No |
| `PaymentMethod` | str | Electronic check / Mailed check / … |
| `MonthlyCharges` | float | Monthly bill |
| `TotalCharges` | float/str | Cumulative bill (NaN → auto-filled) |
| `Churn` | str/bool/int | **Target** — Yes/No/True/False/1/0 |
""")
        return

    # ── Store & show preview ───────────────────────────────────────────────────
    st.session_state[_S_RAW]      = raw_df
    st.session_state[_S_FILENAME] = file_name
    # Reset downstream state when a new file is loaded
    st.session_state[_S_CLEAN]    = None
    st.session_state[_S_CLEAN_OK] = False
    st.session_state[_S_TRAINED]  = None

    st.success(f"✅ Loaded **{file_name}** — {raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{raw_df.shape[0]:,}")
    c2.metric("Columns", raw_df.shape[1])
    total_nulls = int(raw_df.isnull().sum().sum())
    c3.metric("Total Missing Values", total_nulls)

    st.markdown("**Preview (first 5 rows)**")
    st.dataframe(raw_df.head(5), use_container_width=True)

    null_rep = _null_report(raw_df)
    if not null_rep.empty:
        with st.expander(f"⚠️ Missing value report ({len(null_rep)} columns affected)"):
            st.dataframe(null_rep, use_container_width=True)

    # Check for Churn column
    if "Churn" not in raw_df.columns:
        st.warning("⚠️ No **Churn** column found. The pipeline can still clean and split the data, but **model training requires a Churn column**.")

    st.info("✅ Data loaded — proceed to **Step 2: Clean**")
    st.session_state[_S_STEP] = max(st.session_state[_S_STEP], 2)


def _stage_clean():
    st.subheader("🧹 Step 2 — Clean & Validate")

    raw_df = st.session_state.get(_S_RAW)
    if raw_df is None:
        st.info("👈 Complete **Step 1** first to load data.")
        return

    st.markdown(f"**Source:** `{st.session_state[_S_FILENAME]}` — {raw_df.shape[0]:,} rows × {raw_df.shape[1]} cols")

    if st.button("▶️ Run Cleaning Pipeline", key="pipe_run_clean", type="primary"):
        with st.status("Cleaning data…", expanded=True) as status:
            try:
                st.write("• Dropping `customerID`")
                st.write("• Mapping Yes/No/True/False → 1/0")
                st.write("• Filling internet-service nulls with 'No internet service'")
                st.write("• Coercing `TotalCharges` to numeric (NaN → MonthlyCharges × tenure)")
                st.write("• Dropping rows with invalid / missing `Churn` label")
                clean_df = clean_data(raw_df)
                st.session_state[_S_CLEAN]    = clean_df
                st.session_state[_S_CLEAN_OK] = True
                status.update(label="✅ Cleaning complete", state="complete")
            except Exception as e:
                status.update(label=f"❌ Cleaning failed: {e}", state="error")
                st.error(str(e))
                return

    clean_df = st.session_state.get(_S_CLEAN)
    if clean_df is None:
        return

    # ── Before / After ────────────────────────────────────────────────────────
    st.markdown("### Before vs After")
    b1, b2, b3, b4 = st.columns(4)
    rows_removed = raw_df.shape[0] - clean_df.shape[0]
    cols_dropped  = raw_df.shape[1] - clean_df.shape[1]
    nulls_before  = int(raw_df.isnull().sum().sum())
    nulls_after   = int(clean_df.isnull().sum().sum())

    b1.metric("Rows Before → After", f"{raw_df.shape[0]:,} → {clean_df.shape[0]:,}",
              delta=f"-{rows_removed:,}" if rows_removed else "0", delta_color="inverse")
    b2.metric("Columns Before → After", f"{raw_df.shape[1]} → {clean_df.shape[1]}",
              delta=f"-{cols_dropped}" if cols_dropped else "0", delta_color="inverse")
    b3.metric("Missing Values Before", f"{nulls_before:,}")
    b4.metric("Missing Values After",  f"{nulls_after:,}", delta_color="inverse")

    st.markdown("**Cleaned data preview**")
    st.dataframe(clean_df.head(8), use_container_width=True)

    # ── Data types summary ────────────────────────────────────────────────────
    with st.expander("📋 Column types & sample values"):
        dtype_df = pd.DataFrame({
            "Column":    clean_df.columns,
            "Type":      clean_df.dtypes.values.astype(str),
            "Unique":    clean_df.nunique().values,
            "Sample":    [str(clean_df[c].dropna().iloc[0]) if len(clean_df[c].dropna()) > 0 else "—"
                          for c in clean_df.columns],
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    # ── Class balance ─────────────────────────────────────────────────────────
    if "Churn" in clean_df.columns:
        st.markdown("### Class Balance")
        cc1, cc2 = st.columns([1, 2])
        with cc1:
            churn_counts = clean_df["Churn"].value_counts()
            for val, label in [(1, "Churn"), (0, "Stay")]:
                n = int(churn_counts.get(val, 0))
                pct = n / len(clean_df) * 100
                st.metric(label, f"{n:,}  ({pct:.1f}%)")
            ratio = int(churn_counts.get(0, 1)) / max(int(churn_counts.get(1, 1)), 1)
            st.caption(f"Class imbalance ratio (Stay:Churn) = {ratio:.1f}:1")
            st.caption("SMOTE will oversample the minority class during training.")
        with cc2:
            st.plotly_chart(_class_balance_fig(clean_df), use_container_width=True)

    # ── Save to disk ──────────────────────────────────────────────────────────
    st.markdown("### Save Cleaned Data")
    save_col1, save_col2 = st.columns(2)
    with save_col1:
        save_path = PROC_DIR / "cleaned_pipeline.csv"
        if st.button("💾 Save to data/processed/cleaned_pipeline.csv", key="pipe_save_clean"):
            PROC_DIR.mkdir(parents=True, exist_ok=True)
            clean_df.to_csv(save_path, index=False)
            st.success(f"✅ Saved to `{save_path.relative_to(ROOT)}`")
    with save_col2:
        st.download_button(
            "⬇️ Download cleaned CSV",
            data=_df_to_csv_bytes(clean_df),
            file_name="telco_cleaned.csv",
            mime="text/csv",
            key="pipe_dl_clean",
        )

    st.info("✅ Data cleaned — proceed to **Step 3: Configure**")
    st.session_state[_S_STEP] = max(st.session_state[_S_STEP], 3)


def _stage_configure():
    st.subheader("⚙️ Step 3 — Configure Train / Test Split")

    clean_df = st.session_state.get(_S_CLEAN)
    if clean_df is None:
        st.info("👈 Complete **Step 2** first to clean the data.")
        return

    if "Churn" not in clean_df.columns:
        st.error("No **Churn** column in the cleaned data — model training is not possible.")
        return

    st.markdown(f"**Dataset:** {clean_df.shape[0]:,} rows after cleaning")

    # ── Split settings ────────────────────────────────────────────────────────
    cc1, cc2 = st.columns(2)
    with cc1:
        test_size = st.slider(
            "Test set size", 0.10, 0.40, 0.20, 0.05,
            format="%g%%", key="pipe_test_size",
            help="Fraction of data held out for evaluation."
        ) 
        # slider returns 0.10..0.40 — treat as fraction
    with cc2:
        random_state = st.number_input(
            "Random seed", 0, 9999, 42, 1,
            key="pipe_random_state",
            help="Controls reproducibility of the split."
        )

    st.session_state[_S_SPLIT] = {
        "test_size":    test_size,
        "random_state": int(random_state),
    }

    X, y, cat_cols, num_cols = preprocess_features(clean_df, target_col="Churn")
    n_train = int(len(X) * (1 - test_size))
    n_test  = len(X) - n_train

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Training rows",   f"{n_train:,}")
    m2.metric("Test rows",       f"{n_test:,}")
    m3.metric("Categorical features", len(cat_cols))
    m4.metric("Numeric features",     len(num_cols))

    # ── Feature breakdown ─────────────────────────────────────────────────────
    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown("**Categorical features**")
        cat_info = pd.DataFrame({
            "Feature":  cat_cols,
            "Unique":   [clean_df[c].nunique() for c in cat_cols],
            "Top value":[str(clean_df[c].mode().iloc[0]) for c in cat_cols],
        })
        st.dataframe(cat_info, use_container_width=True, hide_index=True)
    with fc2:
        st.markdown("**Numeric features**")
        num_info = clean_df[num_cols].agg(["mean", "std", "min", "max"]).T.round(2)
        num_info.index.name = "Feature"
        st.dataframe(num_info, use_container_width=True)

    # ── Stratification check ──────────────────────────────────────────────────
    churn_rate = clean_df["Churn"].mean()
    st.info(
        f"ℹ️ Overall churn rate: **{churn_rate:.1%}**. "
        "Stratified splitting ensures the same ratio in train and test sets. "
        "SMOTE will synthetically oversample the minority class in the training set only."
    )

    st.info("✅ Configuration done — proceed to **Step 4: Train**")
    st.session_state[_S_STEP] = max(st.session_state[_S_STEP], 4)


# ── Logging capture helper ─────────────────────────────────────────────────────

class _QueueHandler(logging.Handler):
    """Push log records into a queue for Streamlit to consume."""
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record):
        try:
            self.q.put_nowait(self.format(record))
        except Exception:
            pass


def _run_quick_train(clean_df, test_size, random_state, log_q):
    """Run fast single-model (XGBoost) training. Returns result dict."""
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    import xgboost as xgb
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    log = logging.getLogger("pipeline.quick")
    log.setLevel(logging.INFO)
    log.addHandler(_QueueHandler(log_q))

    log.info("Preprocessing features…")
    X, y, cat_cols, num_cols = preprocess_features(clean_df, target_col="Churn")

    log.info(f"Splitting data — test_size={test_size}, random_state={random_state}")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=random_state)
    log.info(f"  Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    log.info("Building preprocessor (StandardScaler + OneHotEncoder)…")
    preprocessor = build_preprocessor(cat_cols, num_cols)

    log.info("Building XGBoost pipeline with SMOTE…")
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=3, random_state=42,
        eval_metric="logloss",
    )
    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42, sampling_strategy=0.8)),
        ("classifier", model),
    ])

    log.info("Training XGBoost…")
    pipeline.fit(X_train, y_train)

    log.info("Running 5-fold CV for AUC estimate…")
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=StratifiedKFold(5), scoring="roc_auc", n_jobs=-1,
    )
    log.info(f"  CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = _metrics_from_arrays(y_test, y_pred, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()

    log.info(f"  Test AUC={metrics['ROC-AUC']:.4f}  F1={metrics['F1']:.4f}")
    log.info("Saving model to models/best_model.pkl…")
    save_best_model(pipeline, "XGBoost", cat_cols, num_cols)

    log.info("Saving processed train/test CSVs…")
    save_processed_data(X_train, X_test, y_train, y_test, cat_cols, num_cols)

    log.info("Generating SHAP summary plot…")
    generate_shap_explanation(pipeline, X_train, X_test, cat_cols, num_cols)

    log.info("✅ Quick training complete!")
    return {
        "mode":       "Quick (XGBoost)",
        "best_model": "XGBoost",
        "metrics":    metrics,
        "cv_auc":     float(cv_scores.mean()),
        "cv_std":     float(cv_scores.std()),
        "confusion":  cm,
        "cat_cols":   cat_cols,
        "num_cols":   num_cols,
        "comparison": pd.DataFrame([{"Model": "XGBoost", **metrics,
                                      "CV AUC": round(float(cv_scores.mean()), 4)}]),
    }


def _run_full_train(clean_df, test_size, random_state, log_q):
    """Run full 5-model comparison + XGBoost/LGBM tuning."""
    log = logging.getLogger("pipeline.full")
    log.setLevel(logging.INFO)
    log.addHandler(_QueueHandler(log_q))

    log.info("Preprocessing features…")
    X, y, cat_cols, num_cols = preprocess_features(clean_df, target_col="Churn")

    log.info(f"Splitting — test_size={test_size}, random_state={random_state}")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=random_state)
    log.info(f"  Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    log.info("Saving processed splits to data/processed/…")
    save_processed_data(X_train, X_test, y_train, y_test, cat_cols, num_cols)

    log.info("Running full model training pipeline (5 models + tuning)…")
    best_pipeline, all_results = run_training_pipeline(
        X_train, y_train, X_test, y_test, cat_cols, num_cols
    )

    # Best model metrics
    y_pred  = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    metrics = _metrics_from_arrays(y_test, y_pred, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Load model name
    best_info = joblib.load(MODEL_DIR / "best_model.pkl")
    best_name = best_info.get("model_name", "Best Model")

    log.info(f"Best model: {best_name}  AUC={metrics['ROC-AUC']:.4f}  F1={metrics['F1']:.4f}")
    log.info("Generating SHAP summary plot…")
    generate_shap_explanation(best_pipeline, X_train, X_test, cat_cols, num_cols)
    log.info("✅ Full training complete!")

    # Build comparison df
    rows = []
    for r in all_results:
        rows.append({
            "Model":     r["name"],
            "ROC-AUC":   round(r.get("roc_auc", 0), 4),
            "F1":        round(r.get("f1", 0), 4),
            "Precision": round(r.get("precision", 0), 4),
            "Recall":    round(r.get("recall", 0), 4),
            "Accuracy":  round(r.get("accuracy", 0), 4),
            "CV AUC":    round(r.get("cv_auc_mean", 0), 4),
        })
    comparison_df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)

    return {
        "mode":       "Full (5 models + tuning)",
        "best_model": best_name,
        "metrics":    metrics,
        "cv_auc":     None,
        "cv_std":     None,
        "confusion":  cm,
        "cat_cols":   cat_cols,
        "num_cols":   num_cols,
        "comparison": comparison_df,
    }


def _stage_train():
    st.subheader("🚀 Step 4 — Train & Evaluate")

    clean_df = st.session_state.get(_S_CLEAN)
    if clean_df is None:
        st.info("👈 Complete **Steps 1–2** first.")
        return
    if "Churn" not in clean_df.columns:
        st.error("No **Churn** column — cannot train.")
        return

    split_cfg = st.session_state.get(_S_SPLIT, {"test_size": 0.20, "random_state": 42})

    # ── Mode selector ──────────────────────────────────────────────────────────
    mode = st.radio(
        "Training mode",
        ["⚡ Quick  (XGBoost only, ~30 s)", "🏆 Full  (5 models + hyperparameter tuning, ~5–10 min)"],
        key="pipe_train_mode",
        captions=[
            "XGBoost with 5-fold CV. Best for rapid iteration.",
            "Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM + Voting Ensemble + RandomizedSearchCV tuning. Overwrites best_model.pkl.",
        ],
    )
    quick = mode.startswith("⚡")

    c1, c2, c3 = st.columns(3)
    c1.metric("Training rows", f"{int(len(clean_df) * (1 - split_cfg['test_size'])):,}")
    c2.metric("Test rows",     f"{int(len(clean_df) * split_cfg['test_size']):,}")
    c3.metric("Mode",          "Quick" if quick else "Full (5 models)")

    already_trained = st.session_state.get(_S_TRAINED) is not None
    if already_trained:
        prev = st.session_state[_S_TRAINED]
        st.success(
            f"✅ Previous run: **{prev['best_model']}** — "
            f"AUC {prev['metrics']['ROC-AUC']:.4f} | F1 {prev['metrics']['F1']:.4f}. "
            "Click below to retrain."
        )

    btn_label = "🔄 Retrain Model" if already_trained else "🚀 Start Training"
    if not st.button(btn_label, key="pipe_train_btn", type="primary"):
        return

    # ── Run training with live log output ─────────────────────────────────────
    log_q: queue.Queue = queue.Queue()
    result_holder: list = []
    error_holder:  list = []

    def _worker():
        try:
            fn = _run_quick_train if quick else _run_full_train
            res = fn(clean_df, split_cfg["test_size"], split_cfg["random_state"], log_q)
            result_holder.append(res)
        except Exception as exc:
            error_holder.append(str(exc))
        finally:
            log_q.put(None)   # sentinel

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    with st.status("Training in progress…", expanded=True) as status:
        log_lines = []
        log_container = st.empty()
        while True:
            try:
                msg = log_q.get(timeout=0.3)
            except queue.Empty:
                if not thread.is_alive():
                    break
                continue
            if msg is None:
                break
            log_lines.append(msg)
            # Keep last 40 lines so Streamlit doesn't slow down
            display = log_lines[-40:]
            log_container.code("\n".join(display), language=None)

        thread.join(timeout=5)

        if error_holder:
            status.update(label=f"❌ Training failed", state="error")
            st.error(error_holder[0])
            return

        if not result_holder:
            status.update(label="❌ Training produced no result", state="error")
            return

        status.update(label="✅ Training complete!", state="complete", expanded=False)

    result = result_holder[0]
    st.session_state[_S_TRAINED] = result
    st.session_state[_S_STEP]    = max(st.session_state[_S_STEP], 5)

    # Clear cached model so app reloads the new one
    st.cache_resource.clear()

    m = result["metrics"]
    st.markdown("### 🎯 Results at a glance")
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("ROC-AUC",   m["ROC-AUC"])
    r2.metric("F1",        m["F1"])
    r3.metric("Precision", m["Precision"])
    r4.metric("Recall",    m["Recall"])
    r5.metric("Accuracy",  m["Accuracy"])

    st.success(
        f"✅ **{result['best_model']}** saved to `models/best_model.pkl`. "
        "The rest of the app now uses this model — no restart required."
    )
    st.info("✅ Training done — view detailed results in **Step 5: Results**")


def _stage_results():
    st.subheader("📊 Step 5 — Results & Downloads")

    result = st.session_state.get(_S_TRAINED)
    if result is None:
        st.info("👈 Complete **Step 4** to train a model first.")
        return

    m     = result["metrics"]
    mode  = result["mode"]
    best  = result["best_model"]
    comp  = result["comparison"]
    cm    = result["confusion"]

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"**Training mode:** {mode}  |  **Best model:** {best}")

    # ── Metric cards ──────────────────────────────────────────────────────────
    st.markdown("### 📐 Test Set Metrics")
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("ROC-AUC",   m["ROC-AUC"])
    r2.metric("F1",        m["F1"])
    r3.metric("Precision", m["Precision"])
    r4.metric("Recall",    m["Recall"])
    r5.metric("Accuracy",  m["Accuracy"])

    if result.get("cv_auc") is not None:
        st.caption(
            f"5-fold CV AUC on training set: "
            f"{result['cv_auc']:.4f} ± {result['cv_std']:.4f}"
        )

    st.markdown("---")

    # ── Model comparison leaderboard ──────────────────────────────────────────
    st.markdown("### 🏅 Model Comparison")
    comp_sorted = comp.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    comp_sorted.index += 1
    st.dataframe(
        comp_sorted.style
            .background_gradient(subset=["ROC-AUC", "F1"], cmap="Blues")
            .format({"ROC-AUC": "{:.4f}", "F1": "{:.4f}",
                     "Precision": "{:.4f}", "Recall": "{:.4f}",
                     "Accuracy": "{:.4f}", "CV AUC": "{:.4f}"}),
        use_container_width=True,
    )

    if len(comp) > 1:
        fig_comp = px.bar(
            comp_sorted, x="Model", y="ROC-AUC",
            color="ROC-AUC", color_continuous_scale="Blues",
            text=[f"{v:.4f}" for v in comp_sorted["ROC-AUC"]],
            title="Model ROC-AUC Comparison",
        )
        fig_comp.update_traces(textposition="outside")
        fig_comp.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    st.markdown("### 🔢 Confusion Matrix (best model on test set)")
    if cm:
        cm_arr = np.array(cm)
        labels = ["Stay (0)", "Churn (1)"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm_arr,
            x=[f"Predicted {l}" for l in labels],
            y=[f"Actual {l}"    for l in labels],
            colorscale="Blues",
            text=cm_arr, texttemplate="%{text}",
            showscale=False,
        ))
        fig_cm.update_layout(
            title=f"Confusion Matrix — {best}",
            height=350,
        )
        # Annotations for TP / TN / FP / FN
        fig_cm.add_annotation(x=0.5, y=1.1, xref="paper", yref="paper",
            text=f"TN={cm_arr[0,0]}  FP={cm_arr[0,1]}  FN={cm_arr[1,0]}  TP={cm_arr[1,1]}",
            showarrow=False, font=dict(size=12))
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")

    # ── SHAP summary ──────────────────────────────────────────────────────────
    shap_path = MODEL_DIR / "shap_summary.png"
    if shap_path.exists():
        st.markdown("### 🧠 SHAP Feature Importance")
        st.markdown(
            "The SHAP beeswarm plot shows which features drive the model's predictions "
            "and in which direction. Colour = feature value (red = high, blue = low). "
            "X-axis = impact on churn probability."
        )
        st.image(str(shap_path), use_container_width=True)
    else:
        st.info("SHAP summary image not yet generated — it will appear here after training.")

    st.markdown("---")

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.markdown("### ⬇️ Downloads")
    dl1, dl2, dl3 = st.columns(3)

    # Cleaned CSV
    clean_df = st.session_state.get(_S_CLEAN)
    if clean_df is not None:
        dl1.download_button(
            "📄 Cleaned data (CSV)",
            data=_df_to_csv_bytes(clean_df),
            file_name="telco_cleaned.csv",
            mime="text/csv",
            key="res_dl_clean",
        )

    # Model comparison CSV
    dl2.download_button(
        "📊 Model comparison (CSV)",
        data=comp.to_csv(index=False).encode(),
        file_name="model_comparison.csv",
        mime="text/csv",
        key="res_dl_comp",
    )

    # Train / test sets
    train_path = PROC_DIR / "train.csv"
    test_path  = PROC_DIR / "test.csv"
    if train_path.exists() and test_path.exists():
        train_bytes = train_path.read_bytes()
        test_bytes  = test_path.read_bytes()
        with dl3:
            st.download_button(
                "🚂 Training set (CSV)",
                data=train_bytes,
                file_name="train.csv",
                mime="text/csv",
                key="res_dl_train",
            )
            st.download_button(
                "🧪 Test set (CSV)",
                data=test_bytes,
                file_name="test.csv",
                mime="text/csv",
                key="res_dl_test",
            )

    st.markdown("---")
    st.success(
        f"🎉 Pipeline complete! **{best}** is now the active model. "
        "Head to **Model Performance** tab for ROC curves, calibration plots, "
        "and threshold analysis using the new model."
    )


# ── Public entry point ─────────────────────────────────────────────────────────

def render_data_pipeline():
    st.header("🔄 Data Pipeline & Model Training")
    st.markdown(
        "End-to-end pipeline: **upload raw data → clean → configure → train ML models → "
        "inspect results**. The trained model is saved and immediately used across the entire app."
    )

    _init_state()

    # ── Progress indicator ────────────────────────────────────────────────────
    step   = st.session_state.get(_S_STEP, 1)
    labels = ["1. Load", "2. Clean", "3. Configure", "4. Train", "5. Results"]
    prog_cols = st.columns(len(labels))
    for i, (col, lbl) in enumerate(zip(prog_cols, labels), start=1):
        done    = i < step
        active  = i == step
        prefix  = "✅ " if done else ("▶️ " if active else "○ ")
        colour  = "#1f77b4" if active else ("#2ecc71" if done else "#aaa")
        col.markdown(
            f"<div style='text-align:center;color:{colour};font-weight:{"bold" if active else "normal"};'>"
            f"{prefix}{lbl}</div>",
            unsafe_allow_html=True,
        )
    st.markdown("---")

    # ── Stage tabs ────────────────────────────────────────────────────────────
    tabs = st.tabs(["📥 1. Load", "🧹 2. Clean", "⚙️ 3. Configure", "🚀 4. Train", "📊 5. Results"])

    with tabs[0]: _stage_load()
    with tabs[1]: _stage_clean()
    with tabs[2]: _stage_configure()
    with tabs[3]: _stage_train()
    with tabs[4]: _stage_results()
