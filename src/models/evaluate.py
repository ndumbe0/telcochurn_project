"""Model evaluation charts: ROC curve, confusion matrix, feature importance, model comparison."""
import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    precision_recall_curve, average_precision_score,
)

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


def load_test_data():
    test_path = DATA_DIR / "test.csv"
    if not test_path.exists():
        return None, None
    df = pd.read_csv(test_path)
    y = df["Churn"].values
    X = df.drop(columns=["Churn"])
    return X, y


def get_model_comparison_fig():
    path = MODEL_DIR / "model_comparison.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df.dropna(subset=["roc_auc"])
    df = df.sort_values("roc_auc", ascending=True)

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    colors = px.colors.qualitative.Plotly

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("ROC-AUC by Model", "All Metrics Comparison"))

    # AUC horizontal bar
    for i, row in df.iterrows():
        fig.add_trace(
            go.Bar(x=[row["roc_auc"]], y=[row["name"]], orientation="h",
                   marker_color=colors[i % len(colors)],
                   text=f"{row['roc_auc']:.4f}", textposition="outside",
                   name=row["name"], showlegend=False),
            row=1, col=1
        )

    # Grouped bar for all metrics
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    for i, m in enumerate(metrics):
        fig.add_trace(
            go.Bar(name=metric_labels[i], x=df["name"], y=df[m],
                   text=[f"{v:.3f}" for v in df[m]], textposition="outside"),
            row=1, col=2
        )

    fig.update_layout(
        title="Model Comparison Dashboard",
        height=420,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    fig.update_xaxes(range=[0, 1.05], row=1, col=1)
    return fig


def get_roc_and_pr_fig(pipeline, X_test, y_test):
    try:
        from src.data.load_data import clean_data
        # X_test from processed CSV may have string columns — apply _preprocess_input
        from src.models.predict import _preprocess_input
        X_proc = _preprocess_input(X_test.copy())
        y_proba = pipeline.predict_proba(X_proc)[:, 1]
    except Exception as e:
        logger.warning(f"ROC curve failed: {e}")
        return None, None

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"ROC (AUC={roc_auc:.3f})",
                                 line=dict(color="royalblue", width=2.5)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 name="Random", line=dict(dash="dash", color="gray")))
    fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate", height=380)

    # Precision-Recall
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                                name=f"PR (AP={ap:.3f})",
                                line=dict(color="darkorange", width=2.5)))
    baseline = y_test.mean()
    fig_pr.add_trace(go.Scatter(x=[0, 1], y=[baseline, baseline], mode="lines",
                                name=f"Baseline ({baseline:.2f})",
                                line=dict(dash="dash", color="gray")))
    fig_pr.update_layout(title="Precision-Recall Curve", xaxis_title="Recall",
                         yaxis_title="Precision", height=380)
    return fig_roc, fig_pr


def get_confusion_matrix_fig(pipeline, X_test, y_test):
    try:
        from src.models.predict import _preprocess_input
        X_proc = _preprocess_input(X_test.copy())
        y_pred = pipeline.predict(X_proc)
    except Exception as e:
        logger.warning(f"Confusion matrix failed: {e}")
        return None

    cm = confusion_matrix(y_test, y_pred)
    labels = ["No Churn", "Churn"]
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    text = [[f"{cm[i][j]}<br>({cm_pct[i][j]:.1f}%)" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        text=text, texttemplate="%{text}",
        colorscale="Blues", showscale=False,
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted", yaxis_title="Actual",
        height=380,
    )
    return fig


def get_feature_importance_fig(pipeline, cat_cols, num_cols, top_n=15):
    try:
        model = pipeline.named_steps["classifier"]
        preprocessor = pipeline.named_steps["preprocessor"]
        cat_feat = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)
        feature_names = num_cols + list(cat_feat)

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            return None

        df = pd.DataFrame({"feature": feature_names, "importance": importances})
        df = df.sort_values("importance", ascending=False).head(top_n)
        df = df.sort_values("importance")

        fig = go.Figure(go.Bar(
            x=df["importance"], y=df["feature"],
            orientation="h",
            marker=dict(
                color=df["importance"],
                colorscale="Viridis",
            ),
            text=[f"{v:.4f}" for v in df["importance"]],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Top {top_n} Feature Importances",
            xaxis_title="Importance Score",
            height=500,
        )
        return fig
    except Exception as e:
        logger.warning(f"Feature importance failed: {e}")
        return None


def get_metrics_summary(pipeline, X_test, y_test):
    """Return a dict of test-set metrics for display as KPI cards."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    try:
        from src.models.predict import _preprocess_input
        X_proc = _preprocess_input(X_test.copy())
        y_pred = pipeline.predict(X_proc)
        y_proba = pipeline.predict_proba(X_proc)[:, 1]
        return {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
        }
    except Exception as e:
        logger.warning(f"Metrics summary failed: {e}")
        return {}


def get_threshold_metrics(pipeline, X_test, y_test, threshold: float = 0.5) -> dict:
    """Return classification metrics at a custom decision threshold."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    try:
        from src.models.predict import _preprocess_input
        X_proc = _preprocess_input(X_test.copy())
        y_proba = pipeline.predict_proba(X_proc)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        return {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "Predicted Churn %": y_pred.mean() * 100,
        }
    except Exception as e:
        logger.warning(f"Threshold metrics failed: {e}")
        return {}


def get_threshold_curve_fig(pipeline, X_test, y_test) -> go.Figure:
    """Plot how Precision, Recall, F1 change across threshold values."""
    try:
        from src.models.predict import _preprocess_input
        from sklearn.metrics import precision_score, recall_score, f1_score
        X_proc = _preprocess_input(X_test.copy())
        y_proba = pipeline.predict_proba(X_proc)[:, 1]

        thresholds = np.linspace(0.05, 0.95, 50)
        precisions, recalls, f1s = [], [], []
        for t in thresholds:
            yp = (y_proba >= t).astype(int)
            precisions.append(precision_score(y_test, yp, zero_division=0))
            recalls.append(recall_score(y_test, yp, zero_division=0))
            f1s.append(f1_score(y_test, yp, zero_division=0))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=thresholds, y=precisions, mode="lines", name="Precision",
                                 line=dict(color="#3498db", width=2)))
        fig.add_trace(go.Scatter(x=thresholds, y=recalls, mode="lines", name="Recall",
                                 line=dict(color="#e74c3c", width=2)))
        fig.add_trace(go.Scatter(x=thresholds, y=f1s, mode="lines", name="F1",
                                 line=dict(color="#2ecc71", width=2.5, dash="dot")))
        fig.update_layout(
            title="Precision / Recall / F1 vs Decision Threshold",
            xaxis_title="Threshold",
            yaxis_title="Score",
            yaxis_range=[0, 1.05],
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )
        return fig
    except Exception as e:
        logger.warning(f"Threshold curve failed: {e}")
        return None


def get_calibration_fig(pipeline, X_test, y_test, n_bins: int = 10) -> go.Figure:
    """Reliability (calibration) diagram: fraction of positives vs mean predicted probability."""
    try:
        from sklearn.calibration import calibration_curve
        from src.models.predict import _preprocess_input
        X_proc = _preprocess_input(X_test.copy())
        y_proba = pipeline.predict_proba(X_proc)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=n_bins, strategy="uniform")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=mean_pred, y=frac_pos,
            mode="lines+markers",
            name="Model",
            line=dict(color="royalblue", width=2.5),
            marker=dict(size=8),
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line=dict(dash="dash", color="gray"),
        ))
        fig.update_layout(
            title="Calibration Curve (Reliability Diagram)",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives (Actual Churn Rate)",
            xaxis_range=[0, 1],
            yaxis_range=[0, 1],
            height=380,
        )
        return fig
    except Exception as e:
        logger.warning(f"Calibration curve failed: {e}")
        return None
