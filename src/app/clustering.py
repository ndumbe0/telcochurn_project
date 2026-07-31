"""Customer clustering using K-Means with PCA visualization and persona labelling."""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)

# Features used for clustering
CLUSTER_NUM_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CLUSTER_CAT_MAP = {
    "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
    "InternetService": {"No": 0, "DSL": 1, "Fiber optic": 2},
    "PaymentMethod": {
        "Mailed check": 0,
        "Bank transfer (automatic)": 1,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
    },
}


def _prepare_cluster_features(df: pd.DataFrame) -> np.ndarray:
    """Extract and scale features for clustering."""
    cols = []
    present_num = [c for c in CLUSTER_NUM_FEATURES if c in df.columns]
    X = df[present_num].copy()
    for cat_col, mapping in CLUSTER_CAT_MAP.items():
        if cat_col in df.columns:
            X[cat_col + "_enc"] = df[cat_col].map(mapping).fillna(0)
    X = X.fillna(0)
    scaler = StandardScaler()
    return scaler.fit_transform(X), X.columns.tolist()


def cluster_customers(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """Run K-Means and return df with Cluster label appended."""
    X_scaled, _ = _prepare_cluster_features(df)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    result = df.copy()
    result["Cluster"] = [f"Cluster {i + 1}" for i in labels]
    return result


def get_silhouette_scores(df: pd.DataFrame, k_range=(2, 7)) -> dict:
    """Return silhouette score for each k to help choose optimal clusters."""
    X_scaled, _ = _prepare_cluster_features(df)
    scores = {}
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        try:
            scores[k] = silhouette_score(X_scaled, labels)
        except Exception:
            scores[k] = 0.0
    return scores


def get_elbow_fig(scores: dict) -> go.Figure:
    """Silhouette score vs k chart to help pick optimal clusters."""
    ks = list(scores.keys())
    vals = list(scores.values())
    best_k = ks[int(np.argmax(vals))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ks, y=vals, mode="lines+markers",
        line=dict(color="royalblue", width=2.5),
        marker=dict(size=9),
        name="Silhouette Score",
    ))
    fig.add_vline(
        x=best_k, line_dash="dot", line_color="green",
        annotation_text=f"Best k={best_k}",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Silhouette Score by Number of Clusters",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Silhouette Score (higher = better)",
        height=320,
    )
    return fig


def get_cluster_scatter_fig(df_clustered: pd.DataFrame) -> go.Figure:
    """2-D PCA scatter plot coloured by cluster, sized by MonthlyCharges."""
    X_scaled, _ = _prepare_cluster_features(df_clustered)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_

    plot_df = df_clustered.copy().reset_index(drop=True)
    plot_df["PC1"] = coords[:, 0]
    plot_df["PC2"] = coords[:, 1]

    churn_label = (
        plot_df["Churn"].map({1: "Churned", 0: "Stayed"})
        if "Churn" in plot_df.columns
        else "Unknown"
    )
    plot_df["Churn_Label"] = churn_label

    hover_cols = ["Cluster", "Churn_Label"]
    for c in ["tenure", "MonthlyCharges", "Contract", "InternetService"]:
        if c in plot_df.columns:
            hover_cols.append(c)

    fig = px.scatter(
        plot_df,
        x="PC1", y="PC2",
        color="Cluster",
        symbol="Churn_Label" if "Churn" in plot_df.columns else None,
        hover_data=hover_cols,
        size="MonthlyCharges" if "MonthlyCharges" in plot_df.columns else None,
        size_max=14,
        title=f"Customer Clusters (PCA — {var[0]:.1%} + {var[1]:.1%} variance explained)",
        labels={"PC1": f"PC1 ({var[0]:.1%})", "PC2": f"PC2 ({var[1]:.1%})"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(height=480)
    return fig


def get_cluster_profile_fig(df_clustered: pd.DataFrame) -> go.Figure:
    """Normalised feature-mean heatmap per cluster (cluster personas)."""
    num_cols = [c for c in CLUSTER_NUM_FEATURES if c in df_clustered.columns]
    if "Churn" in df_clustered.columns:
        num_cols.append("Churn")

    profile = df_clustered.groupby("Cluster")[num_cols].mean()

    # Normalise each column 0–1 for comparability
    norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)
    norm = norm.round(3)

    fig = px.imshow(
        norm.T,
        text_auto=".2f",
        color_continuous_scale="RdYlGn",
        aspect="auto",
        title="Cluster Feature Profile (Normalised Mean — green = high, red = low)",
        labels=dict(x="Cluster", y="Feature", color="Normalised Value"),
    )
    fig.update_layout(height=380)
    return fig


def get_cluster_churn_fig(df_clustered: pd.DataFrame) -> go.Figure:
    """Churn rate and size per cluster."""
    if "Churn" not in df_clustered.columns:
        return None

    grp = (
        df_clustered.groupby("Cluster")
        .agg(Count=("Churn", "count"), Churn_Rate=("Churn", "mean"),
             Avg_Monthly=("MonthlyCharges", "mean") if "MonthlyCharges" in df_clustered.columns else ("Churn", "sum"))
        .reset_index()
    )
    grp["Churn_Pct"] = grp["Churn_Rate"] * 100

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Churn Rate %", "Cluster Size"))

    fig.add_trace(
        go.Bar(
            x=grp["Cluster"], y=grp["Churn_Pct"],
            marker=dict(color=grp["Churn_Pct"], colorscale="RdYlGn_r"),
            text=[f"{v:.1f}%" for v in grp["Churn_Pct"]],
            textposition="outside",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=grp["Cluster"], y=grp["Count"],
            marker_color="steelblue",
            text=grp["Count"],
            textposition="outside",
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.update_layout(title="Cluster Churn Rate & Size", height=380)
    return fig


def get_cluster_summary_table(df_clustered: pd.DataFrame) -> pd.DataFrame:
    """Summary stats per cluster for display as a table."""
    agg: dict = {"Churn": ["count", "mean"]}
    for c in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if c in df_clustered.columns:
            agg[c] = "mean"
    if "Contract" in df_clustered.columns:
        agg["Contract"] = lambda x: x.mode().iloc[0] if not x.empty else "—"
    if "InternetService" in df_clustered.columns:
        agg["InternetService"] = lambda x: x.mode().iloc[0] if not x.empty else "—"

    grp = df_clustered.groupby("Cluster").agg(agg)
    grp.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in grp.columns]
    grp = grp.rename(columns={
        "Churn_count": "Customers",
        "Churn_mean": "Churn Rate",
        "tenure_mean": "Avg Tenure (mo)",
        "MonthlyCharges_mean": "Avg Monthly ($)",
        "TotalCharges_mean": "Avg Total ($)",
        "Contract_<lambda>": "Dominant Contract",
        "InternetService_<lambda>": "Dominant Internet",
    })
    if "Churn Rate" in grp.columns:
        grp["Churn Rate"] = grp["Churn Rate"].map("{:.1%}".format)
    for col in ["Avg Monthly ($)", "Avg Total ($)"]:
        if col in grp.columns:
            grp[col] = grp[col].map("${:,.0f}".format)
    if "Avg Tenure (mo)" in grp.columns:
        grp["Avg Tenure (mo)"] = grp["Avg Tenure (mo)"].map("{:.1f}".format)
    return grp.reset_index()
