import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

EDA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "eda_output"
EDA_DIR.mkdir(parents=True, exist_ok=True)

CHURN_COLOR = {"0": "#2ecc71", "1": "#e74c3c", 0: "#2ecc71", 1: "#e74c3c",
               "No Churn": "#2ecc71", "Churn": "#e74c3c"}


def generate_eda_report(df: pd.DataFrame, target: str = "Churn"):
    logger.info("Generating EDA report...")
    df = df.copy()
    if target in df.columns:
        df[target] = df[target].astype(int)
    figs = {}

    # ── 1. Churn overview ────────────────────────────────────────────────────
    counts = df[target].value_counts()
    labels = ["No Churn", "Churn"]
    vals = [counts.get(0, 0), counts.get(1, 0)]
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "xy"}]],
                        subplot_titles=("Churn Split", "Churn Count"))
    fig.add_trace(go.Pie(labels=labels, values=vals,
                         marker_colors=["#2ecc71", "#e74c3c"],
                         hole=0.45, textinfo="label+percent"), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=vals, marker_color=["#2ecc71", "#e74c3c"],
                         text=vals, textposition="outside"), row=1, col=2)
    fig.update_layout(title="Customer Churn Overview", height=380, showlegend=False)
    figs["churn_overview"] = fig

    # ── 2. Tenure distribution ───────────────────────────────────────────────
    if "tenure" in df.columns:
        fig = px.histogram(df, x="tenure",
                           color=df[target].map({0: "No Churn", 1: "Churn"}),
                           barmode="overlay", opacity=0.75,
                           color_discrete_map={"No Churn": "#2ecc71", "Churn": "#e74c3c"},
                           title="Tenure Distribution by Churn",
                           labels={"color": "Status"})
        fig.update_layout(height=360)
        figs["tenure_by_churn"] = fig

    # ── 3. Churn rate by tenure band (line + bar) ────────────────────────────
    if "tenure" in df.columns:
        bins = [0, 12, 24, 36, 48, 60, 72, 100]
        labels_b = ["0-12", "13-24", "25-36", "37-48", "49-60", "61-72", "73+"]
        df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels_b, right=True)
        tg = df.groupby("tenure_group", observed=True).agg(
            churn_rate=(target, "mean"),
            count=(target, "count")
        ).reset_index()
        tg["churn_rate"] *= 100

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=tg["tenure_group"], y=tg["count"],
                             name="Customer Count", marker_color="lightsteelblue",
                             opacity=0.6), secondary_y=False)
        fig.add_trace(go.Scatter(x=tg["tenure_group"], y=tg["churn_rate"],
                                 mode="lines+markers", name="Churn Rate (%)",
                                 line=dict(color="#e74c3c", width=2.5),
                                 marker=dict(size=8)), secondary_y=True)
        fig.update_layout(title="Churn Rate & Volume by Tenure Band", height=380)
        fig.update_yaxes(title_text="# Customers", secondary_y=False)
        fig.update_yaxes(title_text="Churn Rate (%)", secondary_y=True)
        figs["tenure_group_churn"] = fig

    # ── 4. Contract type ─────────────────────────────────────────────────────
    if "Contract" in df.columns:
        grp = df.groupby("Contract")[target].agg(["mean", "count"]).reset_index()
        grp["mean"] *= 100
        fig = px.bar(grp, x="Contract", y="mean", color="Contract",
                     text=grp["mean"].map("{:.1f}%".format),
                     title="Churn Rate by Contract Type",
                     labels={"mean": "Churn Rate (%)"},
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=380)
        figs["contract_churn"] = fig

    # ── 5. Payment method ────────────────────────────────────────────────────
    if "PaymentMethod" in df.columns:
        grp = df.groupby("PaymentMethod")[target].mean().reset_index()
        grp[target] *= 100
        grp = grp.sort_values(target, ascending=True)
        fig = go.Figure(go.Bar(
            x=grp[target], y=grp["PaymentMethod"], orientation="h",
            marker=dict(color=grp[target], colorscale="RdYlGn_r"),
            text=[f"{v:.1f}%" for v in grp[target]], textposition="outside",
        ))
        fig.update_layout(title="Churn Rate by Payment Method", height=340,
                          xaxis_title="Churn Rate (%)")
        figs["payment_churn"] = fig

    # ── 6. Monthly charges boxplot ───────────────────────────────────────────
    if "MonthlyCharges" in df.columns:
        df["Churn_Label"] = df[target].map({0: "No Churn", 1: "Churn"})
        fig = px.box(df, x="Churn_Label", y="MonthlyCharges", color="Churn_Label",
                     color_discrete_map={"No Churn": "#2ecc71", "Churn": "#e74c3c"},
                     points="outliers",
                     title="Monthly Charges Distribution by Churn",
                     labels={"Churn_Label": ""})
        fig.update_layout(showlegend=False, height=360)
        figs["monthly_charges_churn"] = fig

    # ── 7. Services adoption heatmap ─────────────────────────────────────────
    service_cols = [c for c in [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines"
    ] if c in df.columns]
    if service_cols:
        # For each service: churn rate when subscribed vs not
        rows = []
        for sc in service_cols:
            vals_in_col = df[sc].unique()
            for val in vals_in_col:
                if str(val).lower() in ("no internet service", "no phone service"):
                    continue
                mask = df[sc] == val
                cr = df.loc[mask, target].mean() * 100
                rows.append({"Service": sc, "Status": str(val), "Churn Rate": cr,
                              "Count": mask.sum()})
        if rows:
            heat_df = pd.DataFrame(rows)
            pivot = heat_df.pivot(index="Service", columns="Status", values="Churn Rate")
            fig = px.imshow(pivot, text_auto=".1f", aspect="auto",
                            color_continuous_scale="RdYlGn_r",
                            title="Churn Rate (%) by Service Subscription",
                            labels={"color": "Churn %"})
            fig.update_layout(height=400)
            figs["services_heatmap"] = fig

    # ── 8. Internet service ──────────────────────────────────────────────────
    if "InternetService" in df.columns:
        grp = df.groupby("InternetService")[target].agg(["mean", "count"]).reset_index()
        grp["mean"] *= 100
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "xy"}]],
                            subplot_titles=("Customer Mix", "Churn Rate (%)"))
        fig.add_trace(go.Pie(labels=grp["InternetService"], values=grp["count"],
                             hole=0.4, textinfo="label+percent"), row=1, col=1)
        fig.add_trace(go.Bar(x=grp["InternetService"], y=grp["mean"],
                             marker_color=px.colors.qualitative.Set1,
                             text=[f"{v:.1f}%" for v in grp["mean"]],
                             textposition="outside"), row=1, col=2)
        fig.update_layout(title="Internet Service: Mix & Churn Rate", height=380,
                          showlegend=False)
        figs["internet_churn"] = fig

    # ── 9. Senior citizen ────────────────────────────────────────────────────
    if "SeniorCitizen" in df.columns:
        grp = df.groupby("SeniorCitizen")[target].agg(["mean", "count"]).reset_index()
        grp["mean"] *= 100
        grp["label"] = grp["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior"})
        fig = px.bar(grp, x="label", y="mean", color="label",
                     text=grp["mean"].map("{:.1f}%".format),
                     title="Churn Rate: Senior vs Non-Senior Citizens",
                     labels={"mean": "Churn Rate (%)", "label": ""},
                     color_discrete_sequence=["#3498db", "#e67e22"])
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=360)
        figs["senior_churn"] = fig

    # ── 10. Tenure vs Monthly Charges scatter ────────────────────────────────
    if "tenure" in df.columns and "MonthlyCharges" in df.columns:
        df["Churn_Label"] = df[target].map({0: "No Churn", 1: "Churn"})
        fig = px.scatter(df, x="tenure", y="MonthlyCharges",
                         color="Churn_Label",
                         color_discrete_map={"No Churn": "#2ecc71", "Churn": "#e74c3c"},
                         opacity=0.45, size_max=6,
                         title="Tenure vs Monthly Charges",
                         labels={"Churn_Label": "Status"})
        fig.update_layout(height=400)
        figs["tenure_vs_charges"] = fig

    # ── 11. Revenue at risk ──────────────────────────────────────────────────
    if "MonthlyCharges" in df.columns:
        churned = df[df[target] == 1]["MonthlyCharges"].sum()
        retained = df[df[target] == 0]["MonthlyCharges"].sum()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Retained Revenue", "At-Risk Revenue"],
            y=[retained, churned],
            marker_color=["#2ecc71", "#e74c3c"],
            text=[f"${retained:,.0f}", f"${churned:,.0f}"],
            textposition="outside",
        ))
        pct = churned / (churned + retained) * 100
        fig.update_layout(
            title=f"Monthly Revenue at Risk ({pct:.1f}% from churned customers)",
            yaxis_title="Monthly Charges ($)", height=360,
        )
        figs["revenue_at_risk"] = fig

    # ── 12. Sunburst: Contract → Internet → Churn ────────────────────────────
    if "Contract" in df.columns and "InternetService" in df.columns:
        df["Churn_Label"] = df[target].map({0: "No Churn", 1: "Churn"})
        grp = df.groupby(["Contract", "InternetService", "Churn_Label"]).size().reset_index(name="count")
        fig = px.sunburst(grp, path=["Contract", "InternetService", "Churn_Label"],
                          values="count",
                          color="Churn_Label",
                          color_discrete_map={"No Churn": "#2ecc71", "Churn": "#e74c3c"},
                          title="Churn Breakdown: Contract → Internet Service → Outcome")
        fig.update_layout(height=500)
        figs["sunburst"] = fig

    # ── 13. Numeric correlation heatmap ─────────────────────────────────────
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    drop = [target, "tenure_group"] if "tenure_group" in df.columns else [target]
    num_cols = [c for c in num_cols if c not in drop]
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Feature Correlation Heatmap")
        fig.update_layout(height=420)
        figs["correlation"] = fig

    logger.info(f"Generated {len(figs)} EDA figures")
    return figs


def plot_feature_distributions(df: pd.DataFrame, target: str = "Churn"):
    df = df.copy()
    if target in df.columns:
        df[target] = df[target].astype(int)
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if target in num_cols:
        num_cols.remove(target)
    num_cols = [c for c in num_cols if df[c].nunique() > 2][:6]
    if not num_cols:
        return {}
    figs = {}
    for col in num_cols:
        fig = px.histogram(df, x=col, color=target, barmode="overlay",
                           title=f"{col} Distribution by Churn", opacity=0.7)
        figs[f"dist_{col}"] = fig
    return figs
