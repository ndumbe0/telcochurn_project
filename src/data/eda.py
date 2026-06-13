import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

EDA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "eda_output"
EDA_DIR.mkdir(parents=True, exist_ok=True)


def generate_eda_report(df: pd.DataFrame, target: str = "Churn"):
    logger.info("Generating EDA report...")
    df = df.copy()
    if target in df.columns:
        df[target] = df[target].astype(int)
    figs = {}

    churn_rate = df[target].value_counts(normalize=True) * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["No Churn", "Churn"], y=churn_rate.values,
                         marker_color=["green", "red"], text=[f"{v:.1f}%" for v in churn_rate.values]))
    fig.update_layout(title="Churn Rate", xaxis_title="Churn", yaxis_title="Percentage (%)")
    figs["churn_rate"] = fig

    if "tenure" in df.columns:
        fig = px.histogram(df, x="tenure", color=target, barmode="overlay",
                           title="Tenure Distribution by Churn", opacity=0.7)
        figs["tenure_by_churn"] = fig

    if "Contract" in df.columns:
        contract_churn = df.groupby("Contract")[target].mean().reset_index()
        contract_churn[target] = contract_churn[target] * 100
        fig = px.bar(contract_churn, x="Contract", y=target,
                     title="Churn Rate by Contract Type",
                     labels={target: "Churn Rate (%)"}, color="Contract",
                     text_auto=".1f")
        figs["contract_churn"] = fig

    if "PaymentMethod" in df.columns:
        pay_churn = df.groupby("PaymentMethod")[target].mean().reset_index()
        pay_churn[target] = pay_churn[target] * 100
        fig = px.bar(pay_churn, x="PaymentMethod", y=target,
                     title="Churn Rate by Payment Method",
                     labels={target: "Churn Rate (%)"}, color="PaymentMethod",
                     text_auto=".1f")
        figs["payment_churn"] = fig

    if "MonthlyCharges" in df.columns:
        fig = px.box(df, x=target, y="MonthlyCharges",
                     title="Monthly Charges by Churn",
                     labels={target: "Churn", "MonthlyCharges": "Monthly Charges"})
        figs["monthly_charges_churn"] = fig

    if "tenure" in df.columns and "MonthlyCharges" in df.columns:
        fig = px.scatter(df, x="tenure", y="MonthlyCharges", color=target,
                         title="Tenure vs Monthly Charges by Churn", opacity=0.5)
        figs["tenure_vs_charges"] = fig

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if target in num_cols:
        num_cols.remove(target)
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        figs["correlation"] = fig

    if "InternetService" in df.columns:
        isp_churn = df.groupby("InternetService")[target].mean().reset_index()
        isp_churn[target] = isp_churn[target] * 100
        fig = px.bar(isp_churn, x="InternetService", y=target,
                     title="Churn Rate by Internet Service",
                     labels={target: "Churn Rate (%)"}, color="InternetService",
                     text_auto=".1f")
        figs["internet_churn"] = fig

    if "SeniorCitizen" in df.columns:
        sr_churn = df.groupby("SeniorCitizen")[target].mean().reset_index()
        sr_churn[target] = sr_churn[target] * 100
        sr_churn["SeniorCitizen"] = sr_churn["SeniorCitizen"].map({0: "No", 1: "Yes"})
        fig = px.bar(sr_churn, x="SeniorCitizen", y=target,
                     title="Churn Rate by Senior Citizen Status",
                     labels={target: "Churn Rate (%)"}, color="SeniorCitizen",
                     text_auto=".1f")
        figs["senior_churn"] = fig

    tenure_bins = [0, 12, 24, 48, 72, 100]
    tenure_labels = ["0-12", "13-24", "25-48", "49-72", "73+"]
    if "tenure" in df.columns:
        df["tenure_group"] = pd.cut(df["tenure"], bins=tenure_bins, labels=tenure_labels, right=True)
        tg_churn = df.groupby("tenure_group", observed=True)[target].mean().reset_index()
        tg_churn[target] = tg_churn[target] * 100
        fig = px.line(tg_churn, x="tenure_group", y=target,
                      title="Churn Rate by Tenure Group",
                      labels={target: "Churn Rate (%)", "tenure_group": "Tenure (months)"},
                      markers=True)
        figs["tenure_group_churn"] = fig

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
