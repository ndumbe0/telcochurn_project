import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging
import os
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.load_data import load_raw_data, clean_data, preprocess_features
from src.data.eda import generate_eda_report
from src.models.predict import predict_single, predict_batch, get_shap_values, load_model_info
from src.models.evaluate import (
    load_test_data, get_model_comparison_fig, get_roc_and_pr_fig,
    get_confusion_matrix_fig, get_feature_importance_fig, get_metrics_summary,
    get_threshold_metrics, get_threshold_curve_fig, get_calibration_fig,
)
from src.app.simulator import WHAT_IF_FEATURES, build_what_if_chart
from src.app.recommender import get_recommendations
from src.app.clv import compute_clv_single, add_clv_to_batch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📊",
    layout="wide",
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")


# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_data():
    try:
        df = load_raw_data()
        df_clean = clean_data(df)
        return df, df_clean
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None, None


@st.cache_resource
def get_eda_figures(df):
    return generate_eda_report(df)


@st.cache_resource
def load_model():
    pipeline, model_name, cat_cols, num_cols = load_model_info()
    if pipeline is None:
        st.error("No trained model found. Please run `python -m src.models.train` first.")
        return None, None, None, None
    return pipeline, model_name, cat_cols, num_cols


@st.cache_resource
def load_eval_data():
    return load_test_data()


# ── Sidebar customer form ──────────────────────────────────────────────────────
def build_customer_input(cat_cols, num_cols):
    st.sidebar.header("Customer Details")
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
    total_charges = st.sidebar.number_input(
        "Total Charges ($)", 0.0, 10000.0, monthly_charges * tenure
    )
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.sidebar.selectbox(
        "Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
    )
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox(
        "Multiple Lines", ["Yes", "No", "No phone service"]
    )
    internet_service = st.sidebar.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )
    online_security = st.sidebar.selectbox(
        "Online Security", ["Yes", "No", "No internet service"]
    )
    online_backup = st.sidebar.selectbox(
        "Online Backup", ["Yes", "No", "No internet service"]
    )
    device_protection = st.sidebar.selectbox(
        "Device Protection", ["Yes", "No", "No internet service"]
    )
    tech_support = st.sidebar.selectbox(
        "Tech Support", ["Yes", "No", "No internet service"]
    )
    streaming_tv = st.sidebar.selectbox(
        "Streaming TV", ["Yes", "No", "No internet service"]
    )
    streaming_movies = st.sidebar.selectbox(
        "Streaming Movies", ["Yes", "No", "No internet service"]
    )
    contract = st.sidebar.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.sidebar.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    return {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }


# ── Chart helpers ──────────────────────────────────────────────────────────────
def display_gauge(probability):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Churn Probability (%)"},
            delta={"reference": 50},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "red" if probability > 0.5 else "green"},
                "steps": [
                    {"range": [0, 30], "color": "lightgreen"},
                    {"range": [30, 60], "color": "yellow"},
                    {"range": [60, 100], "color": "salmon"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(height=300)
    return fig


def display_shap_bar(shap_values, feature_names):
    if shap_values is None or feature_names is None:
        st.info("SHAP explanation not available for this model.")
        return
    pairs = sorted(zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True)
    top = pairs[:10]
    names = [p[0] for p in top]
    vals = [p[1] for p in top]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in vals]
    fig = go.Figure(
        go.Bar(
            x=vals, y=names, orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in vals],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Top 10 Feature Contributions (SHAP)",
        xaxis_title="SHAP Value — red pushes toward churn, blue away",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


def display_shap_waterfall(shap_values, feature_names, base_value: float, final_prob: float):
    """Plotly waterfall chart showing cumulative SHAP contributions."""
    if shap_values is None or feature_names is None:
        return
    pairs = sorted(zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True)
    top_n = 8
    top = pairs[:top_n]
    rest_val = sum(v for _, v in pairs[top_n:])

    names = [p[0] for p in top]
    vals = [p[1] for p in top]
    if abs(rest_val) > 0.001:
        names.append(f"Other ({len(pairs) - top_n} features)")
        vals.append(rest_val)

    measure = ["relative"] * len(vals) + ["total"]
    x_labels = names + ["Final Score"]
    y_vals = vals + [None]

    colors = []
    for v in vals:
        colors.append("#e74c3c" if v > 0 else "#2ecc71")
    colors.append("#3498db")

    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=measure,
            x=y_vals,
            y=x_labels,
            base=base_value,
            connector={"line": {"color": "rgba(0,0,0,0.2)"}},
            increasing={"marker": {"color": "#e74c3c"}},
            decreasing={"marker": {"color": "#2ecc71"}},
            totals={"marker": {"color": "#3498db"}},
            textposition="outside",
            text=[f"{v:+.3f}" for v in vals] + [f"{final_prob:.3f}"],
        )
    )
    fig.update_layout(
        title=f"SHAP Waterfall — Base: {base_value:.3f} → Prediction: {final_prob:.3f}",
        xaxis_title="Cumulative SHAP Score",
        height=420,
        waterfallgap=0.3,
    )
    st.plotly_chart(fig, use_container_width=True)


def display_radar_chart(customer_data: dict, df_clean: pd.DataFrame):
    if df_clean is None:
        return
    num_feats = ["tenure", "MonthlyCharges", "TotalCharges"]
    available = [f for f in num_feats if f in df_clean.columns]
    if not available:
        return

    mins = df_clean[available].min()
    maxs = df_clean[available].max()
    rng = (maxs - mins).replace(0, 1)

    churner_avg = df_clean[df_clean["Churn"] == 1][available].mean()
    stayer_avg = df_clean[df_clean["Churn"] == 0][available].mean()
    cust_vals = pd.Series({f: customer_data.get(f, 0) for f in available})
    norm = lambda s: ((s - mins) / rng).clip(0, 1)

    fig = go.Figure()
    for label, series, color in [
        ("This Customer", norm(cust_vals), "#f39c12"),
        ("Avg Churner", norm(churner_avg), "#e74c3c"),
        ("Avg Stayer", norm(stayer_avg), "#2ecc71"),
    ]:
        vals = series.tolist()
        fig.add_trace(
            go.Scatterpolar(
                r=vals + [vals[0]],
                theta=available + [available[0]],
                fill="toself",
                name=label,
                line=dict(color=color),
                opacity=0.7,
            )
        )
    fig.update_layout(
        title="Customer Profile vs. Avg Churner / Stayer",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)


def display_batch_insights(results: pd.DataFrame):
    st.markdown("---")
    st.subheader("📊 Batch Insights")

    total = len(results)
    churners = (results["Prediction"] == "Churn").sum()
    stayers = total - churners
    avg_prob = results["Churn_Probability"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", total)
    col2.metric("Predicted Churn", f"{churners} ({churners/total:.1%})")
    col3.metric("Predicted Stay", stayers)
    col4.metric("Avg Churn Probability", f"{avg_prob:.1f}%")

    # CLV summary metrics
    if "Revenue_at_Risk ($)" in results.columns:
        rar_total = results["Revenue_at_Risk ($)"].sum()
        clv_total = results["CLV_Estimate ($)"].sum()
        c1, c2 = st.columns(2)
        c1.metric("💸 Total Revenue at Risk", f"${rar_total:,.0f}")
        c2.metric("💰 Total CLV (retained)", f"${clv_total:,.0f}")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        if "Risk_Level" in results.columns:
            risk_counts = results["Risk_Level"].value_counts().reset_index()
            risk_counts.columns = ["Risk", "Count"]
            color_map = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
            fig = px.pie(
                risk_counts, names="Risk", values="Count", hole=0.45,
                color="Risk", color_discrete_map=color_map,
                title="Customer Risk Distribution",
            )
            fig.update_traces(textinfo="label+percent+value")
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(
            results, x="Churn_Probability", color="Prediction",
            color_discrete_map={"Churn": "#e74c3c", "Stay": "#2ecc71"},
            nbins=25, title="Churn Probability Distribution",
            labels={"Churn_Probability": "Churn Probability (%)"},
        )
        fig.update_layout(barmode="overlay", bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    # Revenue at risk by segment
    if "Contract" in results.columns and "Revenue_at_Risk ($)" in results.columns:
        seg = (
            results.groupby("Contract")["Revenue_at_Risk ($)"]
            .sum()
            .reset_index()
            .sort_values("Revenue_at_Risk ($)", ascending=True)
        )
        fig = go.Figure(
            go.Bar(
                x=seg["Revenue_at_Risk ($)"],
                y=seg["Contract"],
                orientation="h",
                marker=dict(color=seg["Revenue_at_Risk ($)"], colorscale="Reds"),
                text=[f"${v:,.0f}" for v in seg["Revenue_at_Risk ($)"]],
                textposition="outside",
            )
        )
        fig.update_layout(title="Revenue at Risk by Contract Type ($)", height=300)
        st.plotly_chart(fig, use_container_width=True)

    elif "Contract" in results.columns:
        seg = (
            results.groupby("Contract")["Prediction"]
            .apply(lambda x: (x == "Churn").mean() * 100)
            .reset_index()
        )
        seg.columns = ["Contract", "Churn Rate (%)"]
        seg = seg.sort_values("Churn Rate (%)", ascending=True)
        fig = go.Figure(
            go.Bar(
                x=seg["Churn Rate (%)"], y=seg["Contract"], orientation="h",
                marker=dict(color=seg["Churn Rate (%)"], colorscale="RdYlGn_r"),
                text=[f"{v:.1f}%" for v in seg["Churn Rate (%)"]],
                textposition="outside",
            )
        )
        fig.update_layout(title="Churn Rate by Contract Type", height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Top 10 highest risk
    st.subheader("🚨 Top 10 Highest-Risk Customers")
    top_risk = results.sort_values("Churn_Probability", ascending=False).head(10)
    display_cols = [
        c for c in [
            "customerID", "Contract", "tenure", "MonthlyCharges",
            "InternetService", "Churn_Probability", "Risk_Level",
            "Revenue_at_Risk ($)", "CLV_Estimate ($)",
        ]
        if c in top_risk.columns
    ]
    styled = top_risk[display_cols].style.background_gradient(
        subset=["Churn_Probability"] if "Churn_Probability" in display_cols else [],
        cmap="RdYlGn_r",
    )
    st.dataframe(styled, use_container_width=True)


def df_to_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Predictions")
        # Summary sheet
        summary = pd.DataFrame(
            {
                "Metric": [
                    "Total Customers",
                    "Predicted Churn",
                    "Predicted Stay",
                    "Churn Rate (%)",
                    "Avg Churn Probability (%)",
                ],
                "Value": [
                    len(df),
                    (df["Prediction"] == "Churn").sum(),
                    (df["Prediction"] == "Stay").sum(),
                    round((df["Prediction"] == "Churn").mean() * 100, 2),
                    round(df["Churn_Probability"].mean(), 2),
                ],
            }
        )
        summary.to_excel(writer, index=False, sheet_name="Summary")
    return buf.getvalue()


def get_gemini_explanation(shap_values, feature_names, prediction, probability, api_key):
    if not api_key:
        return "Gemini AI assistant is not configured. Set GOOGLE_AI_API_KEY in your environment."
    if shap_values is None:
        return "SHAP values not available for explanation."
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        pairs = sorted(zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True)
        top_5 = pairs[:5]
        feature_details = "; ".join([f"{f}: {v:.4f}" for f, v in top_5])
        prompt = (
            f"A telco customer has a churn probability of {probability*100:.1f}% "
            f"(predicted to {'churn' if prediction == 1 else 'not churn'}). "
            f"The top SHAP feature contributions are: {feature_details}. "
            f"Explain in simple business terms why this customer is likely or unlikely to churn "
            f"and suggest 1-2 retention actions."
        )
        response = client.models.generate_content(model="gemini-2.0-flash-exp", contents=[prompt])
        return response.text
    except Exception as e:
        return f"Gemini AI error: {e}"


# ── Segment Profiler helpers ───────────────────────────────────────────────────
def render_segment_profiler(df_clean: pd.DataFrame):
    st.header("🔍 Segment Profiler")
    st.markdown(
        "Filter the customer base by any combination of attributes and instantly see "
        "churn statistics, revenue exposure, and cohort breakdowns for that segment."
    )

    if df_clean is None or df_clean.empty:
        st.warning("Data not available.")
        return

    with st.expander("⚙️ Segment Filters", expanded=True):
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            contracts = st.multiselect(
                "Contract", df_clean["Contract"].unique().tolist(),
                default=df_clean["Contract"].unique().tolist(),
            )
        with fc2:
            if "InternetService" in df_clean.columns:
                internets = st.multiselect(
                    "Internet Service", df_clean["InternetService"].unique().tolist(),
                    default=df_clean["InternetService"].unique().tolist(),
                )
            else:
                internets = []
        with fc3:
            senior_opt = st.multiselect(
                "Senior Citizen", [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                default=[0, 1],
            )
        with fc4:
            gender_opt = st.multiselect(
                "Gender", df_clean["gender"].unique().tolist() if "gender" in df_clean.columns
                else ["Male", "Female"],
                default=df_clean["gender"].unique().tolist() if "gender" in df_clean.columns
                else ["Male", "Female"],
            )

    mask = pd.Series([True] * len(df_clean), index=df_clean.index)
    if contracts:
        mask &= df_clean["Contract"].isin(contracts)
    if internets and "InternetService" in df_clean.columns:
        mask &= df_clean["InternetService"].isin(internets)
    if senior_opt:
        mask &= df_clean["SeniorCitizen"].isin(senior_opt)
    if gender_opt and "gender" in df_clean.columns:
        mask &= df_clean["gender"].isin(gender_opt)

    seg = df_clean[mask]

    if seg.empty:
        st.warning("No customers match the selected filters.")
        return

    total = len(seg)
    churners = seg["Churn"].sum() if "Churn" in seg.columns else 0
    churn_rate = churners / total if total > 0 else 0
    avg_monthly = seg["MonthlyCharges"].mean() if "MonthlyCharges" in seg.columns else 0
    avg_tenure = seg["tenure"].mean() if "tenure" in seg.columns else 0

    st.markdown(f"**Segment size:** {total:,} customers ({total/len(df_clean):.1%} of dataset)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Customers in Segment", f"{total:,}")
    m2.metric("Churn Rate", f"{churn_rate:.1%}")
    m3.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")
    m4.metric("Avg Tenure", f"{avg_tenure:.1f} mo")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        # Churn rate by Contract within segment
        if "Contract" in seg.columns and "Churn" in seg.columns:
            grp = (
                seg.groupby("Contract")["Churn"]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": "Churn Rate", "count": "Count"})
            )
            grp["Churn Rate %"] = grp["Churn Rate"] * 100
            fig = px.bar(
                grp, x="Contract", y="Churn Rate %",
                color="Churn Rate %", color_continuous_scale="RdYlGn_r",
                text=[f"{v:.1f}%" for v in grp["Churn Rate %"]],
                title="Churn Rate by Contract (Segment)",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Churn rate by Internet Service
        if "InternetService" in seg.columns and "Churn" in seg.columns:
            grp2 = (
                seg.groupby("InternetService")["Churn"]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": "Churn Rate", "count": "Count"})
            )
            grp2["Churn Rate %"] = grp2["Churn Rate"] * 100
            fig2 = px.bar(
                grp2, x="InternetService", y="Churn Rate %",
                color="Churn Rate %", color_continuous_scale="RdYlGn_r",
                text=[f"{v:.1f}%" for v in grp2["Churn Rate %"]],
                title="Churn Rate by Internet Service (Segment)",
            )
            fig2.update_traces(textposition="outside")
            fig2.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("📅 Cohort Analysis — Churn by Tenure Band")

    if "tenure" in seg.columns and "Churn" in seg.columns:
        seg = seg.copy()
        seg["Tenure Band"] = pd.cut(
            seg["tenure"],
            bins=[0, 12, 24, 36, 48, 60, 72],
            labels=["0–12 mo", "12–24 mo", "24–36 mo", "36–48 mo", "48–60 mo", "60–72 mo"],
            include_lowest=True,
        )
        cohort = (
            seg.groupby("Tenure Band", observed=True)
            .agg(
                Count=("Churn", "count"),
                Churn_Rate=("Churn", "mean"),
                Avg_Monthly=("MonthlyCharges", "mean"),
            )
            .reset_index()
        )
        cohort["Churn Rate %"] = cohort["Churn_Rate"] * 100

        fig_coh = go.Figure()
        fig_coh.add_trace(
            go.Bar(
                x=cohort["Tenure Band"].astype(str),
                y=cohort["Churn Rate %"],
                name="Churn Rate %",
                marker=dict(color=cohort["Churn Rate %"], colorscale="RdYlGn_r"),
                text=[f"{v:.1f}%" for v in cohort["Churn Rate %"]],
                textposition="outside",
                yaxis="y1",
            )
        )
        fig_coh.add_trace(
            go.Scatter(
                x=cohort["Tenure Band"].astype(str),
                y=cohort["Count"],
                mode="lines+markers",
                name="Customer Count",
                line=dict(color="#3498db", width=2),
                yaxis="y2",
            )
        )
        fig_coh.update_layout(
            title="Churn Rate & Customer Count by Tenure Cohort",
            yaxis=dict(title="Churn Rate (%)", range=[0, max(cohort["Churn Rate %"]) * 1.3]),
            yaxis2=dict(title="Customer Count", overlaying="y", side="right"),
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )
        st.plotly_chart(fig_coh, use_container_width=True)

    st.markdown("---")
    with st.expander("📋 Full Segment Data"):
        st.dataframe(seg.reset_index(drop=True), use_container_width=True)


# ── What-if & Interventions tab ────────────────────────────────────────────────
def render_what_if(customer_data: dict, last_result: dict):
    st.header("🔧 What-if Simulator")
    st.markdown(
        "Adjust individual customer attributes below and instantly see how each change "
        "shifts the churn probability — without affecting the sidebar prediction."
    )

    baseline_prob = last_result.get("churn_probability", 0.5)

    b1, b2 = st.columns([1, 2])
    with b1:
        st.metric("Baseline Churn Probability", f"{baseline_prob:.1%}")
        risk = "🔴 High" if baseline_prob > 0.6 else ("🟡 Medium" if baseline_prob > 0.3 else "🟢 Low")
        st.metric("Baseline Risk", risk)

    st.markdown("---")
    st.subheader("Scenario Builder")
    st.markdown("Each row below is one what-if scenario. All scenarios compare against the baseline simultaneously.")

    scenarios = []
    col_a, col_b, col_c, col_d = st.columns(4)

    # Scenario widgets
    scenario_configs = [
        ("Contract Type", "Contract", WHAT_IF_FEATURES["Contract"], col_a),
        ("Monthly Charges ($)", "MonthlyCharges", WHAT_IF_FEATURES["MonthlyCharges"], col_b),
        ("Tech Support", "TechSupport", WHAT_IF_FEATURES["TechSupport"], col_c),
        ("Payment Method", "PaymentMethod", WHAT_IF_FEATURES["PaymentMethod"], col_d),
    ]

    modified_inputs = {}
    for label, key, feat, col in scenario_configs:
        with col:
            st.markdown(f"**{label}**")
            current_val = customer_data.get(key)
            if feat["type"] == "select":
                try:
                    idx = feat["options"].index(str(current_val)) if current_val in feat["options"] else 0
                except ValueError:
                    idx = 0
                new_val = st.selectbox(
                    label,
                    feat["options"],
                    index=idx,
                    key=f"wif_{key}",
                    label_visibility="collapsed",
                )
            else:
                new_val = st.slider(
                    label,
                    feat["min"], feat["max"],
                    float(current_val) if current_val is not None else feat["min"],
                    step=feat.get("step", 1),
                    key=f"wif_{key}",
                    label_visibility="collapsed",
                )
            modified_inputs[key] = new_val

    # Extra scenarios row
    col_e, col_f, col_g, col_h = st.columns(4)
    extra_configs = [
        ("Online Security", "OnlineSecurity", WHAT_IF_FEATURES["OnlineSecurity"], col_e),
        ("Internet Service", "InternetService", WHAT_IF_FEATURES["InternetService"], col_f),
        ("Tenure (months)", "tenure", WHAT_IF_FEATURES["tenure"], col_g),
    ]
    for label, key, feat, col in extra_configs:
        with col:
            st.markdown(f"**{label}**")
            current_val = customer_data.get(key)
            if feat["type"] == "select":
                try:
                    idx = feat["options"].index(str(current_val)) if current_val in feat["options"] else 0
                except ValueError:
                    idx = 0
                new_val = st.selectbox(
                    label,
                    feat["options"],
                    index=idx,
                    key=f"wif_{key}",
                    label_visibility="collapsed",
                )
            else:
                new_val = st.slider(
                    label,
                    feat["min"], feat["max"],
                    int(current_val) if current_val is not None else feat["min"],
                    step=feat.get("step", 1),
                    key=f"wif_{key}",
                    label_visibility="collapsed",
                )
            modified_inputs[key] = new_val

    # Build per-feature scenarios: one scenario per changed feature
    for key, new_val in modified_inputs.items():
        current_val = customer_data.get(key)
        # Only include if value actually changed
        if str(new_val) != str(current_val):
            mod = customer_data.copy()
            mod[key] = new_val
            try:
                res = predict_single(mod)
                feat_label_map = {v["label"]: k for k, v in WHAT_IF_FEATURES.items()}
                display_label = WHAT_IF_FEATURES.get(key, {}).get("label", key)
                scenarios.append({
                    "label": f"{display_label}: {new_val}",
                    "probability": res.get("churn_probability", baseline_prob),
                    "feature": key,
                    "new_val": new_val,
                })
            except Exception:
                pass

    # Combined scenario (all changes applied at once)
    if modified_inputs:
        combined = customer_data.copy()
        combined.update(modified_inputs)
        try:
            combined_res = predict_single(combined)
            scenarios.append({
                "label": "✨ All Changes Combined",
                "probability": combined_res.get("churn_probability", baseline_prob),
                "feature": "combined",
                "new_val": None,
            })
        except Exception:
            pass

    if scenarios:
        fig = build_what_if_chart(baseline_prob, scenarios)
        st.plotly_chart(fig, use_container_width=True)

        # Delta table
        rows = []
        for s in scenarios:
            delta = s["probability"] - baseline_prob
            rows.append({
                "Scenario": s["label"],
                "Churn Probability": f"{s['probability']:.1%}",
                "Δ vs Baseline": f"{delta:+.1%}",
                "Direction": "⬆️ Higher Risk" if delta > 0.01 else ("⬇️ Lower Risk" if delta < -0.01 else "➡️ No Change"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Change any field above to generate what-if scenarios. Each changed field creates a separate bar.")


def render_interventions(customer_data: dict, last_result: dict):
    st.markdown("---")
    st.header("💡 Retention Intervention Recommender")
    st.markdown(
        "The model evaluates every eligible retention action below and estimates "
        "how much each one would reduce this customer's churn probability."
    )

    baseline_prob = last_result.get("churn_probability", 0.5)
    recs = get_recommendations(customer_data, predict_single)

    if not recs:
        st.success("✅ No high-impact interventions identified. This customer has low churn risk or is already on optimal settings.")
        return

    # Summary bar chart
    rec_labels = [r["title"] for r in recs]
    rec_new = [r["new_prob"] * 100 for r in recs]
    rec_deltas = [r["delta_pct"] for r in recs]
    colors = ["#2ecc71" if d > 5 else ("#f39c12" if d > 0 else "#e74c3c") for d in rec_deltas]

    fig = go.Figure(
        go.Bar(
            x=rec_deltas,
            y=rec_labels,
            orientation="h",
            marker_color=colors,
            text=[f"−{d:.1f} pp → {n:.1f}%" for d, n in zip(rec_deltas, rec_new)],
            textposition="outside",
        )
    )
    fig.add_vline(x=0, line_color="gray", line_dash="dash")
    fig.update_layout(
        title=f"Estimated Churn Probability Reduction per Intervention (Baseline: {baseline_prob:.1%})",
        xaxis_title="Probability Reduction (percentage points)",
        height=max(300, len(recs) * 55),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detailed cards
    effort_colors = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
    category_icon = {"Contract": "📋", "Services": "🔧", "Billing": "💳", "Pricing": "💰"}

    for rec in recs:
        icon = category_icon.get(rec["category"], "📌")
        effort = effort_colors.get(rec["effort"], "⚪")
        delta = rec["delta_pct"]
        new_prob = rec["new_prob"]

        with st.container(border=True):
            h1, h2, h3 = st.columns([3, 1, 1])
            with h1:
                st.markdown(f"**{rec['title']}**")
                st.caption(rec["description"])
            with h2:
                st.metric("New Churn Prob.", f"{new_prob:.1%}", delta=f"{-delta:.1f} pp")
            with h3:
                st.markdown(f"**Effort:** {effort} {rec['effort']}")
                st.markdown(f"**Category:** {icon} {rec['category']}")


# ── Main app ───────────────────────────────────────────────────────────────────
def main():
    st.title("📊 Telco Customer Churn Prediction")
    st.markdown("Predict customer churn probability with ML-powered analysis.")

    df_raw, df_clean = load_data()
    pipeline, model_name, cat_cols, num_cols = load_model()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🔮 Single Prediction",
        "📁 Batch Prediction",
        "🔧 What-if & Interventions",
        "🔍 Segment Profiler",
        "📈 EDA Dashboard",
        "🏆 Model Performance",
        "🤖 AI Assistant",
    ])

    # ── Tab 1: Single Prediction ───────────────────────────────────────────────
    with tab1:
        st.header("Single Customer Prediction")
        st.markdown(f"**Active Model:** `{model_name or 'Not loaded'}`")

        customer_data = build_customer_input(cat_cols, num_cols)

        if st.button("Predict Churn", type="primary", use_container_width=True):
            result = predict_single(customer_data)
            if "error" in result:
                st.error(result["error"])
            else:
                prob = result["churn_probability"]
                risk = "🔴 High" if prob > 0.6 else ("🟡 Medium" if prob > 0.3 else "🟢 Low")

                if result["prediction"] == 1:
                    st.error(f"⚠️ Customer is **likely to churn** (Probability: {prob:.1%})")
                else:
                    st.success(f"✅ Customer is **likely to stay** (Probability: {prob:.1%})")

                m1, m2, m3 = st.columns(3)
                m1.metric("Risk Level", risk)
                m2.metric("Churn Score", f"{prob:.1%}")
                m3.metric("Model", model_name or "—")

                # CLV metrics
                clv_info = compute_clv_single(
                    monthly_charges=float(customer_data.get("MonthlyCharges", 65)),
                    tenure=float(customer_data.get("tenure", 12)),
                    churn_prob=prob,
                    contract=str(customer_data.get("Contract", "Month-to-month")),
                )
                cv1, cv2, cv3 = st.columns(3)
                cv1.metric("💰 CLV Estimate", f"${clv_info['CLV Estimate ($)']:,.0f}")
                cv2.metric("💸 Revenue at Risk", f"${clv_info['Revenue at Risk ($)']:,.0f}")
                cv3.metric("⏳ Est. Remaining Months", clv_info["Est. Remaining Months"])

                st.markdown("---")
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(display_gauge(prob), use_container_width=True)
                with c2:
                    display_radar_chart(customer_data, df_clean)

                # SHAP
                shap_vals, feat_names = get_shap_values(
                    customer_data, pipeline, cat_cols, num_cols
                )

                if shap_vals is not None:
                    view = st.radio(
                        "SHAP View", ["Bar Chart", "Waterfall"],
                        horizontal=True, key="shap_view",
                    )
                    base_val = prob - float(np.sum(shap_vals))
                    if view == "Bar Chart":
                        display_shap_bar(shap_vals, feat_names)
                    else:
                        display_shap_waterfall(shap_vals, feat_names, base_val, prob)

                # Store in session
                st.session_state["last_shap"] = (shap_vals, feat_names, result["prediction"], prob)
                st.session_state["last_customer"] = customer_data
                st.session_state["last_result"] = result

                # Append to prediction history
                if "prediction_history" not in st.session_state:
                    st.session_state["prediction_history"] = []
                history_entry = {
                    "Contract": customer_data.get("Contract"),
                    "Tenure (mo)": customer_data.get("tenure"),
                    "Monthly $": customer_data.get("MonthlyCharges"),
                    "Internet": customer_data.get("InternetService"),
                    "Tech Support": customer_data.get("TechSupport"),
                    "Churn Prob %": round(prob * 100, 1),
                    "Prediction": result["churn_label"],
                    "Risk": risk,
                    "CLV ($)": clv_info["CLV Estimate ($)"],
                    "Revenue at Risk ($)": clv_info["Revenue at Risk ($)"],
                }
                st.session_state["prediction_history"].insert(0, history_entry)
                st.session_state["prediction_history"] = st.session_state["prediction_history"][:20]

        # Prediction history
        history = st.session_state.get("prediction_history", [])
        if history:
            with st.expander(f"📜 Prediction History ({len(history)} records)", expanded=False):
                hist_df = pd.DataFrame(history)
                styled_hist = hist_df.style.background_gradient(
                    subset=["Churn Prob %"], cmap="RdYlGn_r"
                )
                st.dataframe(styled_hist, use_container_width=True, hide_index=True)
                if st.button("🗑️ Clear History"):
                    st.session_state["prediction_history"] = []
                    st.rerun()

    # ── Tab 2: Batch Prediction ────────────────────────────────────────────────
    with tab2:
        st.header("Batch Prediction")
        st.markdown("Upload a CSV file with customer data to predict churn for multiple customers.")

        sample_path = (
            Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "test.csv"
        )
        if sample_path.exists():
            with open(sample_path, "rb") as f:
                st.download_button(
                    "⬇️ Download sample CSV (test set)",
                    data=f,
                    file_name="sample_customers.csv",
                    mime="text/csv",
                )

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df_batch = pd.read_csv(uploaded_file)
            st.markdown(f"**Uploaded:** {len(df_batch):,} rows × {df_batch.shape[1]} columns")
            with st.expander("Preview uploaded data"):
                st.dataframe(df_batch.head(10), use_container_width=True)

            if st.button("Run Batch Prediction", type="primary"):
                with st.spinner("Running predictions…"):
                    try:
                        results = predict_batch(df_batch)
                        results = add_clv_to_batch(results)
                        st.success(f"✅ Predictions complete for {len(results):,} customers!")

                        # Downloads
                        dl1, dl2 = st.columns(2)
                        with dl1:
                            csv = results.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "⬇️ Download CSV",
                                data=csv,
                                file_name="churn_predictions.csv",
                                mime="text/csv",
                            )
                        with dl2:
                            excel_bytes = df_to_excel(results)
                            st.download_button(
                                "⬇️ Download Excel",
                                data=excel_bytes,
                                file_name="churn_predictions.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            )

                        with st.expander("Full results table", expanded=False):
                            st.dataframe(results, use_container_width=True)

                        display_batch_insights(results)

                    except Exception as e:
                        st.error(f"Batch prediction failed: {e}")
                        logger.exception(e)

    # ── Tab 3: What-if & Interventions ────────────────────────────────────────
    with tab3:
        last_customer = st.session_state.get("last_customer")
        last_result = st.session_state.get("last_result")

        if last_customer is None or last_result is None:
            st.info(
                "👈 First run a **Single Prediction** (Tab 1) using the sidebar controls. "
                "The What-if Simulator and Intervention Recommender will then appear here."
            )
        else:
            render_what_if(last_customer, last_result)
            render_interventions(last_customer, last_result)

    # ── Tab 4: Segment Profiler ────────────────────────────────────────────────
    with tab4:
        render_segment_profiler(df_clean)

    # ── Tab 5: EDA Dashboard ──────────────────────────────────────────────────
    with tab5:
        st.header("EDA Dashboard")
        if df_clean is None:
            st.warning("Data not available.")
        else:
            with st.spinner("Building charts…"):
                figs = get_eda_figures(df_clean)

            sections = [
                ("🎯 Churn Overview", ["churn_overview"]),
                ("📅 Tenure Analysis", ["tenure_by_churn", "tenure_group_churn"]),
                ("💰 Revenue & Charges", ["monthly_charges_churn", "revenue_at_risk"]),
                ("📋 Contract & Payment", ["contract_churn", "payment_churn"]),
                ("🌐 Internet & Services", ["internet_churn", "services_heatmap"]),
                ("👤 Demographics", ["senior_churn"]),
                ("🔭 Deep Dives", ["tenure_vs_charges", "sunburst", "correlation"]),
            ]

            for section_title, keys in sections:
                present = [k for k in keys if k in figs]
                if not present:
                    continue
                st.markdown(f"### {section_title}")
                if len(present) == 1:
                    st.plotly_chart(figs[present[0]], use_container_width=True)
                else:
                    cols = st.columns(len(present))
                    for col, key in zip(cols, present):
                        with col:
                            st.plotly_chart(figs[key], use_container_width=True)
                st.markdown("---")

    # ── Tab 6: Model Performance ──────────────────────────────────────────────
    with tab6:
        st.header("Model Performance")
        if pipeline is None:
            st.warning("No model loaded.")
        else:
            X_test, y_test = load_eval_data()

            # KPI scorecards
            if X_test is not None and y_test is not None:
                metrics = get_metrics_summary(pipeline, X_test, y_test)
                if metrics:
                    st.subheader("📐 Test Set Metrics (default threshold = 0.5)")
                    cols = st.columns(len(metrics))
                    for col, (name, val) in zip(cols, metrics.items()):
                        col.metric(name, f"{val:.4f}")

            st.markdown("---")

            # ── Threshold Tuning ──────────────────────────────────────────────
            st.subheader("🎚️ Decision Threshold Tuning")
            st.markdown(
                "Adjust the classification threshold to balance precision (fewer false alarms) "
                "and recall (catching more churners). Default is 0.50."
            )
            threshold = st.slider(
                "Decision Threshold", 0.05, 0.95, 0.50, step=0.01,
                help="Customers with churn probability ≥ threshold are predicted to churn.",
            )

            if X_test is not None and y_test is not None:
                t_metrics = get_threshold_metrics(pipeline, X_test, y_test, threshold)
                if t_metrics:
                    tm_cols = st.columns(len(t_metrics))
                    for col, (name, val) in zip(tm_cols, t_metrics.items()):
                        col.metric(name, f"{val:.4f}" if name != "Predicted Churn %" else f"{val:.1f}%")

                thresh_fig = get_threshold_curve_fig(pipeline, X_test, y_test)
                if thresh_fig:
                    thresh_fig.add_vline(
                        x=threshold, line_dash="dot", line_color="purple",
                        annotation_text=f"Current: {threshold:.2f}",
                        annotation_position="top right",
                    )
                    st.plotly_chart(thresh_fig, use_container_width=True)

            st.markdown("---")

            # Model comparison
            st.subheader("🏅 Model Comparison Leaderboard")
            comp_fig = get_model_comparison_fig()
            if comp_fig:
                st.plotly_chart(comp_fig, use_container_width=True)

            st.markdown("---")

            # Feature importance
            st.subheader("🔑 Feature Importance")
            fi_fig = get_feature_importance_fig(pipeline, cat_cols or [], num_cols or [])
            if fi_fig:
                st.plotly_chart(fi_fig, use_container_width=True)
            else:
                shap_img = (
                    Path(__file__).resolve().parent.parent.parent / "models" / "shap_summary.png"
                )
                if shap_img.exists():
                    st.image(str(shap_img), caption="SHAP Feature Importance Summary")

            st.markdown("---")

            if X_test is not None and y_test is not None:
                c1, c2 = st.columns(2)

                with c1:
                    st.subheader("📉 Confusion Matrix")
                    cm_fig = get_confusion_matrix_fig(pipeline, X_test, y_test)
                    if cm_fig:
                        st.plotly_chart(cm_fig, use_container_width=True)

                with c2:
                    st.subheader("📈 ROC Curve")
                    roc_fig, pr_fig = get_roc_and_pr_fig(pipeline, X_test, y_test)
                    if roc_fig:
                        st.plotly_chart(roc_fig, use_container_width=True)

                if pr_fig:
                    st.subheader("🎯 Precision-Recall Curve")
                    st.plotly_chart(pr_fig, use_container_width=True)

                # ── Calibration Curve ─────────────────────────────────────────
                st.markdown("---")
                st.subheader("📏 Calibration Curve")
                st.markdown(
                    "Shows whether the model's predicted probabilities are reliable. "
                    "A perfectly calibrated model follows the dashed diagonal line."
                )
                cal_fig = get_calibration_fig(pipeline, X_test, y_test)
                if cal_fig:
                    st.plotly_chart(cal_fig, use_container_width=True)

    # ── Tab 7: AI Assistant ───────────────────────────────────────────────────
    with tab7:
        st.header("AI Assistant")
        st.markdown("Ask questions about churn predictions. Powered by Google Gemini.")

        api_key = GOOGLE_AI_API_KEY
        if not api_key:
            st.warning("GOOGLE_AI_API_KEY not set. Enter it below to enable AI features.")
            api_key = st.text_input("Gemini API key (optional)", type="password")

        last_shap = st.session_state.get("last_shap")
        if last_shap and api_key:
            shap_vals, feat_names, pred, prob = last_shap
            if st.button("Why is this customer predicted to churn?"):
                with st.spinner("Analysing with Gemini…"):
                    explanation = get_gemini_explanation(
                        shap_vals, feat_names, pred, prob, api_key
                    )
                    st.markdown("### AI Analysis")
                    st.write(explanation)
        elif not last_shap:
            st.info("Run a single prediction first (Tab 1), then come back here for an AI explanation.")

        question = st.text_input("Or ask any question about churn:")
        if question and api_key:
            with st.spinner("Thinking…"):
                try:
                    from google import genai
                    client = genai.Client(api_key=api_key)
                    response = client.models.generate_content(
                        model="gemini-2.0-flash-exp",
                        contents=[
                            f"Context: This is a telco customer churn prediction system using "
                            f"features like tenure, contract type, monthly charges, payment method, etc. "
                            f"Question: {question}"
                        ],
                    )
                    st.markdown("### Answer")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
