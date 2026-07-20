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
    get_shap_dependence_fig, get_pdp_fig,
)
from src.app.simulator import WHAT_IF_FEATURES, build_what_if_chart
from src.app.recommender import get_recommendations
from src.app.clv import compute_clv_single, add_clv_to_batch
from src.app.clustering import (
    cluster_customers, get_silhouette_scores, get_elbow_fig,
    get_cluster_scatter_fig, get_cluster_profile_fig,
    get_cluster_churn_fig, get_cluster_summary_table,
)
from src.app.roi import compute_roi, get_roi_comparison_fig, get_breakeven_fig
from src.app.pdf_report import generate_customer_pdf
from src.app.pipeline_tab import render_data_pipeline

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


@st.cache_data(show_spinner=False)
def cached_silhouette(df_hash: int, _df: pd.DataFrame):
    return get_silhouette_scores(_df)


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
        st.info("SHAP explanation not available.")
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
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=available + [available[0]],
            fill="toself", name=label,
            line=dict(color=color), opacity=0.7,
        ))
    fig.update_layout(
        title="Customer Profile vs. Avg Churner / Stayer",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)


def df_to_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Predictions")
        summary = pd.DataFrame({
            "Metric": ["Total Customers", "Predicted Churn", "Predicted Stay",
                        "Churn Rate (%)", "Avg Churn Probability (%)"],
            "Value": [
                len(df),
                (df["Prediction"] == "Churn").sum(),
                (df["Prediction"] == "Stay").sum(),
                round((df["Prediction"] == "Churn").mean() * 100, 2),
                round(df["Churn_Probability"].mean(), 2),
            ],
        })
        summary.to_excel(writer, index=False, sheet_name="Summary")
    return buf.getvalue()


def display_batch_insights(results: pd.DataFrame):
    st.markdown("---")
    st.subheader("📊 Batch Insights")
    total = len(results)
    churners = (results["Prediction"] == "Churn").sum()
    avg_prob = results["Churn_Probability"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", total)
    col2.metric("Predicted Churn", f"{churners} ({churners/total:.1%})")
    col3.metric("Predicted Stay", total - churners)
    col4.metric("Avg Churn Probability", f"{avg_prob:.1f}%")

    if "Revenue_at_Risk ($)" in results.columns:
        c1, c2 = st.columns(2)
        c1.metric("💸 Total Revenue at Risk", f"${results['Revenue_at_Risk ($)'].sum():,.0f}")
        c2.metric("💰 Total CLV (retained)", f"${results['CLV_Estimate ($)'].sum():,.0f}")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if "Risk_Level" in results.columns:
            risk_counts = results["Risk_Level"].value_counts().reset_index()
            risk_counts.columns = ["Risk", "Count"]
            fig = px.pie(risk_counts, names="Risk", values="Count", hole=0.45,
                         color="Risk",
                         color_discrete_map={"Low": "#1f77b4", "Medium": "#ff7f0e", "High": "#d62728"},
                         title="Customer Risk Distribution")
            fig.update_traces(textinfo="label+percent+value")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(results, x="Churn_Probability", color="Prediction",
                           color_discrete_map={"Churn": "#d62728", "Stay": "#1f77b4"},
                           nbins=25, title="Churn Probability Distribution",
                           labels={"Churn_Probability": "Churn Probability (%)"})
        fig.update_layout(barmode="overlay", bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    if "Contract" in results.columns and "Revenue_at_Risk ($)" in results.columns:
        seg = (results.groupby("Contract")["Revenue_at_Risk ($)"].sum()
               .reset_index().sort_values("Revenue_at_Risk ($)", ascending=True))
        fig = go.Figure(go.Bar(
            x=seg["Revenue_at_Risk ($)"], y=seg["Contract"], orientation="h",
            marker=dict(color=seg["Revenue_at_Risk ($)"], colorscale="Reds"),
            text=[f"${v:,.0f}" for v in seg["Revenue_at_Risk ($)"]],
            textposition="outside",
        ))
        fig.update_layout(title="Revenue at Risk by Contract Type ($)", height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🚨 Top 10 Highest-Risk Customers")
    top_risk = results.sort_values("Churn_Probability", ascending=False).head(10)
    display_cols = [c for c in ["customerID", "Contract", "tenure", "MonthlyCharges",
                                 "InternetService", "Churn_Probability", "Risk_Level",
                                 "Revenue_at_Risk ($)", "CLV_Estimate ($)"]
                    if c in top_risk.columns]
    st.dataframe(
        top_risk[display_cols].style.background_gradient(
            subset=["Churn_Probability"] if "Churn_Probability" in display_cols else [],
            cmap="YlOrRd"),
        use_container_width=True,
    )


def get_gemini_explanation(shap_values, feature_names, prediction, probability, api_key):
    if not api_key:
        return "Gemini AI not configured. Set GOOGLE_AI_API_KEY in your environment."
    if shap_values is None:
        return "SHAP values not available."
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        pairs = sorted(zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True)
        top_5 = pairs[:5]
        feature_details = "; ".join([f"{f}: {v:.4f}" for f, v in top_5])
        prompt = (
            f"A telco customer has a churn probability of {probability*100:.1f}% "
            f"(predicted to {'churn' if prediction == 1 else 'not churn'}). "
            f"Top SHAP features: {feature_details}. "
            f"Explain in simple business terms why this customer is likely or unlikely to churn "
            f"and suggest 1-2 retention actions."
        )
        response = client.models.generate_content(model="gemini-2.0-flash-exp", contents=[prompt])
        return response.text
    except Exception as e:
        return f"Gemini AI error: {e}"


# ── Tab renderers ──────────────────────────────────────────────────────────────
def render_executive_summary(df_clean, pipeline, model_name, cat_cols, num_cols):
    st.header("📊 Executive Summary")
    st.markdown("Real-time overview of the customer base, churn risk exposure, and model health.")

    if df_clean is None:
        st.warning("Data not available.")
        return

    total = len(df_clean)
    churners = int(df_clean["Churn"].sum()) if "Churn" in df_clean.columns else 0
    churn_rate = churners / total if total > 0 else 0
    avg_monthly = df_clean["MonthlyCharges"].mean() if "MonthlyCharges" in df_clean.columns else 0
    avg_tenure = df_clean["tenure"].mean() if "tenure" in df_clean.columns else 0

    # Revenue at risk estimate
    if "MonthlyCharges" in df_clean.columns and "Churn" in df_clean.columns:
        churner_revenue = df_clean[df_clean["Churn"] == 1]["MonthlyCharges"].sum()
        total_revenue = df_clean["MonthlyCharges"].sum()
    else:
        churner_revenue = 0
        total_revenue = 0

    # Model metrics
    X_test, y_test = load_eval_data()
    model_auc = None
    if pipeline is not None and X_test is not None and y_test is not None:
        metrics = get_metrics_summary(pipeline, X_test, y_test)
        model_auc = metrics.get("ROC-AUC")

    # ── KPI cards ─────────────────────────────────────────────────────────────
    st.subheader("📌 Key Metrics")
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Customers", f"{total:,}")
    k2.metric("Churn Rate", f"{churn_rate:.1%}")
    k3.metric("Monthly Revenue at Risk", f"${churner_revenue:,.0f}")
    k4, k5, k6 = st.columns(3)
    k4.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")
    k5.metric("Avg Tenure", f"{avg_tenure:.1f} mo")
    if model_auc:
        k6.metric("Model ROC-AUC", f"{model_auc:.4f}")

    st.markdown("---")

    # ── Overview charts ────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        if "Contract" in df_clean.columns and "Churn" in df_clean.columns:
            grp = (df_clean.groupby("Contract")["Churn"].mean() * 100).reset_index()
            grp.columns = ["Contract", "Churn Rate (%)"]
            fig = px.bar(grp, x="Contract", y="Churn Rate (%)",
                         color="Churn Rate (%)", color_continuous_scale="Oranges",
                         text=[f"{v:.1f}%" for v in grp["Churn Rate (%)"]],
                         title="Churn Rate by Contract")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=320, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "InternetService" in df_clean.columns and "Churn" in df_clean.columns:
            grp2 = (df_clean.groupby("InternetService")["Churn"].mean() * 100).reset_index()
            grp2.columns = ["Internet", "Churn Rate (%)"]
            fig2 = px.bar(grp2, x="Internet", y="Churn Rate (%)",
                          color="Churn Rate (%)", color_continuous_scale="Oranges",
                          text=[f"{v:.1f}%" for v in grp2["Churn Rate (%)"]],
                          title="Churn Rate by Internet Service")
            fig2.update_traces(textposition="outside")
            fig2.update_layout(height=320, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

    with c3:
        if "tenure" in df_clean.columns and "Churn" in df_clean.columns:
            df_t = df_clean.copy()
            df_t["Tenure Band"] = pd.cut(df_t["tenure"],
                bins=[0,12,24,36,48,60,72],
                labels=["0–12","12–24","24–36","36–48","48–60","60–72"],
                include_lowest=True)
            cohort = (df_t.groupby("Tenure Band", observed=True)["Churn"].mean() * 100).reset_index()
            cohort.columns = ["Tenure Band", "Churn Rate (%)"]
            fig3 = px.line(cohort, x="Tenure Band", y="Churn Rate (%)",
                           markers=True, title="Churn Rate by Tenure Cohort",
                           color_discrete_sequence=["#e74c3c"])
            fig3.update_layout(height=320)
            st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # ── Revenue exposure ───────────────────────────────────────────────────────
    if "MonthlyCharges" in df_clean.columns and "Churn" in df_clean.columns:
        st.subheader("💸 Revenue Exposure")
        rev_data = pd.DataFrame({
            "Segment": ["Monthly Revenue (Retained)", "Monthly Revenue at Risk"],
            "Amount": [total_revenue - churner_revenue, churner_revenue],
        })
        fig_rev = px.pie(rev_data, names="Segment", values="Amount", hole=0.5,
                         color="Segment",
                         color_discrete_map={
                             "Monthly Revenue (Retained)": "#1f77b4",
                             "Monthly Revenue at Risk": "#d62728",
                         },
                         title=f"Total Monthly Revenue: ${total_revenue:,.0f}")
        fig_rev.update_traces(textinfo="label+percent+value",
                               texttemplate="%{label}<br>$%{value:,.0f} (%{percent})")
        fig_rev.update_layout(height=350)
        st.plotly_chart(fig_rev, use_container_width=True)

    st.markdown("---")

    # ── Top risk segments ──────────────────────────────────────────────────────
    st.subheader("🚨 Highest-Risk Segments")
    risk_rows = []
    for feat in ["Contract", "InternetService", "PaymentMethod"]:
        if feat in df_clean.columns and "Churn" in df_clean.columns:
            grp = df_clean.groupby(feat)["Churn"].agg(["mean", "count"]).reset_index()
            grp.columns = [feat, "Churn Rate", "Count"]
            top_row = grp.sort_values("Churn Rate", ascending=False).iloc[0]
            risk_rows.append({
                "Feature": feat,
                "Value": top_row[feat],
                "Churn Rate": f"{top_row['Churn Rate']:.1%}",
                "Customers": int(top_row["Count"]),
            })
    if risk_rows:
        st.dataframe(pd.DataFrame(risk_rows), use_container_width=True, hide_index=True)

    # ── Prediction history preview ─────────────────────────────────────────────
    history = st.session_state.get("prediction_history", [])
    if history:
        st.markdown("---")
        st.subheader("🕓 Recent Predictions (this session)")
        hist_df = pd.DataFrame(history[:5])
        st.dataframe(
            hist_df.style.background_gradient(subset=["Churn Prob %"], cmap="YlOrRd"),
            use_container_width=True, hide_index=True,
        )


def render_single_prediction(df_clean, pipeline, model_name, cat_cols, num_cols):
    st.header("🔮 Single Customer Prediction")
    if pipeline is None:
        st.error("⚠️ No trained model found. Please run `python -m src.models.train` first.")
        return
    st.markdown(f"**Active Model:** `{model_name or 'Not loaded'}`")

    customer_data = build_customer_input(cat_cols, num_cols)

    if st.button("Predict Churn", type="primary", use_container_width=True):
        result = predict_single(customer_data)
        if "error" in result:
            st.error(result["error"])
            return

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
        gc1, gc2 = st.columns(2)
        with gc1:
            st.plotly_chart(display_gauge(prob), use_container_width=True)
        with gc2:
            display_radar_chart(customer_data, df_clean)

        # SHAP
        shap_vals, feat_names = get_shap_values(customer_data, pipeline, cat_cols, num_cols)
        if shap_vals is not None:
            view = st.radio("SHAP View", ["Bar Chart", "Waterfall"], horizontal=True, key="shap_view")
            base_val = prob - float(np.sum(shap_vals))
            if view == "Bar Chart":
                display_shap_bar(shap_vals, feat_names)
            else:
                display_shap_waterfall(shap_vals, feat_names, base_val, prob)

        # Recommendations quick preview
        recs = get_recommendations(customer_data, predict_single)

        # PDF download
        if prob > 0:
            pdf_bytes = generate_customer_pdf(
                customer_data=customer_data,
                prediction_result=result,
                shap_values=shap_vals,
                feature_names=feat_names,
                clv_info=clv_info,
                recommendations=recs,
                model_name=model_name or "ML Model",
            )
            st.download_button(
                "📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"churn_report_{customer_data.get('Contract','').replace(' ','')}_{int(prob*100)}pct.pdf",
                mime="application/pdf",
            )

        # Store in session
        st.session_state["last_shap"] = (shap_vals, feat_names, result["prediction"], prob)
        st.session_state["last_customer"] = customer_data
        st.session_state["last_result"] = result
        st.session_state["last_clv"] = clv_info
        st.session_state["last_recs"] = recs

        # Prediction history
        if "prediction_history" not in st.session_state:
            st.session_state["prediction_history"] = []
        st.session_state["prediction_history"].insert(0, {
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
        })
        st.session_state["prediction_history"] = st.session_state["prediction_history"][:20]

    # History expander
    history = st.session_state.get("prediction_history", [])
    if history:
        with st.expander(f"📜 Prediction History ({len(history)} records)", expanded=False):
            hist_df = pd.DataFrame(history)
            st.dataframe(
                hist_df.style.background_gradient(subset=["Churn Prob %"], cmap="YlOrRd"),
                use_container_width=True, hide_index=True,
            )
            if st.button("🗑️ Clear History"):
                st.session_state["prediction_history"] = []
                st.rerun()


def render_batch_prediction():
    st.header("📁 Batch Prediction")
    st.markdown("Upload a CSV file with customer data to predict churn for multiple customers at once.")

    pipeline, model_name, cat_cols, num_cols = load_model()
    if pipeline is None:
        st.error("⚠️ No trained model found. Please run `python -m src.models.train` first.")
        return

    sample_path = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "test.csv"

    # ── Empty-state instructions ───────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("#### 📋 How it works")
        ic1, ic2, ic3 = st.columns(3)
        ic1.markdown("**1️⃣ Download sample**  \nUse our sample CSV to see the required column format.")
        ic2.markdown("**2️⃣ Upload your file**  \nCSV must include tenure, MonthlyCharges, Contract, InternetService and other feature columns.")
        ic3.markdown("**3️⃣ Get predictions**  \nChurn probability, risk level, CLV, and revenue at risk for every customer.")
        if sample_path.exists():
            with open(sample_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Sample CSV", data=f,
                    file_name="sample_customers.csv", mime="text/csv",
                    use_container_width=True,
                )

    uploaded_file = st.file_uploader("Choose a CSV file (max 10 MB)", type="csv")

    if uploaded_file is None:
        return

    # ── File size validation ───────────────────────────────────────────────────
    MAX_MB = 10
    size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    if size_mb > MAX_MB:
        st.error(
            f"❌ File is **{size_mb:.1f} MB** — maximum allowed size is {MAX_MB} MB.  "
            f"Please split your data into smaller batches."
        )
        return

    # ── Parse CSV ──────────────────────────────────────────────────────────────
    try:
        df_batch = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    except Exception as e:
        st.error(f"❌ Could not read CSV file: {e}")
        return

    # ── Column schema validation ───────────────────────────────────────────────
    REQUIRED_COLS = [
        "tenure", "MonthlyCharges", "TotalCharges", "Contract",
        "InternetService", "PaymentMethod",
    ]
    missing_cols = [c for c in REQUIRED_COLS if c not in df_batch.columns]
    if missing_cols:
        st.error(
            f"❌ Missing required columns: **{', '.join(missing_cols)}**  \n"
            f"Download the sample CSV above to see the full expected format."
        )
        return

    st.success(f"✅ File loaded: **{len(df_batch):,} rows × {df_batch.shape[1]} columns** ({size_mb:.2f} MB)")
    with st.expander("Preview uploaded data"):
        st.dataframe(df_batch.head(10), use_container_width=True)

    # ── Caching keyed to file content hash ────────────────────────────────────
    file_hash = hash(uploaded_file.getvalue())
    cache_key = f"batch_results_{file_hash}"

    run_col, _ = st.columns([1, 3])
    if run_col.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
        with st.spinner(f"Running predictions on {len(df_batch):,} customers…"):
            try:
                results = predict_batch(df_batch)
                results = add_clv_to_batch(results)
                st.session_state[cache_key] = results
            except Exception as e:
                st.error(f"❌ Batch prediction failed: {e}")
                logger.exception(e)
                return

    if cache_key not in st.session_state:
        return

    results = st.session_state[cache_key]
    st.success(f"✅ Predictions complete for **{len(results):,} customers**!")

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "⬇️ Download CSV",
            data=results.to_csv(index=False).encode(),
            file_name="churn_predictions.csv", mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "⬇️ Download Excel",
            data=df_to_excel(results),
            file_name="churn_predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    with st.expander("Full results table", expanded=False):
        st.dataframe(results, use_container_width=True)

    display_batch_insights(results)


def render_what_if_and_roi(customer_data: dict, last_result: dict, clv_info: dict, recs: list):
    # ── What-if Simulator ──────────────────────────────────────────────────────
    st.header("🔧 What-if Simulator")
    st.markdown(
        "Adjust individual customer attributes and see how each change shifts the churn probability — "
        "each field generates its own bar, plus an 'All Changes Combined' scenario."
    )

    baseline_prob = last_result.get("churn_probability", 0.5)
    b1, b2 = st.columns(2)
    b1.metric("Baseline Churn Probability", f"{baseline_prob:.1%}")
    b2.metric("Baseline Risk",
              "🔴 High" if baseline_prob > 0.6 else ("🟡 Medium" if baseline_prob > 0.3 else "🟢 Low"))

    st.markdown("---")

    col_a, col_b, col_c, col_d = st.columns(4)
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
            cur = customer_data.get(key)
            if feat["type"] == "select":
                try:
                    idx = feat["options"].index(str(cur)) if cur in feat["options"] else 0
                except ValueError:
                    idx = 0
                new_val = st.selectbox(label, feat["options"], index=idx,
                                       key=f"wif_{key}", label_visibility="collapsed")
            else:
                new_val = st.slider(label, feat["min"], feat["max"],
                                    float(cur) if cur is not None else feat["min"],
                                    step=feat.get("step", 1),
                                    key=f"wif_{key}", label_visibility="collapsed")
            modified_inputs[key] = new_val

    col_e, col_f, col_g, _ = st.columns(4)
    extra_configs = [
        ("Online Security", "OnlineSecurity", WHAT_IF_FEATURES["OnlineSecurity"], col_e),
        ("Internet Service", "InternetService", WHAT_IF_FEATURES["InternetService"], col_f),
        ("Tenure (months)", "tenure", WHAT_IF_FEATURES["tenure"], col_g),
    ]
    for label, key, feat, col in extra_configs:
        with col:
            st.markdown(f"**{label}**")
            cur = customer_data.get(key)
            if feat["type"] == "select":
                try:
                    idx = feat["options"].index(str(cur)) if cur in feat["options"] else 0
                except ValueError:
                    idx = 0
                new_val = st.selectbox(label, feat["options"], index=idx,
                                       key=f"wif_{key}", label_visibility="collapsed")
            else:
                new_val = st.slider(label, feat["min"], feat["max"],
                                    int(cur) if cur is not None else feat["min"],
                                    step=feat.get("step", 1),
                                    key=f"wif_{key}", label_visibility="collapsed")
            modified_inputs[key] = new_val

    scenarios = []
    for key, new_val in modified_inputs.items():
        if str(new_val) != str(customer_data.get(key)):
            mod = customer_data.copy()
            mod[key] = new_val
            try:
                res = predict_single(mod)
                scenarios.append({
                    "label": f"{WHAT_IF_FEATURES.get(key, {}).get('label', key)}: {new_val}",
                    "probability": res.get("churn_probability", baseline_prob),
                    "feature": key, "new_val": new_val,
                })
            except Exception:
                pass

    if modified_inputs:
        combined = customer_data.copy()
        combined.update(modified_inputs)
        try:
            combined_res = predict_single(combined)
            scenarios.append({
                "label": "✨ All Changes Combined",
                "probability": combined_res.get("churn_probability", baseline_prob),
                "feature": "combined", "new_val": None,
            })
        except Exception:
            pass

    if scenarios:
        st.plotly_chart(build_what_if_chart(baseline_prob, scenarios), use_container_width=True)
        rows = [{
            "Scenario": s["label"],
            "Churn Probability": f"{s['probability']:.1%}",
            "Δ vs Baseline": f"{s['probability'] - baseline_prob:+.1%}",
            "Direction": (
                "⬆️ Higher Risk" if s["probability"] - baseline_prob > 0.01
                else ("⬇️ Lower Risk" if s["probability"] - baseline_prob < -0.01 else "➡️ No Change")
            ),
        } for s in scenarios]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Change any field above to generate what-if scenarios.")

    # ── Interventions ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("💡 Retention Intervention Recommender")
    st.markdown(
        "Each intervention is run through the model to estimate its real impact on churn probability."
    )

    if not recs:
        st.success("✅ No high-impact interventions identified — customer is low risk or already optimally configured.")
    else:
        effort_colors = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
        category_icon = {"Contract": "📋", "Services": "🔧", "Billing": "💳", "Pricing": "💰"}

        # Summary chart
        fig = go.Figure(go.Bar(
            x=[r["delta_pct"] for r in recs],
            y=[r["title"] for r in recs],
            orientation="h",
            marker_color=["#1f77b4" if r["delta_pct"] > 5 else "#ff7f0e" for r in recs],
            text=[f"−{r['delta_pct']:.1f}pp → {r['new_prob']:.1%}" for r in recs],
            textposition="outside",
        ))
        fig.add_vline(x=0, line_color="gray", line_dash="dash")
        fig.update_layout(
            title=f"Estimated Churn Probability Reduction (Baseline: {baseline_prob:.1%})",
            xaxis_title="Probability Reduction (pp)",
            height=max(300, len(recs) * 55),
        )
        st.plotly_chart(fig, use_container_width=True)

        for rec in recs:
            icon = category_icon.get(rec["category"], "📌")
            effort_icon = effort_colors.get(rec["effort"], "⚪")
            with st.container(border=True):
                h1, h2, h3 = st.columns([3, 1, 1])
                with h1:
                    st.markdown(f"**{rec['title']}**")
                    st.caption(rec["description"])
                with h2:
                    st.metric("New Prob.", f"{rec['new_prob']:.1%}",
                              delta=f"{-rec['delta_pct']:.1f} pp")
                with h3:
                    st.markdown(f"**Effort:** {effort_icon} {rec['effort']}")
                    st.markdown(f"**Category:** {icon} {rec['category']}")

    # ── ROI Calculator ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("💼 Retention ROI Calculator")
    st.markdown(
        "Enter the cost of each retention action and see whether the revenue saved "
        "justifies the investment — with a break-even timeline."
    )

    if not recs:
        st.info("Run a prediction with high churn probability to see ROI calculations.")
        return

    monthly = float(customer_data.get("MonthlyCharges", 65))
    tenure_val = float(customer_data.get("tenure", 12))
    contract_val = str(customer_data.get("Contract", "Month-to-month"))

    rc1, rc2 = st.columns([1, 2])
    with rc1:
        st.markdown("**Set Intervention Costs**")
        cost_inputs = {}
        for rec in recs[:6]:
            cost_inputs[rec["id"]] = st.number_input(
                f"{rec['title'].split(' ', 1)[1] if ' ' in rec['title'] else rec['title']} ($)",
                min_value=0.0, max_value=5000.0,
                value=50.0 if rec["category"] == "Pricing" else 20.0,
                step=5.0,
                key=f"cost_{rec['id']}",
            )

    with rc2:
        recs_with_roi = []
        for rec in recs[:6]:
            cost = cost_inputs.get(rec["id"], 20.0)
            roi_data = compute_roi(
                baseline_prob=baseline_prob,
                new_prob=rec["new_prob"],
                monthly_charges=monthly,
                tenure=tenure_val,
                contract=contract_val,
                cost_per_customer=cost,
            )
            recs_with_roi.append({
                "title": rec["title"],
                "revenue_saved": roi_data["Revenue Saved ($)"],
                "cost": roi_data["Intervention Cost ($)"],
                "net_roi": roi_data["Net ROI ($)"],
                "roi_pct": roi_data["ROI (%)"],
                "break_even": roi_data["Break-even (months)"],
                **roi_data,
            })

        roi_fig = get_roi_comparison_fig(recs_with_roi)
        if roi_fig:
            st.plotly_chart(roi_fig, use_container_width=True)

    # ROI summary table
    roi_table = pd.DataFrame([{
        "Intervention": r["title"],
        "Revenue Saved": f"${r['revenue_saved']:,.0f}",
        "Cost": f"${r['cost']:,.0f}",
        "Net ROI": f"${r['net_roi']:,.0f}",
        "ROI %": f"{r['roi_pct']:.0f}%",
        "Break-even": f"{r['break_even']} mo" if r["break_even"] != "N/A" else "N/A",
    } for r in recs_with_roi])
    st.dataframe(roi_table, use_container_width=True, hide_index=True)

    # Break-even chart for top ROI intervention
    if recs_with_roi:
        best = max(recs_with_roi, key=lambda x: x["roi_pct"])
        best_rec = next((r for r in recs if r["title"] == best["title"]), None)
        if best_rec:
            st.markdown(f"**Break-even Analysis for Top Intervention: {best['title']}**")
            be_fig = get_breakeven_fig(
                baseline_prob=baseline_prob,
                new_prob=best_rec["new_prob"],
                monthly_charges=monthly,
                cost=best["cost"],
            )
            st.plotly_chart(be_fig, use_container_width=True)


def render_segment_profiler(df_clean: pd.DataFrame):
    st.header("🔍 Segment Profiler")
    st.markdown(
        "Filter the customer base by any combination of attributes and instantly see "
        "churn statistics, revenue exposure, and cohort breakdowns."
    )
    if df_clean is None or df_clean.empty:
        st.warning("Data not available.")
        return

    with st.expander("⚙️ Segment Filters", expanded=True):
        fc1, fc2, fc3, fc4 = st.columns(4)
        contracts = fc1.multiselect("Contract", df_clean["Contract"].unique().tolist(),
                                    default=df_clean["Contract"].unique().tolist())
        internets = (fc2.multiselect("Internet Service", df_clean["InternetService"].unique().tolist(),
                                     default=df_clean["InternetService"].unique().tolist())
                     if "InternetService" in df_clean.columns else [])
        senior_opt = fc3.multiselect("Senior Citizen", [0, 1],
                                     format_func=lambda x: "Yes" if x == 1 else "No", default=[0, 1])
        gender_opt = (fc4.multiselect("Gender", df_clean["gender"].unique().tolist(),
                                      default=df_clean["gender"].unique().tolist())
                      if "gender" in df_clean.columns else [])

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
    churn_rate = seg["Churn"].mean() if "Churn" in seg.columns else 0
    st.markdown(f"**Segment size:** {total:,} customers ({total/len(df_clean):.1%} of dataset)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Customers", f"{total:,}")
    m2.metric("Churn Rate", f"{churn_rate:.1%}")
    m3.metric("Avg Monthly $", f"${seg['MonthlyCharges'].mean():.2f}" if "MonthlyCharges" in seg.columns else "—")
    m4.metric("Avg Tenure", f"{seg['tenure'].mean():.1f} mo" if "tenure" in seg.columns else "—")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if "Contract" in seg.columns and "Churn" in seg.columns:
            grp = seg.groupby("Contract")["Churn"].mean().reset_index()
            grp["Churn Rate %"] = grp["Churn"] * 100
            fig = px.bar(grp, x="Contract", y="Churn Rate %",
                         color="Churn Rate %", color_continuous_scale="Oranges",
                         text=[f"{v:.1f}%" for v in grp["Churn Rate %"]],
                         title="Churn Rate by Contract (Segment)")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if "InternetService" in seg.columns and "Churn" in seg.columns:
            grp2 = seg.groupby("InternetService")["Churn"].mean().reset_index()
            grp2["Churn Rate %"] = grp2["Churn"] * 100
            fig2 = px.bar(grp2, x="InternetService", y="Churn Rate %",
                          color="Churn Rate %", color_continuous_scale="Oranges",
                          text=[f"{v:.1f}%" for v in grp2["Churn Rate %"]],
                          title="Churn Rate by Internet Service (Segment)")
            fig2.update_traces(textposition="outside")
            fig2.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("📅 Cohort Analysis — Churn by Tenure Band")
    if "tenure" in seg.columns and "Churn" in seg.columns:
        seg = seg.copy()
        seg["Tenure Band"] = pd.cut(seg["tenure"], bins=[0,12,24,36,48,60,72],
                                    labels=["0–12","12–24","24–36","36–48","48–60","60–72"],
                                    include_lowest=True)
        cohort = (seg.groupby("Tenure Band", observed=True)
                  .agg(Count=("Churn","count"), Churn_Rate=("Churn","mean"),
                       Avg_Monthly=("MonthlyCharges","mean"))
                  .reset_index())
        cohort["Churn Rate %"] = cohort["Churn_Rate"] * 100

        fig_coh = go.Figure()
        fig_coh.add_trace(go.Bar(
            x=cohort["Tenure Band"].astype(str), y=cohort["Churn Rate %"],
            name="Churn Rate %",
            marker=dict(color=cohort["Churn Rate %"], colorscale="Oranges"),
            text=[f"{v:.1f}%" for v in cohort["Churn Rate %"]], textposition="outside", yaxis="y1",
        ))
        fig_coh.add_trace(go.Scatter(
            x=cohort["Tenure Band"].astype(str), y=cohort["Count"],
            mode="lines+markers", name="Customer Count",
            line=dict(color="#3498db", width=2), yaxis="y2",
        ))
        fig_coh.update_layout(
            title="Churn Rate & Customer Count by Tenure Cohort",
            yaxis=dict(title="Churn Rate (%)"),
            yaxis2=dict(title="Customer Count", overlaying="y", side="right"),
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )
        st.plotly_chart(fig_coh, use_container_width=True)

    with st.expander("📋 Full Segment Data"):
        st.dataframe(seg.reset_index(drop=True), use_container_width=True)


def render_clusters(df_clean: pd.DataFrame):
    st.header("👥 Customer Clusters")
    st.markdown(
        "Unsupervised K-Means clustering groups customers into personas based on "
        "usage patterns. The silhouette score helps you pick the right number of clusters."
    )
    if df_clean is None or df_clean.empty:
        st.warning("Data not available.")
        return

    col_ctrl, col_info = st.columns([1, 3])
    with col_ctrl:
        n_clusters = st.slider("Number of Clusters (k)", 2, 7, 4, key="k_clusters")

    with col_info:
        with st.spinner("Computing silhouette scores…"):
            sil_scores = get_silhouette_scores(df_clean)
        best_k = max(sil_scores, key=sil_scores.get)
        st.info(f"📈 Optimal k by silhouette score: **{best_k}** "
                f"(score: {sil_scores[best_k]:.3f}). "
                f"Current selection: **k={n_clusters}**.")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(get_elbow_fig(sil_scores), use_container_width=True)

    with st.spinner(f"Clustering customers into {n_clusters} groups…"):
        df_clustered = cluster_customers(df_clean, n_clusters=n_clusters)

    with c2:
        churn_fig = get_cluster_churn_fig(df_clustered)
        if churn_fig:
            st.plotly_chart(churn_fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🗺️ Customer Map (PCA 2D)")
    st.markdown("Each dot is a customer, coloured by cluster. Size = Monthly Charges. "
                "Symbol = Churn status.")
    scatter_fig = get_cluster_scatter_fig(df_clustered)
    st.plotly_chart(scatter_fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🔥 Cluster Feature Heatmap (Personas)")
    st.markdown("Normalised feature averages per cluster — green = high value, red = low.")
    profile_fig = get_cluster_profile_fig(df_clustered)
    st.plotly_chart(profile_fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Cluster Summary Table")
    summary = get_cluster_summary_table(df_clustered)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    with st.expander("🔎 Browse Customers by Cluster"):
        chosen_cluster = st.selectbox("Select Cluster", sorted(df_clustered["Cluster"].unique()))
        cluster_df = df_clustered[df_clustered["Cluster"] == chosen_cluster].reset_index(drop=True)
        st.markdown(f"**{len(cluster_df):,} customers** in {chosen_cluster}")
        st.dataframe(cluster_df, use_container_width=True)


def render_model_performance(pipeline, model_name, cat_cols, num_cols):
    st.header("Model Performance")
    if pipeline is None:
        st.warning("No model loaded.")
        return

    X_test, y_test = load_eval_data()

    # KPI cards
    if X_test is not None and y_test is not None:
        metrics = get_metrics_summary(pipeline, X_test, y_test)
        if metrics:
            st.subheader("📐 Test Set Metrics (default threshold = 0.5)")
            cols = st.columns(len(metrics))
            for col, (name, val) in zip(cols, metrics.items()):
                col.metric(name, f"{val:.4f}")

    st.markdown("---")

    # Threshold tuning
    st.subheader("🎚️ Decision Threshold Tuning")
    threshold = st.slider("Decision Threshold", 0.05, 0.95, 0.50, step=0.01)
    if X_test is not None and y_test is not None:
        with st.spinner("Computing threshold metrics…"):
            t_metrics = get_threshold_metrics(pipeline, X_test, y_test, threshold)
        if t_metrics:
            tm_cols = st.columns(len(t_metrics))
            for col, (name, val) in zip(tm_cols, t_metrics.items()):
                col.metric(name, f"{val:.4f}" if name != "Predicted Churn %" else f"{val:.1f}%")
        with st.spinner("Building threshold curve…"):
            thresh_fig = get_threshold_curve_fig(pipeline, X_test, y_test)
        if thresh_fig:
            thresh_fig.add_vline(x=threshold, line_dash="dot", line_color="purple",
                                  annotation_text=f"Current: {threshold:.2f}",
                                  annotation_position="top right")
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

        # Calibration
        st.markdown("---")
        st.subheader("📏 Calibration Curve")
        cal_fig = get_calibration_fig(pipeline, X_test, y_test)
        if cal_fig:
            st.plotly_chart(cal_fig, use_container_width=True)

    st.markdown("---")

    # SHAP Dependence Plots
    st.subheader("🔗 SHAP Dependence Plot")
    st.markdown(
        "Shows how the SHAP value (feature impact) for a chosen feature varies with its raw value. "
        "Colour indicates the feature with the strongest interaction effect."
    )
    if X_test is not None and cat_cols and num_cols:
        all_feats = num_cols + cat_cols
        dep_feature = st.selectbox("Select feature for SHAP dependence", all_feats, key="dep_feat")
        with st.spinner("Computing SHAP dependence (may take a few seconds)…"):
            dep_fig = get_shap_dependence_fig(pipeline, X_test, dep_feature, cat_cols, num_cols)
        if dep_fig:
            st.plotly_chart(dep_fig, use_container_width=True)
        else:
            st.info("SHAP dependence plot not available for this feature / model combination.")
    else:
        st.info("Load model and test data to enable SHAP dependence plots.")

    st.markdown("---")

    # Partial Dependence Plot
    st.subheader("📐 Partial Dependence Plot (PDP)")
    st.markdown(
        "Shows the average predicted churn probability as a single feature varies, "
        "holding all other features at their observed values."
    )
    if X_test is not None and cat_cols and num_cols:
        pdp_feature = st.selectbox("Select feature for PDP", num_cols, key="pdp_feat")
        with st.spinner("Computing partial dependence…"):
            pdp_fig = get_pdp_fig(pipeline, X_test, pdp_feature, cat_cols, num_cols)
        if pdp_fig:
            st.plotly_chart(pdp_fig, use_container_width=True)
        else:
            st.info("PDP not available for this feature / model combination.")


# ── Main app ───────────────────────────────────────────────────────────────────
def main():
    st.title("📊 Telco Customer Churn Prediction")
    st.markdown("Predict customer churn probability with ML-powered analysis.")

    df_raw, df_clean = load_data()
    pipeline, model_name, cat_cols, num_cols = load_model()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📊 Executive Summary",
        "🔮 Single Prediction",
        "📁 Batch Prediction",
        "🔧 What-if & ROI",
        "🔍 Segment Profiler",
        "👥 Customer Clusters",
        "📈 EDA Dashboard",
        "🏆 Model Performance",
        "🤖 AI Assistant",
        "🔄 Data Pipeline",
    ])

    with tab1:
        render_executive_summary(df_clean, pipeline, model_name, cat_cols, num_cols)

    with tab2:
        render_single_prediction(df_clean, pipeline, model_name, cat_cols, num_cols)

    with tab3:
        render_batch_prediction()

    with tab4:
        last_customer = st.session_state.get("last_customer")
        last_result = st.session_state.get("last_result")
        clv_info = st.session_state.get("last_clv", {})
        recs = st.session_state.get("last_recs", [])
        if last_customer is None or last_result is None:
            st.info("👈 Run a **Single Prediction** first (Tab 2). "
                    "The What-if Simulator, Intervention Recommender, and ROI Calculator will appear here.")
        else:
            render_what_if_and_roi(last_customer, last_result, clv_info, recs)

    with tab5:
        render_segment_profiler(df_clean)

    with tab6:
        render_clusters(df_clean)

    with tab7:
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

    with tab8:
        render_model_performance(pipeline, model_name, cat_cols, num_cols)

    with tab10:
        render_data_pipeline()

    with tab9:
        st.header("🤖 AI Assistant")
        st.markdown("Chat with Gemini AI about churn predictions, customer insights, and retention strategies.")

        # ── API key setup ──────────────────────────────────────────────────────
        api_key = GOOGLE_AI_API_KEY or st.session_state.get("_session_api_key", "")
        if not GOOGLE_AI_API_KEY:
            with st.expander("🔑 Configure API Key", expanded=not bool(api_key)):
                st.warning(
                    "**GOOGLE_AI_API_KEY** not found in environment. "
                    "Add it to Replit Secrets (🔒 in the sidebar) for permanent access."
                )
                entered_key = st.text_input(
                    "Or enter your Gemini API key for this session:",
                    type="password", key="api_key_input",
                    placeholder="AIza...",
                )
                if entered_key:
                    st.session_state["_session_api_key"] = entered_key
                    api_key = entered_key
                    st.success("✅ API key saved for this session.")

        # ── Session chat history ───────────────────────────────────────────────
        if "ai_messages" not in st.session_state:
            st.session_state["ai_messages"] = []

        # ── Build prediction context if available ──────────────────────────────
        last_shap = st.session_state.get("last_shap")
        last_result = st.session_state.get("last_result")
        prediction_context = ""
        if last_shap and last_result:
            shap_vals, feat_names, pred, prob = last_shap
            pairs = sorted(zip(feat_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:5]
            top_features = "; ".join([f"{f}: {v:+.3f}" for f, v in pairs])
            prediction_context = (
                f"Most recent prediction: customer predicted to {'CHURN' if pred == 1 else 'STAY'} "
                f"with {prob:.1%} probability. "
                f"Top 5 SHAP drivers: {top_features}."
            )
        else:
            st.info("💡 Run a **Single Prediction** (Tab 2) first — the AI will then have full context about the customer and their SHAP drivers.")

        # ── Suggested prompts ──────────────────────────────────────────────────
        st.markdown("**Quick questions:**")
        SUGGESTED = [
            "Why is this customer likely to churn?",
            "What retention action has the best ROI?",
            "Which customer segments churn the most?",
            "How does tenure affect churn risk?",
            "What does a SHAP value of +0.3 mean?",
            "How can I retain fiber optic customers?",
        ]
        sp_cols = st.columns(3)
        clicked_prompt = None
        for i, prompt in enumerate(SUGGESTED):
            if sp_cols[i % 3].button(prompt, key=f"sp_{i}", use_container_width=True):
                clicked_prompt = prompt

        st.divider()

        # ── Display conversation history ───────────────────────────────────────
        for msg in st.session_state["ai_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # ── Chat input ─────────────────────────────────────────────────────────
        user_input = st.chat_input("Ask anything about churn…") or clicked_prompt

        if user_input:
            if not api_key:
                st.warning("⚠️ Please configure a Gemini API key above to use the AI Assistant.")
            else:
                st.session_state["ai_messages"].append({"role": "user", "content": user_input})

                # Build system context
                system_ctx = (
                    "You are an expert AI assistant embedded in a Telco Customer Churn Prediction platform. "
                    "The platform uses an XGBoost model (ROC-AUC ≈ 0.84) trained on the IBM Telco dataset (5,042 customers). "
                    "Key features: tenure, MonthlyCharges, TotalCharges, Contract type (Month-to-month/One year/Two year), "
                    "InternetService (DSL/Fiber optic/No), TechSupport, OnlineSecurity, PaymentMethod, SeniorCitizen, gender. "
                    "Overall churn rate is ~26.5%. Month-to-month contracts and Fiber optic internet show the highest churn rates. "
                    "SHAP values explain individual predictions: positive values push toward churn, negative values reduce churn risk. "
                    + (prediction_context if prediction_context else "No individual prediction has been run yet in this session.")
                )

                # Include conversation history for multi-turn context
                history_text = ""
                for msg in st.session_state["ai_messages"][:-1]:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    history_text += f"\n{role}: {msg['content']}"

                full_prompt = (
                    f"System context: {system_ctx}"
                    f"{history_text}"
                    f"\nUser: {user_input}"
                    f"\nAssistant:"
                )

                with st.spinner("Thinking…"):
                    try:
                        from google import genai
                        client = genai.Client(api_key=api_key)
                        response = client.models.generate_content(
                            model="gemini-2.0-flash-exp",
                            contents=[full_prompt],
                        )
                        ai_text = response.text
                        st.session_state["ai_messages"].append({"role": "assistant", "content": ai_text})
                        st.rerun()
                    except Exception as e:
                        err = str(e)
                        # Remove the failed user message so they can retry cleanly
                        st.session_state["ai_messages"].pop()
                        if any(k in err.upper() for k in ["API_KEY", "401", "403", "PERMISSION"]):
                            st.error("❌ Invalid or expired API key. Please check your key in the expander above.")
                        elif any(k in err.lower() for k in ["quota", "429", "rate limit"]):
                            st.error("⏱️ Rate limit reached. Please wait a moment before sending another message.")
                        else:
                            st.error(f"❌ Gemini error: {err}")

        # ── Clear conversation ─────────────────────────────────────────────────
        if st.session_state.get("ai_messages"):
            if st.button("🗑️ Clear conversation", key="clear_chat"):
                st.session_state["ai_messages"] = []
                st.rerun()

        if not api_key:
            st.info("💡 **Tip:** Add `GOOGLE_AI_API_KEY` to Replit Secrets (🔒 icon in the sidebar) for persistent access without re-entering the key each session.")


if __name__ == "__main__":
    main()
