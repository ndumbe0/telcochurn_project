import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.load_data import load_raw_data, clean_data, preprocess_features
from src.data.eda import generate_eda_report
from src.models.predict import predict_single, predict_batch, get_shap_values, load_model_info
from src.models.evaluate import (
    load_test_data, get_model_comparison_fig, get_roc_and_pr_fig,
    get_confusion_matrix_fig, get_feature_importance_fig, get_metrics_summary,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📊", layout="wide")

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")


# ── Cached loaders ────────────────────────────────────────────────────────────
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


# ── Sidebar customer form ─────────────────────────────────────────────────────
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


# ── Chart helpers ─────────────────────────────────────────────────────────────
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


def display_shap_chart(shap_values, feature_names):
    if shap_values is None or feature_names is None:
        st.info("SHAP explanation not available for this model.")
        return
    pairs = sorted(zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True)
    top_features = pairs[:10]
    names = [p[0] for p in top_features]
    vals = [p[1] for p in top_features]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in vals]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in vals],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Top 10 Feature Contributions (SHAP)",
        xaxis_title="SHAP Value — red = pushes toward churn, blue = away",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


def display_radar_chart(customer_data: dict, df_clean: pd.DataFrame):
    """Radar chart: this customer vs. avg churner vs. avg stayer."""
    if df_clean is None:
        return
    num_feats = ["tenure", "MonthlyCharges", "TotalCharges"]
    available = [f for f in num_feats if f in df_clean.columns]
    if not available:
        return

    # Normalise 0-1 using dataset min/max
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
        title="Customer Profile vs. Average Churner / Stayer",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)


def display_batch_insights(results: pd.DataFrame):
    """Extra visualisations shown after a batch prediction run."""
    st.markdown("---")
    st.subheader("📊 Batch Insights")

    col1, col2, col3, col4 = st.columns(4)
    total = len(results)
    churners = (results["Prediction"] == "Churn").sum()
    stayers = total - churners
    avg_prob = results["Churn_Probability"].mean()
    col1.metric("Total Customers", total)
    col2.metric("Predicted Churn", f"{churners} ({churners/total:.1%})")
    col3.metric("Predicted Stay", stayers)
    col4.metric("Avg Churn Probability", f"{avg_prob:.1f}%")

    c1, c2 = st.columns(2)

    with c1:
        # Risk level donut
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
        # Probability histogram
        fig = px.histogram(
            results, x="Churn_Probability", color="Prediction",
            color_discrete_map={"Churn": "#e74c3c", "Stay": "#2ecc71"},
            nbins=25, title="Churn Probability Distribution",
            labels={"Churn_Probability": "Churn Probability (%)"},
        )
        fig.update_layout(barmode="overlay", bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    # Segment breakdown by Contract if present
    if "Contract" in results.columns:
        seg = (
            results.groupby("Contract")["Prediction"]
            .apply(lambda x: (x == "Churn").mean() * 100)
            .reset_index()
        )
        seg.columns = ["Contract", "Churn Rate (%)"]
        seg = seg.sort_values("Churn Rate (%)", ascending=True)
        fig = go.Figure(
            go.Bar(
                x=seg["Churn Rate (%)"],
                y=seg["Contract"],
                orientation="h",
                marker=dict(color=seg["Churn Rate (%)"], colorscale="RdYlGn_r"),
                text=[f"{v:.1f}%" for v in seg["Churn Rate (%)"]],
                textposition="outside",
            )
        )
        fig.update_layout(title="Churn Rate by Contract Type (Batch)", height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Top 10 highest-risk customers
    st.subheader("🚨 Top 10 Highest-Risk Customers")
    top_risk = results.sort_values("Churn_Probability", ascending=False).head(10)
    display_cols = [c for c in ["customerID", "Contract", "tenure", "MonthlyCharges",
                                "InternetService", "Churn_Probability", "Risk_Level"]
                    if c in top_risk.columns]
    styled = top_risk[display_cols].style.background_gradient(
        subset=["Churn_Probability"] if "Churn_Probability" in display_cols else [],
        cmap="RdYlGn_r",
    )
    st.dataframe(styled, use_container_width=True)


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
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp", contents=[prompt]
        )
        return response.text
    except Exception as e:
        return f"Gemini AI error: {e}"


# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    st.title("📊 Telco Customer Churn Prediction")
    st.markdown("Predict customer churn probability with ML-powered analysis.")

    df_raw, df_clean = load_data()
    pipeline, model_name, cat_cols, num_cols = load_model()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 Single Prediction",
        "📁 Batch Prediction",
        "📈 EDA Dashboard",
        "🏆 Model Performance",
        "🤖 AI Assistant",
    ])

    # ── Tab 1: Single Prediction ──────────────────────────────────────────────
    with tab1:
        st.header("Single Customer Prediction")
        st.markdown(f"**Active Model:** `{model_name or 'Not loaded'}`")

        col1, col2 = st.columns([1, 1])
        with col1:
            customer_data = build_customer_input(cat_cols, num_cols)
        with col2:
            if st.button("Predict Churn", type="primary", use_container_width=True):
                result = predict_single(customer_data)
                if "error" in result:
                    st.error(result["error"])
                else:
                    prob = result["churn_probability"]
                    st.plotly_chart(display_gauge(prob), use_container_width=True)

                    if result["prediction"] == 1:
                        st.error(f"⚠️ Customer is **likely to churn** (Probability: {prob:.1%})")
                    else:
                        st.success(f"✅ Customer is **likely to stay** (Probability: {prob:.1%})")

                    # Risk factor cards
                    risk = "🔴 High" if prob > 0.6 else ("🟡 Medium" if prob > 0.3 else "🟢 Low")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Risk Level", risk)
                    m2.metric("Churn Score", f"{prob:.1%}")
                    m3.metric("Model", model_name or "—")

                    shap_vals, feat_names = get_shap_values(
                        customer_data, pipeline, cat_cols, num_cols
                    )
                    display_shap_chart(shap_vals, feat_names)
                    display_radar_chart(customer_data, df_clean)

                    st.session_state["last_shap"] = (
                        shap_vals, feat_names, result["prediction"], prob
                    )
                    st.session_state["last_customer"] = customer_data

    # ── Tab 2: Batch Prediction ───────────────────────────────────────────────
    with tab2:
        st.header("Batch Prediction")
        st.markdown("Upload a CSV file with customer data to predict churn for multiple customers.")

        sample_path = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "test.csv"
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
                        st.success(f"✅ Predictions complete for {len(results):,} customers!")

                        with st.expander("Full results table", expanded=False):
                            st.dataframe(results, use_container_width=True)

                        csv = results.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="⬇️ Download Results as CSV",
                            data=csv,
                            file_name="churn_predictions.csv",
                            mime="text/csv",
                        )

                        display_batch_insights(results)

                    except Exception as e:
                        st.error(f"Batch prediction failed: {e}")

    # ── Tab 3: EDA Dashboard ──────────────────────────────────────────────────
    with tab3:
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

    # ── Tab 4: Model Performance ──────────────────────────────────────────────
    with tab4:
        st.header("Model Performance")
        if pipeline is None:
            st.warning("No model loaded.")
        else:
            X_test, y_test = load_eval_data()

            # KPI scorecards
            if X_test is not None and y_test is not None:
                metrics = get_metrics_summary(pipeline, X_test, y_test)
                if metrics:
                    st.subheader("📐 Test Set Metrics")
                    cols = st.columns(len(metrics))
                    colors = {"Accuracy": None, "Precision": None,
                              "Recall": None, "F1 Score": None, "ROC-AUC": None}
                    for col, (name, val) in zip(cols, metrics.items()):
                        col.metric(name, f"{val:.4f}")

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
                shap_img = Path(__file__).resolve().parent.parent.parent / "models" / "shap_summary.png"
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

    # ── Tab 5: AI Assistant ───────────────────────────────────────────────────
    with tab5:
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
            st.info("Run a single prediction first, then come back here for an AI explanation.")

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
