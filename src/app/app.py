import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.load_data import load_raw_data, clean_data, preprocess_features
from src.data.eda import generate_eda_report
from src.models.predict import predict_single, predict_batch, get_shap_values, load_model_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📊", layout="wide")

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")


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


def build_customer_input(cat_cols, num_cols):
    st.sidebar.header("Customer Details")
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
    total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, monthly_charges * tenure)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    return {
        "gender": gender, "SeniorCitizen": senior_citizen, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
        "MultipleLines": multiple_lines, "InternetService": internet_service,
        "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_protection, "TechSupport": tech_support,
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        "Contract": contract, "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }


def display_gauge(probability):
    fig = go.Figure(go.Indicator(
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
    ))
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
    colors = ["red" if v > 0 else "blue" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals,
        y=names,
        orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        title="Top 10 Feature Contributions (SHAP)",
        xaxis_title="SHAP Value (impact on model output)",
        yaxis_title="Feature",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def get_gemini_explanation(shap_values, feature_names, prediction, probability, api_key):
    if not api_key:
        return "Gemini AI assistant is not configured. Set GOOGLE_AI_API_KEY in your .env file."
    if shap_values is None:
        return "SHAP values not available for explanation."
    try:
        from google import genai
        from google.genai import types
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
            model="gemini-2.0-flash-exp",
            contents=[prompt],
        )
        return response.text
    except Exception as e:
        return f"Gemini AI error: {e}"


def main():
    st.title("📊 Telco Customer Churn Prediction")
    st.markdown("Predict customer churn probability with ML-powered analysis.")

    df_raw, df_clean = load_data()
    pipeline, model_name, cat_cols, num_cols = load_model()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Single Prediction", "📁 Batch Prediction", "📈 EDA Dashboard", "🤖 AI Assistant"
    ])

    with tab1:
        st.header("Single Customer Prediction")
        st.markdown(f"**Active Model:** {model_name or 'Not loaded'}")
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
                    gauge = display_gauge(prob)
                    st.plotly_chart(gauge, use_container_width=True)
                    if result["prediction"] == 1:
                        st.error(f"⚠️ Customer is **likely to churn** (Probability: {prob:.1%})")
                    else:
                        st.success(f"✅ Customer is **likely to stay** (Probability: {prob:.1%})")
                    shap_vals, feat_names = get_shap_values(customer_data, pipeline, cat_cols, num_cols)
                    display_shap_chart(shap_vals, feat_names)
                    st.session_state["last_shap"] = (shap_vals, feat_names, result["prediction"], prob)

    with tab2:
        st.header("Batch Prediction")
        st.markdown("Upload a CSV file with customer data to predict churn for multiple customers.")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df_batch = pd.read_csv(uploaded_file)
            st.write("Preview:", df_batch.head())
            if st.button("Run Batch Prediction"):
                with st.spinner("Predicting..."):
                    try:
                        results = predict_batch(df_batch)
                        st.success("Predictions complete!")
                        st.dataframe(results)
                        csv = results.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="churn_predictions.csv",
                            mime="text/csv",
                        )
                        fig = px.histogram(results, x="Churn_Probability", color="Prediction",
                                           title="Batch Prediction Distribution", nbins=20)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Batch prediction failed: {e}")

    with tab3:
        st.header("EDA Dashboard")
        if df_clean is not None:
            with st.spinner("Generating EDA figures..."):
                figs = get_eda_figures(df_clean)
            for name, fig in figs.items():
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Data not available for EDA.")

    with tab4:
        st.header("AI Assistant")
        st.markdown("Ask questions about churn predictions. Powered by Google Gemini.")
        api_key = GOOGLE_AI_API_KEY
        if not api_key:
            st.warning("GOOGLE_AI_API_KEY not found in environment. Add it to .env file.")
            api_key = st.text_input("Enter Gemini API key (optional)", type="password")
        last_shap = st.session_state.get("last_shap")
        if last_shap and api_key:
            shap_vals, feat_names, pred, prob = last_shap
            if st.button("Why is this customer predicted to churn?"):
                with st.spinner("Analyzing with Gemini..."):
                    explanation = get_gemini_explanation(shap_vals, feat_names, pred, prob, api_key)
                    st.markdown("### AI Analysis")
                    st.write(explanation)
        question = st.text_input("Or ask any question about churn:")
        if question and api_key:
            with st.spinner("Thinking..."):
                try:
                    from google import genai
                    client = genai.Client(api_key=api_key)
                    response = client.models.generate_content(
                        model="gemini-2.0-flash-exp",
                        contents=[f"Context: This is a telco customer churn prediction system using features like tenure, contract type, monthly charges, payment method, etc. Question: {question}"],
                    )
                    st.markdown("### Answer")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
