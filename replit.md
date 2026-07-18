# Telco Customer Churn Prediction

A production-grade machine learning dashboard built with **Streamlit** that predicts telecom customer churn. Includes interactive single/batch predictions, SHAP explainability charts, an EDA dashboard, and an optional Google Gemini AI assistant.

## How to Run

The **"Start application"** workflow launches the Streamlit dashboard:

```
streamlit run src/app/app.py --server.port 5000 --server.address 0.0.0.0 --server.headless true
```

The pre-trained XGBoost model and cleaned data are already present — no pipeline re-run is needed to start the app.

## Re-training the Model

To re-run the full ML pipeline (~5 minutes):

```bash
python run_pipeline.py
```

This retrains all 6 models (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, Gradient Boosting), evaluates them, and saves the best model to `models/best_model.pkl`.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_AI_API_KEY` | No | Google Gemini key for the AI Assistant tab |

Set via Replit Secrets (no `.env` file needed on Replit).

## Project Structure

```
src/
  app/app.py          # Streamlit dashboard (main entry point)
  data/               # Data loading, cleaning, EDA
  models/             # Training, prediction, evaluation
data/                 # Raw and cleaned CSV files
models/               # Saved model artifacts (best_model.pkl)
run_pipeline.py       # Full ML pipeline runner
run_app.py            # App launcher (alternative to workflow command)
```

## Stack

- **Frontend:** Streamlit, Plotly
- **ML:** scikit-learn, XGBoost, LightGBM, SHAP, imbalanced-learn (SMOTE)
- **Generative AI:** Google Gemini (optional)
- **Core:** Python 3.12, pandas, numpy, joblib

## User Preferences

_No preferences recorded yet._
