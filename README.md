# Telco Customer Churn Prediction

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-purple)](https://xgboost.ai)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0-green)](https://lightgbm.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.44-yellow)](https://shap.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

A **production-grade** machine learning system for predicting customer churn in telecommunications. Features an interactive Streamlit dashboard with real-time predictions, SHAP explainability, AI-powered insights via Google Gemini, and full Docker support.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Features](#features)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Performance](#model-performance)
- [Feature Importance (SHAP)](#feature-importance-shap)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [Tech Stack](#tech-stack)
- [License & Credits](#license--credits)

---

## Overview

| Aspect | Detail |
|--------|--------|
| **Goal** | Predict which customers are likely to churn |
| **Dataset** | 7,043 customers, 20 features (IBM Telco Dataset) |
| **Best Model** | XGBoost / LightGBM (tuned via RandomizedSearchCV) |
| **Performance** | AUC-ROC ≈ 0.85, F1 ≈ 0.64 |
| **Imbalance Handling** | SMOTE oversampling + `class_weight="balanced"` |
| **Deployment** | Streamlit + Docker |

---

## Pipeline

```
┌──────────┐     ┌──────────┐     ┌───────────┐     ┌──────────┐     ┌───────────┐
│   Raw    │ ──► │  Clean   │ ──► │  SMOTE    │ ──► │  Train   │ ──► │  Evaluate │
│  Data    │     │  & Encode│     │ Oversample│     │ 6 Models │     │ + SHAP    │
└──────────┘     └──────────┘     └───────────┘     └──────────┘     └───────────┘
                                                          │
                                                          ▼
                                                  ┌──────────────┐
                                                  │ Best Model   │
                                                  │ (Voting /    │
                                                  │  XGBoost)    │
                                                  └──────┬───────┘
                                                         │
                                                  ┌──────▼───────┐
                                                  │  Streamlit   │
                                                  │  Dashboard   │
                                                  └──────────────┘
```

**Data Flow:**
1. Raw CSV ingested → missing values handled → categorical features encoded
2. Train/test split (80/20) → SMOTE applied to training set only
3. 6 models trained + tuned with RandomizedSearchCV + StratifiedKFold
4. Best model saved → SHAP TreeExplainer generates global + local explanations
5. Streamlit app loads model + preprocessor for real-time inference

---

## Features

### 🔮 Churn Prediction
- **Single customer**: Fill in details via sidebar sliders/dropdowns → instant prediction with probability gauge
- **Batch upload**: Upload CSV → download predictions with probability scores
- **Explainability**: SHAP force plots & bar charts show why a customer is predicted to churn

### 📈 EDA Dashboard
Interactive visualizations covering:
- Churn rate by contract type, payment method, internet service
- Tenure distribution by churn status
- Monthly charges and total charges analysis
- Feature correlation heatmap
- Senior citizen and demographic analysis

### 🤖 AI Assistant (Gemini)
Ask natural language questions like *"Why is this customer likely to churn?"* — the AI receives SHAP values and returns business-friendly explanations with retention recommendations.

---

## Exploratory Data Analysis

### Churn Rate
![Churn Rate](images/readme_churn_rate.png)

Dataset is imbalanced — ~27% of customers churned. SMOTE oversampling brings the training minority class to 80% of majority.

---

### Churn by Contract Type
![Contract Churn](images/readme_contract_churn.png)

Month-to-month contracts have significantly higher churn rates. Customers on one-year or two-year contracts are far more likely to stay.

---

### Churn by Gender
![Churn by Gender](images/churn%20by%20gender-train.png)

Churn is relatively balanced across genders, indicating gender alone is not a strong predictor.

---

### Tenure Distribution
![Tenure by Churn](images/readme_tenure_by_churn.png)

Customers who churn tend to have shorter tenure (under 20 months), while long-tenure customers (40+ months) overwhelmingly stay.

---

### Monthly Charges Analysis
![Monthly Charges](images/readme_monthly_charges_churn.png)

Customers paying higher monthly charges (> $70) show elevated churn rates. Lower monthly charges correlate with higher retention.

---

### Monthly Charges Distribution
![Monthly Charges Distribution](images/monthly%20charges%20distribution-train.png)

Distribution reveals a bimodal pattern — clusters of customers at low (~$20) and high (~$95) monthly charges.

---

### Monthly Charges by Contract Type
![Monthly Charges by Contract](images/monthly%20charges%20by%20contract%20type-train.png)

Month-to-month customers span a wider range of monthly charges, while long-term contracts cluster at higher price points (likely premium plans).

---

### Total Charges vs Tenure
![Total Charges vs Tenure](images/total%20charges%20vs%20tenure-train.png)

Strong linear relationship between tenure and total charges. Churned customers (orange) tend to have lower total charges due to shorter tenure.

---

### Churn by Payment Method
![Payment Method](images/readme_payment_churn.png)

Electronic check users churn at a significantly higher rate than customers using automated payment methods (bank transfer, credit card).

---

### Feature Correlation Heatmap
![Correlation](images/readme_correlation.png)

Tenure and total charges are strongly correlated. Contract type and payment method show moderate correlation with churn.

---

## Model Performance

| Model | AUC-ROC | F1 | Precision | Recall |
|-------|---------|----|-----------|--------|
| **Logistic Regression** | 0.85 | 0.64 | 0.59 | 0.69 |
| **Voting Ensemble** | 0.84 | 0.63 | 0.58 | 0.68 |
| **LightGBM** | 0.83 | 0.59 | 0.55 | 0.65 |
| **Random Forest** | 0.83 | 0.61 | 0.57 | 0.65 |
| **XGBoost** | 0.81 | 0.58 | 0.54 | 0.64 |
| **Decision Tree** | 0.75 | 0.57 | 0.52 | 0.64 |

The best model is tuned via **RandomizedSearchCV with StratifiedKFold (5-fold)** cross-validation. **SMOTE** (`sampling_strategy=0.8`) is applied to the training set to address class imbalance, and all estimators use `class_weight="balanced"`.

---

## Feature Importance (SHAP)

![SHAP Summary](images/readme_shap_summary.png)

**Top predictors of churn** (by SHAP value magnitude):
1. **Contract type** — Month-to-month contracts strongly increase churn probability
2. **Tenure** — Short tenure is a top churn indicator
3. **Monthly Charges** — Higher charges increase churn risk
4. **Internet Service** — Fiber optic customers are more likely to churn
5. **Payment Method** — Electronic check users are at higher risk

---

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── load_data.py      # Data loading, cleaning, preprocessing
│   │   └── eda.py             # EDA visualization generation
│   ├── models/
│   │   ├── train.py           # Model training, tuning, SHAP
│   │   ├── quick_train.py     # Lightweight training variant
│   │   └── predict.py         # Single & batch prediction
│   └── app/
│       └── app.py             # Streamlit web application
├── data/
│   ├── telco.csv              # Raw dataset (7,043 rows)
│   ├── cleaned.csv            # Cleaned dataset (5,042 rows)
│   └── processed/             # Train/test splits (generated)
├── models/
│   ├── best_model.pkl         # Trained best model (818 KB)
│   └── model_comparison.csv   # Performance metrics comparison
├── images/                    # README figures & EDA exports
├── notebooks/
│   └── telco churn project.ipynb  # Original reference notebook
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── run_pipeline.py            # Run full ML pipeline
├── run_app.py                 # Launch Streamlit app
└── .env.example               # Environment variable template
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- pip

### Local Installation

```bash
# Clone the repository
git clone https://github.com/ndumbe0/telcochurn_project.git
cd telcochurn_project

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (optional — needed for AI Assistant)
cp .env.example .env
# Edit .env and add: GOOGLE_AI_API_KEY=your_gemini_key_here

# Run the full ML pipeline (~5 minutes)
python run_pipeline.py

# Launch the Streamlit dashboard
python run_app.py
# Or: streamlit run src/app/app.py
```

### Docker Deployment

```bash
docker-compose up --build
# Open http://localhost:8501
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_AI_API_KEY` | No | Gemini API key for AI assistant features in the dashboard |

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Frontend** | Streamlit, Plotly, Altair |
| **ML/AI** | scikit-learn, XGBoost, LightGBM, SHAP, imbalanced-learn (SMOTE) |
| **Generative AI** | Google Gemini (google-genai) |
| **Deployment** | Docker, docker-compose |
| **Core** | Python 3.11, pandas, numpy, joblib, python-dotenv |

---

## License & Credits

**Owner:** Moses N Ndumbe ([ndumbemoses@gmail.com](mailto:ndumbemoses@gmail.com))

Built as part of the **Azubi Africa Career Accelerator Program** (LP2 Classification Project).

**Team Lead:** Ms. Portia Bentum ([portia.bentum@azubiafrica.org](mailto:portia.bentum@azubiafrica.org))

---

> **Disclaimer:** Environment variables, database credentials, and API keys are never committed. See `.env.example` for the required template.
