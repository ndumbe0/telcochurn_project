# Telco Customer Churn Prediction

A production-grade machine learning system that predicts customer churn for a telecommunications company. Includes a Streamlit web app with real-time predictions, SHAP explanations, EDA dashboard, batch processing, and an AI assistant powered by Google Gemini.

## Features

- **Multiple ML Models**: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, Voting Ensemble
- **Class Imbalance Handling**: SMOTE oversampling + balanced class weights
- **Hyperparameter Tuning**: RandomizedSearchCV with StratifiedKFold cross-validation
- **Model Explainability**: SHAP value analysis with visual feature importance charts
- **Interactive Web App**: Streamlit dashboard with:
  - Single customer prediction with gauge chart
  - Batch CSV upload with downloadable results
  - EDA dashboard (churn by tenure, contract, payment method, etc.)
  - Gemini AI assistant for natural language explanations
- **Docker Support**: Ready-to-deploy with Docker and docker-compose

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── load_data.py    # Data loading, cleaning, preprocessing
│   │   └── eda.py          # Exploratory data analysis & visualization
│   ├── models/
│   │   ├── train.py        # Model training, tuning, SHAP explanations
│   │   └── predict.py      # Single & batch prediction functions
│   └── app/
│       └── app.py          # Streamlit web application
├── data/
│   ├── telco.csv           # Raw dataset
│   ├── cleaned.csv         # Cleaned dataset
│   └── processed/          # Train/test splits (generated)
├── models/
│   ├── best_model.pkl      # Trained best model (generated)
│   └── model_comparison.csv # Model performance comparison
├── notebooks/
│   └── telco churn project.ipynb  # Original analysis notebook
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── run_pipeline.py         # Run full training pipeline
├── run_app.py              # Launch Streamlit app
└── .env.example            # Environment variables template
```

## Setup

### Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/ndumbe0/telcochurn_project.git
cd telcochurn_project
```

2. **Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables** (optional, for Gemini AI)
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_AI_API_KEY
```

5. **Run the training pipeline**
```bash
python run_pipeline.py
```

6. **Launch the web app**
```bash
streamlit run src/app/app.py
```

### Docker Deployment

```bash
docker-compose up --build
```

Then open http://localhost:8501 in your browser.

## Usage

### Single Prediction
Fill in customer details in the sidebar and click "Predict Churn" to get:
- Churn probability gauge chart
- Prediction (churn / stay)
- SHAP feature contribution bar chart
- AI analysis (if Gemini API key is configured)

### Batch Prediction
Upload a CSV file with customer data (same columns as training data). Download results with prediction probabilities.

### EDA Dashboard
Explore interactive visualizations of the dataset including churn rates by tenure, contract type, payment method, and more.

### AI Assistant
Ask questions about churn predictions. The assistant can analyze individual predictions using SHAP values and provide business recommendations.

## Models

The pipeline trains and evaluates 6 models:
| Model | Technique |
|-------|-----------|
| Logistic Regression | Baseline linear model |
| Decision Tree | Interpretable tree model |
| Random Forest | Ensemble of decision trees |
| XGBoost | Gradient boosted trees |
| LightGBM | Efficient gradient boosting |
| Voting Ensemble | Soft voting of all models |

The best model is selected by ROC-AUC score and tuned via RandomizedSearchCV.

## Dataset

The dataset contains ~7,000 customers with 20 features:
- **Demographics**: gender, SeniorCitizen, Partner, Dependents
- **Services**: PhoneService, InternetService, OnlineSecurity, TechSupport, etc.
- **Account**: tenure, Contract, PaymentMethod, MonthlyCharges, TotalCharges
- **Target**: Churn (Yes/No)

## License

This project was developed as part of the Azubi Africa Career Accelerator Program.

**Owner**: Moses N Ndumbe (ndumbemoses@gmail.com)
