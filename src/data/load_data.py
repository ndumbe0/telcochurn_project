import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


def load_raw_data(filepath: str | Path = None) -> pd.DataFrame:
    if filepath is None:
        filepath = DATA_DIR / "telco.csv"
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning(f"{filepath} not found. Checking alternative sources...")
        alt = DATA_DIR / "cleaned.csv"
        if alt.exists():
            filepath = alt
        else:
            raise FileNotFoundError(f"No data file found in {DATA_DIR}")
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath, index_col=0)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning data...")
    df = df.copy()
    df = df.drop(columns=["customerID"], errors="ignore")
    yes_no_map = {"Yes": 1, "No": 0, "True": 1, "False": 0, True: 1, False: 0}
    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().map(yes_no_map).fillna(0).astype(int)
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].astype(str).str.strip().map(yes_no_map)
        df = df.dropna(subset=["Churn"])
        df["Churn"] = df["Churn"].astype(int)
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    internet_services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    for col in internet_services:
        if col in df.columns:
            df[col] = df[col].fillna("No internet service")
    if "MultipleLines" in df.columns:
        df["MultipleLines"] = df["MultipleLines"].fillna("No")
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
    df = df.dropna()
    logger.info(f"Data cleaned: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def preprocess_features(df: pd.DataFrame, target_col: str = "Churn") -> tuple:
    logger.info("Preprocessing features...")
    df = df.copy()
    y = df[target_col].values if target_col in df.columns else None
    X = df.drop(columns=[target_col], errors="ignore")
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return X, y, cat_cols, num_cols


def split_data(X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def save_processed_data(X_train, X_test, y_train, y_test, cat_cols, num_cols):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df = X_train.copy()
    train_df["Churn"] = y_train
    test_df = X_test.copy()
    test_df["Churn"] = y_test
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    import joblib
    joblib.dump({"cat_cols": cat_cols, "num_cols": num_cols}, PROCESSED_DIR / "column_info.pkl")
    logger.info("Processed data saved to data/processed/")


def load_and_prepare_data():
    df = load_raw_data()
    df = clean_data(df)
    X, y, cat_cols, num_cols = preprocess_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    save_processed_data(X_train, X_test, y_train, y_test, cat_cols, num_cols)
    return X_train, X_test, y_train, y_test, cat_cols, num_cols
