"""Data loading and leakage-free preprocessing pipeline."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "diabetes.csv"
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
TARGET = "Outcome"
# A zero is physiologically impossible for these columns -> it means "missing".
ZERO_IS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
RANDOM_STATE = 42


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """Stratified train/test split. Test set is never touched until final evaluation."""
    return train_test_split(
        df[FEATURES], df[TARGET], test_size=test_size,
        stratify=df[TARGET], random_state=RANDOM_STATE,
    )


def zero_to_nan(X: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(X).copy()
    for col in ZERO_IS_MISSING:
        X[col] = X[col].replace(0, np.nan)
    return X


def build_pipeline(model) -> Pipeline:
    """zero->NaN, median-impute, scale, model. Fit only on training folds => no leakage."""
    return Pipeline([
        ("zero_to_nan", FunctionTransformer(zero_to_nan)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", model),
    ])
