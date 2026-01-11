from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _build_model(model_type: str, model_params: Dict[str, Any]) -> object:
    if model_type == "logistic_regression":
        return LogisticRegression(**model_params)
    raise ValueError(f"Unsupported model type: {model_type}. Try 'logistic_regression'.")


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    categorical: List[str],
    numeric: List[str],
    model_type: str,
    model_params: Dict[str, Any],
    out_path: Path,
):
    """Train a model pipeline and persist as joblib."""
    missing = [c for c in (categorical + numeric) if c not in X_train.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", Pipeline([("scaler", StandardScaler())]), numeric),
        ],
        remainder="drop",
    )

    clf = _build_model(model_type, model_params)

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", clf),
        ]
    )

    model.fit(X_train, y_train)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)

    return model
