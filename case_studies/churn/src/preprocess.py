from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_and_split(
    df_raw: pd.DataFrame,
    target: str,
    test_size: float,
    random_seed: int,
    train_path: Path,
    test_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Minimal preprocessing and leakage-safe splitting.

    For this synthetic dataset, we mainly:
    - ensure target exists
    - drop obvious invalid rows (none expected)
    - perform stratified split
    - persist processed splits for transparency/reproducibility
    """
    if target not in df_raw.columns:
        raise ValueError(f"Target column '{target}' not found in input data.")

    df = df_raw.copy()

    # Basic sanity cleaning (extendable)
    df = df.dropna(subset=[target])

    y = df[target].astype(int)
    X = df.drop(columns=[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

    # Save processed splits (so repo users can see what's used)
    train_df = X_train.copy()
    train_df[target] = y_train.values
    test_df = X_test.copy()
    test_df[target] = y_test.values

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return X_train, X_test, y_train, y_test
