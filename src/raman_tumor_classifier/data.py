from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


class DataFormatError(ValueError):
    """Ошибка формата входных данных."""


def load_dataset(path: str | Path, label_column: str = "label") -> tuple[pd.DataFrame, pd.Series]:
    """Загружает CSV-датасет спектров и возвращает признаки и целевую переменную."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if label_column not in df.columns:
        raise DataFormatError(f"Label column '{label_column}' is missing")

    y = df[label_column].astype(int)
    x = df.drop(columns=[label_column])

    if x.empty:
        raise DataFormatError("No feature columns found")

    return x, y


def split_dataset(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Стратифицированное разбиение на train/test."""
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
