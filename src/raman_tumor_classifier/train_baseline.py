from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from raman_tumor_classifier.data import DataFormatError, load_dataset, split_dataset


def build_pipeline() -> Pipeline:
    """Собирает baseline-пайплайн классификации."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )


def train_and_evaluate(dataset_path: str | Path) -> dict[str, float]:
    """Обучает baseline модель и возвращает ключевые метрики."""
    x, y = load_dataset(dataset_path)
    x_train, x_test, y_train, y_test = split_dataset(x, y)

    model = build_pipeline()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred)),
    }

    print("Metrics:", metrics)
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred, digits=3))

    return metrics


def _make_demo_dataset(path: Path, n_samples: int = 120, n_features: int = 64) -> None:
    """Генерирует игрушечный датасет для быстрой проверки пайплайна."""
    import numpy as np

    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, size=(n_samples, n_features))
    signal = x[:, :8].mean(axis=1)
    y = (signal + rng.normal(0, 0.5, size=n_samples) > 0).astype(int)

    columns = [f"shift_{i}" for i in range(n_features)]
    df = pd.DataFrame(x, columns=columns)
    df["label"] = y
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    dataset_path = Path("data/processed/demo_raman_spectra.csv")
    if not dataset_path.exists():
        _make_demo_dataset(dataset_path)
        print(f"Demo dataset created at {dataset_path}")

    try:
        train_and_evaluate(dataset_path)
    except (FileNotFoundError, DataFormatError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
