# Raman Tumor Classifier

Базовый шаблон проекта на **Python + uv** для задачи бинарной классификации рамановских спектров:
- `1` — спектр с признаками опухоли;
- `0` — спектр без признаков опухоли.

## 1) Быстрый старт

```bash
uv sync
uv run python -m raman_tumor_classifier.train_baseline
```

## 2) Структура

- `src/raman_tumor_classifier/` — исходный код.
- `data/raw/` — сырые спектры (CSV/Parquet).
- `data/processed/` — подготовленные данные.
- `notebooks/` — Jupyter-ноутбуки для исследования.
- `tests/` — тесты.

## 3) Формат данных (рекомендуемый)

Ожидается таблица, где:
- `label` — целевая переменная (`0/1`),
- остальные колонки — интенсивности по рамановским сдвигам
  (например `shift_400`, `shift_401`, ...).

## 4) Что уже есть

- `train_baseline.py` — baseline-пайплайн:
  - нормализация,
  - `LogisticRegression` для классификации,
  - метрики: `ROC-AUC`, `F1`, `classification_report`.
- `data.py` — утилиты загрузки и разбиения датасета.

## 5) Проверки качества кода

```bash
uv run ruff check .
uv run pytest
```

## 6) Следующие шаги

1. Добавить предобработку спектров:
   - baseline correction,
   - сглаживание (Savitzky–Golay),
   - нормировку (SNV / area normalization).
2. Проверить более сильные модели:
   - `RandomForest`, `XGBoost`, 1D-CNN.
3. Ввести протокол валидации (stratified k-fold + внешняя валидация).
