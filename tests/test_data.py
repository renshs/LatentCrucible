from pathlib import Path

import pandas as pd

from raman_tumor_classifier.data import load_dataset


def test_load_dataset_splits_features_and_target(tmp_path: Path) -> None:
    csv_path = tmp_path / "spectra.csv"
    pd.DataFrame(
        {
            "shift_400": [0.1, 0.2, 0.3],
            "shift_401": [1.0, 1.1, 1.2],
            "label": [0, 1, 0],
        }
    ).to_csv(csv_path, index=False)

    x, y = load_dataset(csv_path)

    assert list(x.columns) == ["shift_400", "shift_401"]
    assert y.tolist() == [0, 1, 0]
