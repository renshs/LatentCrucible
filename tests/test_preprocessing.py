import numpy as np
import pytest

from raman_tumor_classifier.preprocessing import RamanPreprocessor


def test_preprocessor_preserves_shape_and_returns_finite_values() -> None:
    x = np.array(
        [
            [1.0, 2.0, 4.0, 3.0, 2.0, 1.0],
            [0.5, 0.8, 1.2, 1.0, 0.7, 0.4],
        ]
    )

    pre = RamanPreprocessor(savgol_window_length=5, savgol_polyorder=2, normalization="snv")
    out = pre.fit_transform(x)

    assert out.shape == x.shape
    assert np.isfinite(out).all()


def test_area_normalization_makes_unit_area_for_positive_signal() -> None:
    x = np.array([[0.0, 1.0, 2.0, 1.0, 0.0]])

    pre = RamanPreprocessor(
        apply_baseline_correction=False,
        apply_savgol=False,
        normalization="area",
    )
    out = pre.fit_transform(x)

    area = np.trapezoid(np.abs(out), axis=1, dx=1.0)
    assert area[0] == pytest.approx(1.0, rel=1e-6)


def test_invalid_normalization_raises_value_error() -> None:
    with pytest.raises(ValueError, match="normalization must be one of"):
        RamanPreprocessor(normalization="bad").fit(np.ones((2, 8)))
