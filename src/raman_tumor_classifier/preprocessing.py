from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

NormalizationType = Literal["snv", "area", "none"]


class RamanPreprocessor(BaseEstimator, TransformerMixin):
    """Предобработка рамановских спектров: baseline correction, Savitzky-Golay и нормализация."""

    def __init__(
        self,
        apply_baseline_correction: bool = True,
        baseline_poly_order: int = 2,
        apply_savgol: bool = True,
        savgol_window_length: int = 9,
        savgol_polyorder: int = 2,
        normalization: NormalizationType = "snv",
    ) -> None:
        self.apply_baseline_correction = apply_baseline_correction
        self.baseline_poly_order = baseline_poly_order
        self.apply_savgol = apply_savgol
        self.savgol_window_length = savgol_window_length
        self.savgol_polyorder = savgol_polyorder
        self.normalization = normalization

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> RamanPreprocessor:
        arr = self._as_2d_array(x)
        self.n_features_in_ = arr.shape[1]
        self._validate_params(self.n_features_in_)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        arr = self._as_2d_array(x)
        self._validate_params(arr.shape[1])

        out = arr.copy()

        if self.apply_baseline_correction:
            out = self._baseline_correct(out)

        if self.apply_savgol:
            out = self._smooth_savgol(out)

        out = self._normalize(out)
        return out

    @staticmethod
    def _as_2d_array(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 2:
            raise ValueError("Input spectra must be a 2D array-like structure")
        return arr

    def _validate_params(self, n_features: int) -> None:
        if self.normalization not in {"snv", "area", "none"}:
            raise ValueError("normalization must be one of: 'snv', 'area', 'none'")

        if self.baseline_poly_order < 0:
            raise ValueError("baseline_poly_order must be >= 0")

        if self.savgol_polyorder < 0:
            raise ValueError("savgol_polyorder must be >= 0")

        if self.savgol_window_length < 3:
            raise ValueError("savgol_window_length must be >= 3")

        if self.savgol_window_length <= self.savgol_polyorder:
            raise ValueError("savgol_window_length must be > savgol_polyorder")

        if self.baseline_poly_order >= n_features:
            raise ValueError("baseline_poly_order must be less than n_features")

    def _baseline_correct(self, x: np.ndarray) -> np.ndarray:
        x_axis = np.arange(x.shape[1], dtype=float)
        corrected = np.empty_like(x)

        for i, row in enumerate(x):
            coefs = np.polyfit(x_axis, row, deg=self.baseline_poly_order)
            baseline = np.polyval(coefs, x_axis)
            corrected[i] = row - baseline

        return corrected

    def _smooth_savgol(self, x: np.ndarray) -> np.ndarray:
        from scipy.signal import savgol_filter

        window = min(self.savgol_window_length, x.shape[1])
        if window % 2 == 0:
            window -= 1

        min_valid_window = self.savgol_polyorder + 2
        if min_valid_window % 2 == 0:
            min_valid_window += 1

        if window < min_valid_window:
            window = min_valid_window

        if window > x.shape[1]:
            return x

        return savgol_filter(x, window_length=window, polyorder=self.savgol_polyorder, axis=1)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.normalization == "none":
            return x

        if self.normalization == "snv":
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True)
            std = np.where(std == 0.0, 1.0, std)
            return (x - mean) / std

        area = np.trapezoid(np.abs(x), axis=1, dx=1.0)
        area = np.where(area == 0.0, 1.0, area)
        return x / area[:, None]
