"""Jena Climate dataset + windowing pipeline.

Predict the temperature (T degC) some hours into the future from the previous
N hours of hourly samples. We use all numeric columns as features.

The pipeline is parameterized so train.py can sweep window size and forecast
horizon. ``WINDOW_PAST`` and ``HORIZON`` remain module-level defaults for
back-compat with code that imports them directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

CSV_URL = (
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    "jena_climate_2009_2016.csv.zip"
)

SAMPLING_RATE_RAW_PER_HOUR = 6  # raw data is every 10 min
WINDOW_PAST_DAYS = 5
WINDOW_PAST = WINDOW_PAST_DAYS * 24  # 120 hours
HORIZON = 24  # predict 24h ahead
TARGET_COL = "T (degC)"


def _download_csv() -> Path:
    from tensorflow import keras

    zip_path = keras.utils.get_file(
        origin=CSV_URL,
        fname="jena_climate_2009_2016.csv.zip",
        extract=True,
    )
    zp = Path(zip_path)
    candidates = list(zp.parent.rglob("jena_climate_2009_2016.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"CSV not found after extraction near {zp}"
        )
    return candidates[0]


def _load_hourly_array() -> Tuple[np.ndarray, int]:
    """Return (features [N, F] float32, target_col_index)."""
    import pandas as pd

    csv_path = _download_csv()
    df = pd.read_csv(csv_path)
    df = df.iloc[::SAMPLING_RATE_RAW_PER_HOUR].reset_index(drop=True)
    if "Date Time" in df.columns:
        df = df.drop(columns=["Date Time"])
    target_idx = list(df.columns).index(TARGET_COL)
    return df.to_numpy(dtype=np.float32), target_idx


def build_datasets(batch_size: int = 128,
                   window_past: int = WINDOW_PAST,
                   horizon: int = HORIZON):
    """Return (train_ds, val_ds, test_ds, n_features) for the given window/horizon."""
    from tensorflow import keras

    data, target_idx = _load_hourly_array()
    n_total = data.shape[0]

    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    train_raw = data[:n_train]
    val_raw = data[n_train:n_train + n_val]
    test_raw = data[n_train + n_val:]

    mean = train_raw.mean(axis=0)
    std = train_raw.std(axis=0) + 1e-8
    train_n = (train_raw - mean) / std
    val_n = (val_raw - mean) / std
    test_n = (test_raw - mean) / std

    def _windowed(arr: np.ndarray):
        targets = arr[window_past + horizon - 1:, target_idx]
        data_for_windows = arr[:len(arr) - horizon]
        return keras.utils.timeseries_dataset_from_array(
            data=data_for_windows,
            targets=targets,
            sequence_length=window_past,
            sampling_rate=1,
            batch_size=batch_size,
        )

    train_ds = _windowed(train_n)
    val_ds = _windowed(val_n)
    test_ds = _windowed(test_n)
    return train_ds, val_ds, test_ds, data.shape[1]
