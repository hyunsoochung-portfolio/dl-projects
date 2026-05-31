"""Recurrent regressors for Jena Climate forecasting.

The constructors all accept ``window_past`` so train.py can sweep window
sizes without forcing the data module to be re-imported.
"""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from data import WINDOW_PAST


def _compile(model: keras.Model, lr: float = 1e-3) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_simple_rnn(n_features: int, units: int = 32,
                     window_past: int = WINDOW_PAST) -> keras.Model:
    model = keras.Sequential(
        [
            keras.Input(shape=(window_past, n_features)),
            layers.SimpleRNN(units),
            layers.Dense(1),
        ],
        name="simple_rnn",
    )
    return _compile(model)


def build_lstm(n_features: int, units: int = 32,
               window_past: int = WINDOW_PAST,
               bidirectional: bool = False) -> keras.Model:
    rnn = layers.LSTM(units)
    if bidirectional:
        rnn = layers.Bidirectional(rnn)
    model = keras.Sequential(
        [
            keras.Input(shape=(window_past, n_features)),
            rnn,
            layers.Dense(1),
        ],
        name=("bi_lstm" if bidirectional else "lstm"),
    )
    return _compile(model)


def build_gru(n_features: int, units: int = 32,
              window_past: int = WINDOW_PAST) -> keras.Model:
    model = keras.Sequential(
        [
            keras.Input(shape=(window_past, n_features)),
            layers.GRU(units),
            layers.Dense(1),
        ],
        name="gru",
    )
    return _compile(model)


def build_lstm_recurrent_dropout(n_features: int,
                                 units: int = 32,
                                 rec_dropout: float = 0.25,
                                 window_past: int = WINDOW_PAST) -> keras.Model:
    """LSTM with ``recurrent_dropout=...`` between hidden-to-hidden weights."""
    model = keras.Sequential(
        [
            keras.Input(shape=(window_past, n_features)),
            layers.LSTM(units, recurrent_dropout=rec_dropout),
            layers.Dense(1),
        ],
        name=f"lstm_recdrop_{rec_dropout}",
    )
    return _compile(model)


def build_stacked_lstm(n_features: int,
                       units: int = 32,
                       window_past: int = WINDOW_PAST) -> keras.Model:
    """Two-layer LSTM: inner layer uses ``return_sequences=True``."""
    model = keras.Sequential(
        [
            keras.Input(shape=(window_past, n_features)),
            layers.LSTM(units, return_sequences=True),
            layers.LSTM(units),
            layers.Dense(1),
        ],
        name="lstm_stacked",
    )
    return _compile(model)
