"""
src/models_gru.py
-----------------
GRU, LSTM-GRU ve LSTM-BiGRU talep tahmin modelleri.
"""

from tensorflow import keras
from tensorflow.keras import layers


def _compile(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss="huber",
        metrics=["mae"],
    )
    return model


def _regression_head(x, dropout):
    x = layers.Dropout(dropout, name="drop_head")(x)
    x = layers.Dense(32, activation="relu", name="dense_32")(x)
    return layers.Dense(1, name="occupancy_output")(x)


def build_gru_baseline(input_shape, gru_units=64, dropout=0.2, lr=1e-3):
    inputs = keras.Input(shape=input_shape, name="input_window")
    x = layers.GRU(gru_units, name="gru")(inputs)
    outputs = _regression_head(x, dropout)
    return _compile(keras.Model(inputs, outputs, name="gru_baseline"), lr)


def build_lstm_gru(input_shape, lstm_units=64, gru_units=32, dropout=0.2, lr=1e-3):
    inputs = keras.Input(shape=input_shape, name="input_window")
    x = layers.LSTM(lstm_units, return_sequences=True, name="lstm")(inputs)
    x = layers.Dropout(dropout, name="drop_lstm")(x)
    x = layers.GRU(gru_units, name="gru")(x)
    outputs = _regression_head(x, dropout)
    return _compile(keras.Model(inputs, outputs, name="lstm_gru"), lr)


def build_lstm_bigru(input_shape, lstm_units=64, gru_units=32, dropout=0.2, lr=1e-3):
    inputs = keras.Input(shape=input_shape, name="input_window")
    x = layers.LSTM(lstm_units, return_sequences=True, name="lstm")(inputs)
    x = layers.Dropout(dropout, name="drop_lstm")(x)
    x = layers.Bidirectional(layers.GRU(gru_units), name="bigru")(x)
    outputs = _regression_head(x, dropout)
    return _compile(keras.Model(inputs, outputs, name="lstm_bigru"), lr)
