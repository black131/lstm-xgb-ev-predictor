"""
src/models.py
-------------
LSTM, CNN-LSTM, CNN-BiLSTM ve Attention tabanlı talep tahmin modelleri.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_baseline(input_shape, lstm_units=64, dropout=0.2, lr=1e-3):
    inputs = keras.Input(shape=input_shape)
    x = layers.LSTM(lstm_units, return_sequences=False, name="lstm")(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs, outputs, name="lstm_baseline")
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="huber", metrics=["mae"])
    return model


def build_cnn_lstm(input_shape,
                    conv_filters=32,
                    conv_kernel=3,
                    lstm_units=64,
                    dropout=0.2,
                    lr=1e-3):
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv1D(conv_filters, conv_kernel, padding="causal", activation="relu")(inputs)
    x = layers.Conv1D(conv_filters, conv_kernel, padding="causal", activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout)(x)

    x = layers.LSTM(lstm_units, return_sequences=False)(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs, outputs, name="cnn_lstm_hybrid")
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="huber", metrics=["mae"])
    return model


def build_cnn_bilstm(input_shape,
                     conv_filters=16,
                     bilstm_units=48,
                     dropout=0.2,
                     lr=1e-3):
    """Çok ölçekli Conv1D + Residual bağlantı + BiLSTM hibrit modeli."""
    inputs = keras.Input(shape=input_shape, name="input_window")

    c_short = layers.Conv1D(
        conv_filters, 3, padding="causal", activation="relu", name="conv1d_k3"
    )(inputs)
    c_short = layers.BatchNormalization(name="bn_k3")(c_short)

    c_medium = layers.Conv1D(
        conv_filters, 5, padding="causal", activation="relu", name="conv1d_k5"
    )(inputs)
    c_medium = layers.BatchNormalization(name="bn_k5")(c_medium)

    feat = layers.Concatenate(name="residual_concat")([inputs, c_short, c_medium])

    x = layers.Bidirectional(
        layers.LSTM(bilstm_units, return_sequences=False),
        name="bilstm"
    )(feat)
    x = layers.Dropout(dropout, name="drop_bilstm")(x)

    x = layers.Dense(32, activation="relu", name="dense_32")(x)
    outputs = layers.Dense(1, name="occupancy_output")(x)

    model = keras.Model(inputs, outputs, name="cnn_bilstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss="huber",
        metrics=["mae"],
    )
    return model


build_cnn_lstm_v2 = build_cnn_bilstm


def _multi_scale_cnn(inputs, conv_filters):
    c_short = layers.Conv1D(
        conv_filters, 3, padding="causal", activation="relu", name="conv1d_k3"
    )(inputs)
    c_short = layers.BatchNormalization(name="bn_k3")(c_short)

    c_medium = layers.Conv1D(
        conv_filters, 5, padding="causal", activation="relu", name="conv1d_k5"
    )(inputs)
    c_medium = layers.BatchNormalization(name="bn_k5")(c_medium)

    return layers.Concatenate(name="residual_concat")([inputs, c_short, c_medium])


def _temporal_attention(h, attn_units):
    """Bahdanau zamansal attention katmanı."""
    window = h.shape[1]

    score = layers.Dense(attn_units, activation="tanh", name="attn_dense")(h)
    score = layers.Dense(1, use_bias=False, name="attn_score")(score)
    alpha = layers.Softmax(axis=1, name="attention_weights")(score)

    context = layers.Dot(axes=1, name="attn_context")([alpha, h])
    context = layers.Flatten(name="attn_context_flat")(context)

    last = layers.Cropping1D((window - 1, 0), name="last_step")(h)
    last = layers.Flatten(name="last_step_flat")(last)

    return layers.Concatenate(name="context_plus_last")([context, last])


def _build_attention_model(input_shape, recurrent_layer, name,
                           conv_filters, attn_units, dropout, lr):
    inputs = keras.Input(shape=input_shape, name="input_window")
    feat = _multi_scale_cnn(inputs, conv_filters)
    h = recurrent_layer(feat)
    x = _temporal_attention(h, attn_units)
    x = layers.Dropout(dropout, name="drop_attn")(x)
    x = layers.Dense(32, activation="relu", name="dense_32")(x)
    outputs = layers.Dense(1, name="occupancy_output")(x)

    model = keras.Model(inputs, outputs, name=name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss="huber",
        metrics=["mae"],
    )
    return model


def build_cnn_bilstm_attention(input_shape,
                               conv_filters=16,
                               bilstm_units=48,
                               attn_units=32,
                               dropout=0.2,
                               lr=1e-3):
    """CNN-BiLSTM-Attention modeli."""
    return _build_attention_model(
        input_shape,
        layers.Bidirectional(layers.LSTM(bilstm_units, return_sequences=True),
                             name="bilstm"),
        "cnn_bilstm_attention", conv_filters, attn_units, dropout, lr,
    )


def build_cnn_lstm_attention(input_shape,
                             conv_filters=16,
                             lstm_units=64,
                             attn_units=32,
                             dropout=0.2,
                             lr=1e-3):
    """CNN-LSTM-Attention modeli."""
    return _build_attention_model(
        input_shape,
        layers.LSTM(lstm_units, return_sequences=True, name="lstm"),
        "cnn_lstm_attention", conv_filters, attn_units, dropout, lr,
    )


def get_attention_model(model):
    """Eğitilmiş attention modelinden zaman adımı ağırlıklarını döndüren alt model."""
    return keras.Model(model.inputs,
                       model.get_layer("attention_weights").output)


def get_callbacks(patience=10):
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5,
            min_lr=1e-6, verbose=1,
        ),
    ]
