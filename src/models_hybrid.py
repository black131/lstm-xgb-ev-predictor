"""
src/models_hybrid.py
--------------------
LSTM+XGBoost (out-of-fold stacking) ve TCN+LSTM hibrit modelleri.
"""

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from xgboost import XGBRegressor

from src import models
from src.models_gru import _compile, _regression_head


# =====================================================================
# Tabular özellikler (XGBoost girdisi)
# =====================================================================
WINDOW_LAGS = (1, 2, 3, 6, 12, 24)


def make_tabular_features(X, feature_names, target_idx) -> pd.DataFrame:
    """(n, window, F) pencere dizisini ağaç modelinin kullanabileceği 2D tabloya çevirir.

    - Son gözlenen adımın tüm F özelliği          (last_<özellik>)
    - Hedefin pencere içi gecikmeleri               (occ_lag_w<k>: k adım önce)
    - Hedefin pencere istatistikleri                (mean/std/min/max/eğim/son fark)
    """
    window = X.shape[1]
    occ = X[:, :, target_idx]
    cols = {f"last_{name}": X[:, -1, i] for i, name in enumerate(feature_names)}

    for k in WINDOW_LAGS + (window - 1,):
        if k < window:
            cols[f"occ_lag_w{k}"] = occ[:, -1 - k]

    t = np.arange(window) - (window - 1) / 2
    cols["occ_win_mean"] = occ.mean(axis=1)
    cols["occ_win_std"] = occ.std(axis=1)
    cols["occ_win_min"] = occ.min(axis=1)
    cols["occ_win_max"] = occ.max(axis=1)
    cols["occ_win_slope"] = (occ * t).sum(axis=1) / (t ** 2).sum()
    cols["occ_diff_1"] = occ[:, -1] - occ[:, -2]
    return pd.DataFrame(cols)


def booster_importance(booster) -> pd.Series:
    """Özellik başına TOPLAM gain. ("gain" bölünme başına ortalamadır; az ama
    büyük bölünme yapan özelliği şişirir, katkı payı için uygun değildir.)"""
    gain = booster.get_score(importance_type="total_gain")
    return pd.Series(gain).sort_values(ascending=False)


def _xgb_regressor(seed):
    return XGBRegressor(
        n_estimators=2000, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        tree_method="hist", early_stopping_rounds=100,
        random_state=seed, n_jobs=-1,
    )


# =====================================================================
# Referans: Persistence (naif) tahmin
# =====================================================================
class PersistenceModel:
    """ŷ(t+1) = y(t). Eğitim gerektirmez.

    5 dk ileri tahminde doluluk çoğu adımda değişmez; bu yüzden her modelin
    gerçekten "öğrenip öğrenmediği" bu alt sınıra göre değerlendirilmelidir.
    X ölçeklenmiş olduğundan hedef sütun scaler parametreleriyle geri çevrilir.
    """

    def __init__(self, scaler, target_idx):
        self.mean = scaler.mean_[target_idx]
        self.scale = scaler.scale_[target_idx]
        self.target_idx = target_idx

    def fit(self, *args, **kwargs):
        return None

    def predict(self, X):
        return X[:, -1, self.target_idx] * self.scale + self.mean

    def count_params(self):
        return 0

    def summary(self):
        print("PersistenceModel: ŷ(t+1) = y(t)")

    def save(self, path_stem):
        pass


class DeltaTarget:
    """Fark (delta) hedefi sarmalayıcısı: iç model Δ = y(t+1) − y(t) öğrenir,
    tahmin ŷ = y(t) + Δ̂ olur.

    5 dk ileride doluluk çoğu adımda değişmez; seviye yerine değişimi öğrenen
    model, durağan adımlarda Δ̂ ≈ 0 üreterek persistence kadar isabetli kalır ve
    kapasitesini gerçek geçişlere ayırır. Metrikler yine seviye üzerinde hesaplanır.
    Diğer tüm öznitelikler (save, count_params, n_trees, ...) iç modele devredilir.
    """

    def __init__(self, model, scaler, target_idx):
        self.model = model
        self.last = PersistenceModel(scaler, target_idx).predict

    def fit(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        return self.model.fit(X_train, y_train - self.last(X_train),
                              X_val, y_val - self.last(X_val), epochs, batch_size)

    def predict(self, X):
        return self.last(X) + self.model.predict(X)

    def __getattr__(self, name):
        return getattr(self.model, name)


# =====================================================================
# Ortak arayüz: Keras modelleri
# =====================================================================
class KerasModel:
    def __init__(self, keras_model, verbose=2):
        self.model = keras_model
        self.verbose = verbose

    def fit(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        return self.model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size,
            callbacks=models.get_callbacks(), verbose=self.verbose,
        )

    def predict(self, X):
        return self.model.predict(X, verbose=0).flatten()

    def count_params(self):
        return self.model.count_params()

    def summary(self):
        self.model.summary()

    def save(self, path_stem):
        self.model.save(f"{path_stem}.keras")


# =====================================================================
# 1) XGBoost_only ve LSTM + XGBoost
# =====================================================================
class XGBoostModel:
    """Ablation: yalnızca tabular özelliklerle XGBoost (LSTM embedding'i yok)."""

    def __init__(self, input_shape, feature_names, target_idx, seed):
        self.feature_names = feature_names
        self.target_idx = target_idx
        self.xgb = _xgb_regressor(seed)

    def _features(self, X):
        return make_tabular_features(X, self.feature_names, self.target_idx)

    def fit(self, X_train, y_train, X_val, y_val, epochs=None, batch_size=None):
        self.xgb.fit(self._features(X_train), y_train,
                     eval_set=[(self._features(X_val), y_val)], verbose=False)
        print(f"  XGBoost: best_iteration={self.xgb.best_iteration}")
        return None

    def predict(self, X):
        return self.xgb.predict(self._features(X))

    def count_params(self):
        return 0                                   # sinir ağı parametresi yok

    @property
    def n_trees(self):
        return int(self.xgb.best_iteration + 1)

    def summary(self):
        print(f"{type(self).__name__}: {self.xgb.get_params()}")

    def feature_importance(self) -> pd.Series:
        return booster_importance(self.xgb.get_booster())

    def save(self, path_stem):
        self.xgb.save_model(f"{path_stem}_xgb.json")


OOF_FOLDS = 4


def blocked_folds(n, k, purge):
    """n örneği k ardışık zaman bloğuna böler. Her blok için (eğitim, blok)
    indekslerini döndürür. Bloğun iki yanındaki `purge` örnek eğitimden çıkarılır:
    kayan pencereler örtüştüğü için komşu örnekler bloğun verisini içerir."""
    for block in np.array_split(np.arange(n), k):
        lo, hi = block[0] - purge, block[-1] + purge
        train = np.r_[0:max(lo, 0), min(hi + 1, n):n]
        yield train, block


class LSTMXGBoostHybrid(XGBoostModel):
    """LSTM + XGBoost out-of-fold stacking hibrit modeli."""

    def __init__(self, input_shape, feature_names, target_idx, seed):
        super().__init__(input_shape, feature_names, target_idx, seed)
        self.input_shape = input_shape
        self.seed = seed
        self.lstm = models.build_lstm_baseline(input_shape)

    def _features(self, X, lstm_pred=None):
        if lstm_pred is None:
            lstm_pred = self.lstm.predict(X, verbose=0, batch_size=1024).flatten()
        tab = super()._features(X)
        tab.insert(0, "lstm_pred", lstm_pred)
        return tab

    def _oof_predictions(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        n, window = len(X_train), X_train.shape[1]
        oof = np.full(n, np.nan)
        for k, (tr, blk) in enumerate(blocked_folds(n, OOF_FOLDS, purge=window)):
            keras.utils.set_random_seed(self.seed * 10 + k)
            fold = KerasModel(models.build_lstm_baseline(self.input_shape), verbose=0)
            h = fold.fit(X_train[tr], y_train[tr], X_val, y_val, epochs, batch_size)
            oof[blk] = fold.predict(X_train[blk])
            print(f"  OOF fold {k + 1}/{OOF_FOLDS}: blok [{blk[0]}, {blk[-1]}], "
                  f"eğitim {len(tr)} örnek, {len(h.history['loss'])} epoch, "
                  f"blok MAE={np.abs(oof[blk] - y_train[blk]).mean():.4f}")
        assert not np.isnan(oof).any(), "Her train örneği tam bir OOF tahmini almalı"
        return oof

    def fit(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        history = KerasModel(self.lstm).fit(X_train, y_train, X_val, y_val, epochs, batch_size)
        oof = self._oof_predictions(X_train, y_train, X_val, y_val, epochs, batch_size)
        val_pred = self.lstm.predict(X_val, verbose=0, batch_size=1024).flatten()
        print(f"  LSTM OOF train MAE={np.abs(oof - y_train).mean():.4f} | "
              f"tam LSTM val MAE={np.abs(val_pred - y_val).mean():.4f}")
        self.xgb.fit(self._features(X_train, oof), y_train,
                     eval_set=[(self._features(X_val, val_pred), y_val)], verbose=False)
        print(f"  XGBoost: best_iteration={self.xgb.best_iteration}")
        return history

    def count_params(self):
        return self.lstm.count_params()

    def summary(self):
        self.lstm.summary()

    def save(self, path_stem):
        self.lstm.save(f"{path_stem}_lstm.keras")
        super().save(path_stem)


TCN_DILATIONS = (1, 2, 4, 8, 16)
TCN_NORM = "layer"


def tcn_receptive_field(kernel=3, dilations=TCN_DILATIONS, convs_per_block=2):
    return 1 + convs_per_block * (kernel - 1) * sum(dilations)


def _norm_layer(norm, name):
    if norm == "layer":
        return layers.LayerNormalization(name=name)
    if norm == "batch":
        return layers.BatchNormalization(name=name)
    return None


def _tcn_block(x, filters, kernel, dilation, dropout, idx, norm):
    """Residual TCN bloğu."""
    shortcut = x
    for j in (1, 2):
        x = layers.Conv1D(filters, kernel, padding="causal", dilation_rate=dilation,
                          name=f"tcn{idx}_conv{j}")(x)
        norm_layer = _norm_layer(norm, f"tcn{idx}_norm{j}")
        if norm_layer is not None:
            x = norm_layer(x)
        x = layers.Activation("relu", name=f"tcn{idx}_relu{j}")(x)
        x = layers.SpatialDropout1D(dropout, name=f"tcn{idx}_drop{j}")(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, name=f"tcn{idx}_proj")(shortcut)
    x = layers.Add(name=f"tcn{idx}_add")([shortcut, x])
    return layers.Activation("relu", name=f"tcn{idx}_out")(x)


def _tcn_stack(inputs, filters, kernel, dilations, dropout, norm):
    x = inputs
    for idx, d in enumerate(dilations, start=1):
        x = _tcn_block(x, filters, kernel, d, dropout, idx, norm)
    return x


def build_tcn_only(input_shape, filters=32, kernel=3, dilations=TCN_DILATIONS,
                   tcn_dropout=0.1, dropout=0.2, lr=1e-3, norm=TCN_NORM):
    inputs = keras.Input(shape=input_shape, name="input_window")
    h = _tcn_stack(inputs, filters, kernel, dilations, tcn_dropout, norm)
    x = layers.Cropping1D((input_shape[0] - 1, 0), name="last_step")(h)
    x = layers.Flatten(name="last_step_flat")(x)
    outputs = _regression_head(x, dropout)
    return _compile(keras.Model(inputs, outputs, name="tcn_only"), lr)


def build_tcn_lstm(input_shape, filters=32, kernel=3, dilations=TCN_DILATIONS,
                   lstm_units=64, tcn_dropout=0.1, dropout=0.2, lr=1e-3, norm=TCN_NORM):
    """TCN-LSTM hibrit modeli."""
    inputs = keras.Input(shape=input_shape, name="input_window")
    h = _tcn_stack(inputs, filters, kernel, dilations, tcn_dropout, norm)
    x = layers.LSTM(lstm_units, name="lstm")(h)
    outputs = _regression_head(x, dropout)
    return _compile(keras.Model(inputs, outputs, name="tcn_lstm"), lr)
