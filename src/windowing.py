"""
src/windowing.py
-----------------
LSTM/CNN girdisi için "kayan pencere" (sliding window) dizileri üretir.

Girdi:  feature_frame -> (zaman x özellik) tablosu (features.build_feature_frame çıktısı)
Çıktı:  X -> (örnek, window, özellik)   ->  modele girdi
        y -> (örnek,)                   ->  hedef (window'dan HORIZON adım sonrası)
"""

import numpy as np
import pandas as pd


def make_windows(feature_frame: pd.DataFrame,
                  target_col: str,
                  window: int,
                  horizon: int = 1):
    values = feature_frame.values.astype(np.float32)
    target_idx = feature_frame.columns.get_loc(target_col)

    n_samples = len(values) - window - horizon + 1
    if n_samples <= 0:
        raise ValueError("Pencere/ufuk boyutuna göre veri yetersiz.")

    n_features = values.shape[1]
    X = np.zeros((n_samples, window, n_features), dtype=np.float32)
    y = np.zeros((n_samples,), dtype=np.float32)

    for i in range(n_samples):
        X[i] = values[i: i + window]
        y[i] = values[i + window + horizon - 1, target_idx]

    return X, y
