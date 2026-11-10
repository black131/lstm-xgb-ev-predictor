"""
src/preprocessing.py
---------------------
occupancy/volume/duration/price tablolarını temizler ve ZAMAN SIRASINA göre
(rastgele DEĞİL) train/val/test bölümlerine ayırır. Zaman serisi tahmininde
rastgele bölme, gelecekteki bilginin sızmasına (data leakage) yol açar; bu
yüzden kronolojik bölme kullanılıyor.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import STEPS_PER_DAY, TRAIN_DAYS, VAL_DAYS, TEST_DAYS


def clean_grid_table(df: pd.DataFrame) -> pd.DataFrame:
    """Negatif değerleri 0'a çeker, eksik değerleri zaman ekseninde interpolasyonla doldurur."""
    df = df.copy()
    df[df < 0] = 0
    df = df.interpolate(method="linear", limit_direction="both")
    df = df.fillna(0)
    return df


def chronological_split(df: pd.DataFrame,
                         train_days: int = TRAIN_DAYS,
                         val_days: int = VAL_DAYS,
                         test_days: int = TEST_DAYS,
                         steps_per_day: int = STEPS_PER_DAY):
    """Veriyi kronolojik sırayla train / val / test olarak böler."""
    n_train = train_days * steps_per_day
    n_val = val_days * steps_per_day
    n_test = test_days * steps_per_day

    total_needed = n_train + n_val + n_test
    if total_needed > len(df):
        raise ValueError(
            f"train+val+test ({total_needed} satır) mevcut veriden "
            f"({len(df)} satır) fazla. config.py içindeki gün sayılarını azaltın."
        )

    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val: n_train + n_val + n_test]
    return train, val, test


def fit_scaler_on_train(train_array: np.ndarray) -> StandardScaler:
    """Ölçekleyiciyi SADECE eğitim verisiyle eğitir (veri sızıntısını önlemek için)."""
    scaler = StandardScaler()
    scaler.fit(train_array.reshape(-1, train_array.shape[-1]))
    return scaler


def apply_scaler(scaler: StandardScaler, array: np.ndarray) -> np.ndarray:
    orig_shape = array.shape
    flat = array.reshape(-1, orig_shape[-1])
    scaled = scaler.transform(flat)
    return scaled.reshape(orig_shape)
