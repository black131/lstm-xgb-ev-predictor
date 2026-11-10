"""
src/features.py
----------------
Takvim, hava durumu, komşuluk, gecikme (lag) ve hareketli istatistik özelliklerinin üretimi.
"""

import numpy as np
import pandas as pd


def add_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour = index.hour + index.minute / 60.0
    dow = index.dayofweek  # 0=Pazartesi

    feats = pd.DataFrame(index=index)
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feats["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    feats["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    feats["is_weekend"] = (dow >= 5).astype(float)
    return feats


def neighbor_average(df: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    """Her grid için komşu gridlerin aynı andaki ortalama değerini hesaplar."""
    grids = df.columns.tolist()
    adj = adj.loc[grids, grids]

    adj_values = adj.values.astype(float)
    row_sums = adj_values.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    adj_norm = adj_values / row_sums

    neighbor_avg = df.values @ adj_norm.T
    return pd.DataFrame(neighbor_avg, index=df.index, columns=grids)


def add_lag_features(series: pd.Series, lags: list) -> pd.DataFrame:
    """Verilen zaman serisinin gecikmeli değerlerini üretir."""
    result = pd.DataFrame(index=series.index)
    for lag in lags:
        result[f"occ_lag_{lag}"] = series.shift(lag)
    return result


def add_rolling_features(series: pd.Series, windows: list) -> pd.DataFrame:
    """Hareketli ortalama ve standart sapma hesaplar."""
    result = pd.DataFrame(index=series.index)
    for w in windows:
        result[f"occ_rmean_{w}"] = series.rolling(w, min_periods=1).mean()
        result[f"occ_rstd_{w}"] = series.rolling(w, min_periods=1).std().fillna(0)
    return result


def get_top_k_neighbors(grid_id: str, distance: pd.DataFrame, k: int = 3) -> list:
    """Mesafe matrisine göre en yakın k komşu grid'in ID'sini döndürür."""
    row = distance.loc[grid_id].copy()
    row = row.drop(grid_id, errors="ignore")
    row = row[row > 0]
    return row.nsmallest(k).index.tolist()


def build_feature_frame(grid_id: str,
                         occupancy: pd.DataFrame,
                         volume: pd.DataFrame,
                         duration: pd.DataFrame,
                         price: pd.DataFrame,
                         neighbor_occ_avg: pd.DataFrame,
                         weather: pd.DataFrame,
                         calendar: pd.DataFrame,
                         capacity: float = None,
                         distance: pd.DataFrame = None,
                         top_k: int = 3) -> pd.DataFrame:
    """Hedef grid için tüm özellikleri tek bir tabloda birleştirir."""
    frame = pd.DataFrame(index=occupancy.index)
    frame["occupancy"] = occupancy[grid_id]
    frame["volume"] = volume[grid_id]
    frame["duration"] = duration[grid_id]
    frame["price"] = price[grid_id]
    frame["neighbor_occupancy_avg"] = neighbor_occ_avg[grid_id]

    if capacity is not None and capacity > 0:
        frame["occupancy_rate"] = frame["occupancy"] / capacity

    lags_df = add_lag_features(frame["occupancy"], [288])
    frame = frame.join(lags_df)

    rolling_df = add_rolling_features(frame["occupancy"], [12, 288])
    frame = frame.join(rolling_df)

    if distance is not None:
        top_k_ids = get_top_k_neighbors(grid_id, distance, top_k)
        for i, nid in enumerate(top_k_ids):
            if nid in occupancy.columns:
                frame[f"neighbor_occ_{i+1}"] = occupancy[nid]

    weather_cols = [c for c in ["T", "U", "Ff", "VV", "Td"] if c in weather.columns]
    frame = frame.join(weather[weather_cols])

    frame = frame.join(calendar)

    frame = frame.ffill().fillna(0)
    return frame
