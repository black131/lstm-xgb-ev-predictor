"""
src/data_loader.py
-------------------
Ham veri dosyalarını yükleme ve zaman indeksine hizalama fonksiyonları.
"""

import numpy as np
import pandas as pd
from config import PATHS


def load_time() -> pd.DataFrame:
    """time.csv -> gerçek bir DatetimeIndex üretir (satır sırası = timestamp)."""
    t = pd.read_csv(PATHS["time"])
    t.columns = [c.strip().lower() for c in t.columns]
    dt = pd.to_datetime(
        dict(year=t["year"], month=t["month"], day=t["day"],
             hour=t["hour"], minute=t["minute"], second=t["second"])
    )
    t["datetime"] = dt
    t["timestamp"] = np.arange(1, len(t) + 1)  # occupancy.csv'deki timestamp ile hizalamak için
    return t[["timestamp", "datetime"]]


def load_grid_table(key: str) -> pd.DataFrame:
    """occupancy / volume / duration / price dosyalarından birini yükler.

    Dönen tablo: index = datetime, columns = grid id (string), values = ölçüm.
    """
    df = pd.read_csv(PATHS[key])
    time_df = load_time()
    df = df.merge(time_df, on="timestamp", how="left")
    df = df.set_index("datetime").drop(columns=["timestamp"])
    df.columns = [str(c) for c in df.columns]
    return df.sort_index()


def load_information() -> pd.DataFrame:
    """information.csv -> grid bazlı sabit (statik) öznitelikler."""
    info = pd.read_csv(PATHS["information"])
    info["grid"] = info["grid"].astype(str)
    return info.set_index("grid")


def load_adjacency() -> pd.DataFrame:
    """adj.csv -> 247x247 komşuluk matrisi (0/1), index/columns = grid id (str)."""
    adj = pd.read_csv(PATHS["adjacency"], index_col=0)
    adj.index = adj.index.astype(str)
    adj.columns = [str(c) for c in adj.columns]
    return adj


def load_distance() -> pd.DataFrame:
    """distance.csv -> 247x247 mesafe matrisi (km ya da m), aynı hizalama."""
    dist = pd.read_csv(PATHS["distance"], index_col=0)
    dist.index = dist.index.astype(str)
    dist.columns = [str(c) for c in dist.columns]
    return dist


def load_stations() -> pd.DataFrame:
    """stations.csv -> sadece görselleştirme için, fiziksel istasyon konumları."""
    return pd.read_csv(PATHS["stations"])


def load_weather() -> pd.DataFrame:
    """SZweather...xls -> saatlik hava durumu, 5 dk çözünürlüğe forward-fill ile
    yeniden örneklenmiş şekilde döner (index = datetime, occupancy.csv ile aynı
    aralıkta)."""
    w = pd.read_excel(PATHS["weather"])
    time_col = w.columns[0]  # "Local time in Shenzhen Bao'an (airport)"
    w["datetime"] = pd.to_datetime(w[time_col], dayfirst=True)
    w = w.drop(columns=[time_col]).set_index("datetime").sort_index()

    # Sayısal olmayan / karışık sütunları temizle (VV bazen "10.0 and more" string'i içeriyor)
    if "VV" in w.columns:
        w["VV"] = (
            w["VV"].astype(str).str.extract(r"([\d.]+)").astype(float)
        )
    numeric_cols = ["T", "P0", "P", "U", "Ff", "ff10", "VV", "Td"]
    numeric_cols = [c for c in numeric_cols if c in w.columns]
    w = w[numeric_cols]

    # occupancy.csv'nin tam zaman aralığına 5 dakikalık frekansla genişlet, sonra
    # bir önceki saatin değeriyle doldur (forward-fill) -> saatlik veriyi 5 dk'ya yay.
    full_index = load_time()["datetime"]
    w_resampled = w.reindex(w.index.union(full_index)).sort_index().ffill()
    w_resampled = w_resampled.reindex(full_index)
    w_resampled.index = full_index
    return w_resampled


if __name__ == "__main__":
    # Küçük bir kendi kendini test (smoke test): her yükleyicinin çalıştığını doğrular.
    occ = load_grid_table("occupancy")
    price = load_grid_table("price")
    info = load_information()
    adj = load_adjacency()
    dist = load_distance()
    stations = load_stations()
    weather = load_weather()

    print("occupancy:", occ.shape, occ.index.min(), "-", occ.index.max())
    print("price:", price.shape)
    print("information:", info.shape)
    print("adjacency:", adj.shape)
    print("distance:", dist.shape)
    print("stations:", stations.shape)
    print("weather:", weather.shape, "NaN sayısı:", weather.isna().sum().sum())
    print("\nGrid id eşleşmesi (occupancy vs information):",
          set(occ.columns) == set(info.index))
