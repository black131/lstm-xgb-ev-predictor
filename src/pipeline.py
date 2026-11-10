"""
src/pipeline.py
---------------
Veri yükleme, temizleme, özellik çıkarımı, kronolojik bölme ve ölçekleme hattı.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import WINDOW_SIZE, HORIZON, TARGET_COLUMN, TRAIN_DAYS, VAL_DAYS, TEST_DAYS
from src import data_loader, preprocessing, features, windowing


@dataclass
class PreparedData:
    target_grid: str
    capacity: float
    feature_names: list
    target_idx: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_df: pd.DataFrame
    test_index: pd.DatetimeIndex   # y_test'in her elemanına karşılık gelen zaman damgası
    scaler: StandardScaler         # X_* dizilerini ölçeklemek için kullanılan (yalnız train ile fit)


def prepare_data(grid=None, window=WINDOW_SIZE) -> PreparedData:
    print("Veriler yükleniyor...")
    occupancy = preprocessing.clean_grid_table(data_loader.load_grid_table("occupancy"))
    volume = preprocessing.clean_grid_table(data_loader.load_grid_table("volume"))
    duration = preprocessing.clean_grid_table(data_loader.load_grid_table("duration"))
    price = preprocessing.clean_grid_table(data_loader.load_grid_table("price"))
    information = data_loader.load_information()
    adjacency = data_loader.load_adjacency()
    distance = data_loader.load_distance()
    weather = data_loader.load_weather()

    target_grid = grid or information["count"].astype(float).idxmax()
    capacity = float(information.loc[target_grid, "count"])
    print(f"Hedef grid: {target_grid}  (kapasite={capacity} soket)")

    feature_frame = features.build_feature_frame(
        target_grid, occupancy, volume, duration, price,
        features.neighbor_average(occupancy, adjacency), weather,
        features.add_calendar_features(occupancy.index),
        capacity=capacity, distance=distance, top_k=3,
    )

    train_df, val_df, test_df = preprocessing.chronological_split(
        feature_frame, TRAIN_DAYS, VAL_DAYS, TEST_DAYS)
    X_train, y_train = windowing.make_windows(train_df, TARGET_COLUMN, window, HORIZON)
    X_val, y_val = windowing.make_windows(val_df, TARGET_COLUMN, window, HORIZON)
    X_test, y_test = windowing.make_windows(test_df, TARGET_COLUMN, window, HORIZON)
    print(f"Train/Val/Test örnek sayısı: {len(X_train)}/{len(X_val)}/{len(X_test)}")

    scaler = preprocessing.fit_scaler_on_train(X_train)
    return PreparedData(
        target_grid=target_grid,
        capacity=capacity,
        feature_names=list(feature_frame.columns),
        target_idx=feature_frame.columns.get_loc(TARGET_COLUMN),
        X_train=preprocessing.apply_scaler(scaler, X_train), y_train=y_train,
        X_val=preprocessing.apply_scaler(scaler, X_val), y_val=y_val,
        X_test=preprocessing.apply_scaler(scaler, X_test), y_test=y_test,
        test_df=test_df,
        test_index=test_df.index[window + HORIZON - 1:],
        scaler=scaler,
    )
