"""
train_demand_model.py
======================

"""

import argparse
import os
import random 
import tensorflow as tf
import numpy as np
import sklearn 

from config import (WINDOW_SIZE, HORIZON, TARGET_COLUMN, BATCH_SIZE,
                     EPOCHS, TRAIN_DAYS, VAL_DAYS, TEST_DAYS, OUTPUT_DIR)
from src import data_loader, preprocessing, features, windowing, models, evaluate, pricing

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# GPU'da bazı operasyonların deterministik olmasını zorlar (biraz yavaşlatabilir)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

MODEL_BUILDERS = {
    "LSTM_baseline": models.build_lstm_baseline,
    "CNN_BiLSTM": models.build_cnn_bilstm,
    "CNN_BiLSTM_Attention": models.build_cnn_bilstm_attention,
    "CNN_LSTM_Attention": models.build_cnn_lstm_attention,
}
ATTENTION_MODELS = {"CNN_BiLSTM_Attention", "CNN_LSTM_Attention"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=str, default=None,
                    help="Hedef grid id (örn. 102). Verilmezse en yoğun grid seçilir.")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--window", type=int, default=WINDOW_SIZE)
    p.add_argument("--models", type=str, default=",".join(MODEL_BUILDERS),
                    help="Virgülle ayrılmış model listesi. Seçenekler: "
                         + ", ".join(MODEL_BUILDERS))
    args = p.parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in args.models if m not in MODEL_BUILDERS]
    if unknown:
        p.error(f"Bilinmeyen model(ler): {unknown}")
    return args


def main():
    args = parse_args()

    # ---------------------------------------------------------------
    # 1) VERİ YÜKLEME
    # ---------------------------------------------------------------
    print("Veriler yükleniyor...")
    occupancy = data_loader.load_grid_table("occupancy")
    volume = data_loader.load_grid_table("volume")
    duration = data_loader.load_grid_table("duration")
    price = data_loader.load_grid_table("price")
    information = data_loader.load_information()
    adjacency = data_loader.load_adjacency()
    distance = data_loader.load_distance()
    weather = data_loader.load_weather()

    # ---------------------------------------------------------------
    # 2) TEMİZLEME
    # ---------------------------------------------------------------
    occupancy = preprocessing.clean_grid_table(occupancy)
    volume = preprocessing.clean_grid_table(volume)
    duration = preprocessing.clean_grid_table(duration)
    price = preprocessing.clean_grid_table(price)

    # ---------------------------------------------------------------
    # 3) HEDEF GRID SEÇİMİ
    # ---------------------------------------------------------------
    target_grid = args.grid or information["count"].astype(float).idxmax()
    capacity = float(information.loc[target_grid, "count"])
    print(f"Hedef grid: {target_grid}  (kapasite={capacity} soket)")

    # ---------------------------------------------------------------
    # 4) ÖZELLİK MÜHENDİSLİĞİ (v2: lag + rolling + top-K komşu)
    # ---------------------------------------------------------------
    print("Özellikler oluşturuluyor (komşu gridler + hava durumu + takvim + lag + rolling)...")
    neighbor_avg = features.neighbor_average(occupancy, adjacency)
    calendar = features.add_calendar_features(occupancy.index)

    feature_frame = features.build_feature_frame(
        target_grid, occupancy, volume, duration, price,
        neighbor_avg, weather, calendar,
        capacity=capacity,
        distance=distance,
        top_k=3,
    )
    print(f"Özellik sütunları ({len(feature_frame.columns)}):", list(feature_frame.columns))

    # ---------------------------------------------------------------
    # 5) KRONOLOJİK BÖLME (önce böl sonra pencereleme yap.)
    # ---------------------------------------------------------------
    train_df, val_df, test_df = preprocessing.chronological_split(
        feature_frame, TRAIN_DAYS, VAL_DAYS, TEST_DAYS)

    X_train, y_train = windowing.make_windows(train_df, TARGET_COLUMN, args.window, HORIZON)
    X_val, y_val = windowing.make_windows(val_df, TARGET_COLUMN, args.window, HORIZON)
    X_test, y_test = windowing.make_windows(test_df, TARGET_COLUMN, args.window, HORIZON)

    print(f"Train/Val/Test örnek sayısı: {len(X_train)}/{len(X_val)}/{len(X_test)}")

    # ---------------------------------------------------------------
    # 6) ÖLÇEKLEME (sadece train ile fit)
    # ---------------------------------------------------------------
    scaler = preprocessing.fit_scaler_on_train(X_train)
    X_train_s = preprocessing.apply_scaler(scaler, X_train)
    X_val_s = preprocessing.apply_scaler(scaler, X_val)
    X_test_s = preprocessing.apply_scaler(scaler, X_test)

    input_shape = X_train_s.shape[1:]  # (window, num_features)
    print(f"Girdi şekli (input_shape): {input_shape}")

    # ---------------------------------------------------------------
    # 7) MODEL EĞİTİMİ: LSTM baseline vs CNN-BiLSTM vs Attention modelleri
    # ---------------------------------------------------------------
    results = {}
    predictions = {}
    histories = {}

    for name in args.models:
        builder = MODEL_BUILDERS[name]
        print(f"\n{'='*60}")
        print(f"  {name} eğitiliyor")
        print(f"{'='*60}")
        model = builder(input_shape)
        model.summary()

        history = model.fit(
            X_train_s, y_train,
            validation_data=(X_val_s, y_val),
            epochs=args.epochs,
            batch_size=BATCH_SIZE,
            callbacks=models.get_callbacks(),
            verbose=2,
        )

        y_pred = model.predict(X_test_s, verbose=0).flatten()
        metrics = evaluate.regression_metrics(y_test, y_pred)
        evaluate.print_metrics_table(name, metrics)

        results[name] = metrics
        predictions[name] = y_pred
        histories[name] = history

        if name in ATTENTION_MODELS:
            attn = models.get_attention_model(model).predict(X_test_s, verbose=0)
            evaluate.plot_attention_weights(
                attn, f"{name} - Grid {target_grid}",
                os.path.join(OUTPUT_DIR, f"attention_{name}_grid{target_grid}.png"))

        # Modeli ve grafiği kaydet
        plot_path = os.path.join(OUTPUT_DIR, f"{name}_predictions_grid{target_grid}.png")
        evaluate.plot_predictions(y_test, y_pred, f"{name} - Grid {target_grid}", plot_path)
        model.save(os.path.join(OUTPUT_DIR, f"{name}_grid{target_grid}.keras"))
        print(f"  Model ve grafik kaydedildi: {OUTPUT_DIR}")

    # ---------------------------------------------------------------
    # 7b) KARŞILAŞTIRMA TABLOSU VE LEARNING CURVES
    # ---------------------------------------------------------------
    evaluate.print_comparison_table(results)

    lc_path = os.path.join(OUTPUT_DIR, f"learning_curves_grid{target_grid}.png")
    evaluate.plot_learning_curves(histories, lc_path)

    # İyileşme raporu: her model, kendinden basit referans modellere göre
    comparisons = [("CNN_BiLSTM", "LSTM_baseline"),
                   ("CNN_BiLSTM_Attention", "LSTM_baseline"),
                   ("CNN_BiLSTM_Attention", "CNN_BiLSTM"),
                   ("CNN_LSTM_Attention", "LSTM_baseline"),
                   ("CNN_LSTM_Attention", "CNN_BiLSTM")]
    for name, ref in comparisons:
        if name not in results or ref not in results:
            continue
        d_r2 = results[name]["R2"] - results[ref]["R2"]
        d_mae = results[name]["MAE"] - results[ref]["MAE"]
        print(f"\n  {name} vs {ref}:")
        print(f"    R2  farki: {d_r2:+.4f} ({'[+] IYILESME' if d_r2 > 0 else '[-] Geride'})")
        print(f"    MAE farki: {d_mae:+.4f} ({'[+] IYILESME' if d_mae < 0 else '[-] Geride'})")

    # ---------------------------------------------------------------
    # 8) DİNAMİK FİYATLANDIRMA SİMÜLASYONU (CNN-BiLSTM-Attention tahminiyle)
    # ---------------------------------------------------------------
    pricing_model = next(m for m in ("CNN_BiLSTM_Attention", "CNN_BiLSTM",
                                      "CNN_LSTM_Attention", "LSTM_baseline")
                         if m in predictions)
    print(f"\n=== Dinamik fiyatlandırma simülasyonu ({pricing_model} tahminiyle) ===")
    test_index = test_df.index[args.window + HORIZON - 1:]
    base_price_test = test_df.loc[test_index, "price"].values
    volume_test = test_df.loc[test_index, "volume"].values

    pricing_result = pricing.evaluate_pricing_policy(
        predicted_occupancy=predictions[pricing_model],
        actual_volume=volume_test,
        base_price=base_price_test,
        capacity=capacity,
    )
    for k, v in pricing_result.items():
        print(f"  {k:20s}: {v:.3f}")

    print(f"\nTüm çıktı grafikleri ve modeller '{OUTPUT_DIR}' klasörüne kaydedildi.")


if __name__ == "__main__":
    main()
