"""
train_gru_models.py
====================
"""

import argparse
import os
import time

os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import numpy as np
import pandas as pd
import keras

from config import WINDOW_SIZE, BATCH_SIZE, EPOCHS, OUTPUT_DIR
from src import pipeline, models, models_gru, evaluate

MODEL_BUILDERS = {
    "LSTM_baseline": models.build_lstm_baseline,
    "GRU_baseline": models_gru.build_gru_baseline,
    "LSTM_GRU": models_gru.build_lstm_gru,
    "LSTM_BiGRU": models_gru.build_lstm_bigru,
    "CNN_BiLSTM_Attention": models.build_cnn_bilstm_attention,
}
METRICS = ["MAE", "RMSE", "MAPE_%", "R2"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=str, default=None,
                   help="Hedef grid id (örn. 102). Verilmezse en yoğun grid seçilir.")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--window", type=int, default=WINDOW_SIZE)
    p.add_argument("--models", type=str, default=",".join(MODEL_BUILDERS),
                   help="Virgülle ayrılmış model listesi. Seçenekler: "
                        + ", ".join(MODEL_BUILDERS))
    p.add_argument("--seeds", type=str, default="42,43,44",
                   help="Virgülle ayrılmış seed listesi.")
    args = p.parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in args.models if m not in MODEL_BUILDERS]
    if unknown:
        p.error(f"Bilinmeyen model(ler): {unknown}")
    args.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    return args


def main():
    args = parse_args()
    data = pipeline.prepare_data(args.grid, args.window)
    target_grid = data.target_grid
    X_train, y_train = data.X_train, data.y_train
    X_val, y_val = data.X_val, data.y_val
    X_test, y_test = data.X_test, data.y_test
    input_shape = X_train.shape[1:]
    print(f"Girdi şekli (input_shape): {input_shape}")

    rows = []
    histories = {}

    for name in args.models:
        for i, seed in enumerate(args.seeds):
            print(f"\n{'='*60}\n  {name}  (seed={seed})\n{'='*60}")
            keras.backend.clear_session()
            keras.utils.set_random_seed(seed)

            model = MODEL_BUILDERS[name](input_shape)
            if i == 0:
                model.summary()

            start = time.time()
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=args.epochs,
                batch_size=BATCH_SIZE,
                callbacks=models.get_callbacks(),
                verbose=2,
            )
            train_sec = time.time() - start

            y_pred = model.predict(X_test, verbose=0).flatten()
            metrics = evaluate.regression_metrics(y_test, y_pred)
            evaluate.print_metrics_table(f"{name} (seed={seed})", metrics)

            rows.append({"model": name, "seed": seed, **metrics,
                         "epochs_run": len(history.history["loss"]),
                         "params": model.count_params(),
                         "train_sec": round(train_sec, 1)})

            # İlk seed'in modeli, tahmin grafiği ve eğitim eğrisi kaydedilir
            if i == 0:
                histories[name] = history
                evaluate.plot_predictions(
                    y_test, y_pred, f"{name} - Grid {target_grid}",
                    os.path.join(OUTPUT_DIR, f"{name}_predictions_grid{target_grid}.png"))
                model.save(os.path.join(OUTPUT_DIR, f"{name}_grid{target_grid}.keras"))

    # ---------------------------------------------------------------
    # SONUÇLAR: seed bazında CSV + ortalama ± std özet
    # ---------------------------------------------------------------
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, f"gru_comparison_grid{target_grid}.csv")
    df.to_csv(csv_path, index=False)

    summary = df.groupby("model", sort=False)[METRICS + ["params", "train_sec"]].agg(["mean", "std"])
    print(f"\n{'='*90}")
    print(f"  ÖZET: Grid {target_grid}, {len(args.seeds)} seed ({args.seeds}) — ortalama ± std")
    print(f"{'='*90}")
    print(f"{'Model':22s}" + "".join(f"{m:>16s}" for m in METRICS) + f"{'Param':>9s}{'Süre(s)':>9s}")
    for name in summary.index:
        s = summary.loc[name]
        cells = "".join(f"{s[(m, 'mean')]:>9.4f}±{np.nan_to_num(s[(m, 'std')]):<6.4f}" for m in METRICS)
        print(f"{name:22s}{cells}{int(s[('params', 'mean')]):>9d}{s[('train_sec', 'mean')]:>9.1f}")

    evaluate.print_comparison_table(
        {name: {m: summary.loc[name, (m, "mean")] for m in METRICS} for name in summary.index})

    evaluate.plot_learning_curves(
        histories, os.path.join(OUTPUT_DIR, f"learning_curves_gru_grid{target_grid}.png"))
    print(f"\nSeed bazında sonuçlar: {csv_path}")


if __name__ == "__main__":
    main()
