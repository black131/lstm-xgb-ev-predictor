"""
train_hybrid_models.py
======================
"""

import argparse
import json
import os
import time
from types import SimpleNamespace

os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import numpy as np
import pandas as pd
import keras

from config import WINDOW_SIZE, BATCH_SIZE, EPOCHS, OUTPUT_DIR
from src import pipeline, models, models_hybrid as hybrid, evaluate, pricing
from src import comparison as cmp

MODEL_BUILDERS = {
    "Persistence": lambda shape, d, seed: hybrid.PersistenceModel(d.scaler, d.target_idx),
    "LSTM_baseline": lambda shape, d, seed: hybrid.KerasModel(models.build_lstm_baseline(shape)),
    "XGBoost_only": lambda shape, d, seed: hybrid.XGBoostModel(shape, d.feature_names, d.target_idx, seed),
    "LSTM_XGBoost": lambda shape, d, seed: hybrid.LSTMXGBoostHybrid(shape, d.feature_names, d.target_idx, seed),
    "TCN_only": lambda shape, d, seed: hybrid.KerasModel(hybrid.build_tcn_only(shape)),
    "TCN_LSTM": lambda shape, d, seed: hybrid.KerasModel(hybrid.build_tcn_lstm(shape)),
}
EXPERIMENTS = [
    ("Referans: Persistence (ŷ(t+1) = y(t))", ["Persistence"], []),
    ("Deney A: LSTM + XGBoost", ["LSTM_baseline", "XGBoost_only", "LSTM_XGBoost"],
     [("LSTM_XGBoost", "LSTM_baseline"), ("LSTM_XGBoost", "XGBoost_only"),
      ("LSTM_XGBoost", "Persistence")]),
    ("Deney B: TCN + LSTM", ["TCN_only", "TCN_LSTM"],
     [("TCN_LSTM", "TCN_only"), ("TCN_LSTM", "LSTM_baseline"), ("TCN_LSTM", "Persistence")]),
]
FINAL_PAIR = ("LSTM_XGBoost", "TCN_LSTM")
# LSTM_XGBoost'un LSTM aşaması LSTM_baseline ile aynı olduğundan eğri tekrar çizilmez
LEARNING_CURVE_SKIP = {"LSTM_XGBoost"}
TARGET_DIRS = {"level": os.path.join(OUTPUT_DIR, "hybrid"),
               "delta": os.path.join(OUTPUT_DIR, "hybrid_delta")}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=str, default=None,
                   help="Hedef grid id (örn. 102). Verilmezse en yoğun grid seçilir.")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--window", type=int, default=WINDOW_SIZE)
    p.add_argument("--models", type=str, default=",".join(MODEL_BUILDERS),
                   help="Virgülle ayrılmış model listesi. Seçenekler: " + ", ".join(MODEL_BUILDERS))
    p.add_argument("--seeds", type=str, default="42,43,44", help="Virgülle ayrılmış seed listesi.")
    p.add_argument("--target", choices=list(TARGET_DIRS), default="delta",
                   help="delta: ŷ = y(t) + Δ̂ (varsayılan) | level: doğrudan y(t+1).")
    p.add_argument("--plots-only", action="store_true",
                   help="Eğitim yapmadan, kayıtlı sonuçlardan tablo ve grafikleri yeniden üret.")
    p.add_argument("--fresh", action="store_true",
                   help="Kayıtlı sonuçlarla birleştirme; yalnızca bu koşunun modellerini raporla.")
    args = p.parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in args.models if m not in MODEL_BUILDERS]
    if unknown:
        p.error(f"Bilinmeyen model(ler): {unknown}")
    args.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    return args


def _paths(grid, target):
    out_dir = TARGET_DIRS[target]
    stem = lambda name: os.path.join(out_dir, f"{name}_grid{grid}")
    return {
        "dir": out_dir,
        "csv": stem("hybrid_comparison") + ".csv",
        "summary_csv": stem("hybrid_summary") + ".csv",
        "summary_md": stem("hybrid_summary") + ".md",
        "preds": stem("predictions") + ".npz",
        "histories": stem("histories") + ".json",
        "importance": stem("xgb_importance") + ".csv",
        "stem": stem,
    }


# ---------------------------------------------------------------------
# Eğitim
# ---------------------------------------------------------------------
def train_one(name, seed, data, epochs, verbose_summary, target):
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    model = MODEL_BUILDERS[name](data.X_train.shape[1:], data, seed)
    if target == "delta" and name != "Persistence":
        model = hybrid.DeltaTarget(model, data.scaler, data.target_idx)
    if verbose_summary:
        model.summary()

    start = time.time()
    history = model.fit(data.X_train, data.y_train, data.X_val, data.y_val, epochs, BATCH_SIZE)
    train_sec = time.time() - start

    y_pred = model.predict(data.X_test)
    metrics = evaluate.regression_metrics(data.y_test, y_pred)
    evaluate.print_metrics_table(f"{name} (seed={seed})", metrics)
    row = {"model": name, "seed": seed, **metrics,
           "epochs_run": len(history.history["loss"]) if history else np.nan,
           "params": model.count_params(),
           "n_trees": getattr(model, "n_trees", np.nan),
           "train_sec": round(train_sec, 1)}
    return model, history, y_pred, row


def run_experiments(args, data, paths):
    rows, preds, histories, importance = [], {}, {}, None
    for title, exp_models, _ in EXPERIMENTS:
        selected = [m for m in exp_models if m in args.models]
        if not selected:
            continue
        print(f"\n{'#' * 70}\n  {title}\n{'#' * 70}")
        for name in selected:
            for i, seed in enumerate(args.seeds):
                print(f"\n{'=' * 60}\n  {name}  (seed={seed})\n{'=' * 60}")
                model, history, y_pred, row = train_one(name, seed, data, args.epochs, i == 0, args.target)
                rows.append(row)
                preds[f"{name}__{seed}"] = y_pred
                if i == 0:   # ilk seed'in modeli, eğrisi ve önem skorları kaydedilir
                    model.save(paths["stem"](name))
                    if history is not None:
                        histories[name] = history.history
                    if name == "LSTM_XGBoost":
                        importance = model.feature_importance()

    df = pd.DataFrame(rows)
    if not args.fresh:
        df, preds, histories, importance = _merge_previous(
            paths, data.y_test, df, preds, histories, importance, set(df["model"]))
    df.to_csv(paths["csv"], index=False)
    np.savez(paths["preds"], y_test=data.y_test,
             test_index=data.test_index.values.astype("datetime64[ns]").astype(np.int64), **preds)
    with open(paths["histories"], "w", encoding="utf-8") as f:
        json.dump({k: {m: [float(v) for v in vals] for m, vals in h.items()}
                   for k, h in histories.items()}, f)
    if importance is not None:
        importance.rename("gain").to_csv(paths["importance"])
    return df


def _merge_previous(paths, y_test, df, preds, histories, importance, rerun):
    """Bir model alt kümesi yeniden koşulduğunda, kayıtlı diğer modellerin
    sonuçlarını korur (yeniden koşulan modellerin eski satırları atılır)."""
    if not os.path.exists(paths["csv"]) or not os.path.exists(paths["preds"]):
        return df, preds, histories, importance
    old_df, old_y, _, old_preds, old_hist, old_imp = load_results(paths)
    if old_y.shape != y_test.shape or not np.allclose(old_y, y_test):
        print("  [UYARI] Kayıtlı sonuçlar farklı bir test setine ait; birleştirilmedi.")
        return df, preds, histories, importance
    kept = sorted(set(old_df["model"]) - rerun)
    if kept:
        print(f"  Önceki sonuçlar korunuyor: {kept}")
    df = pd.concat([old_df[old_df["model"].isin(kept)], df], ignore_index=True)
    order = {m: i for i, m in enumerate(MODEL_BUILDERS)}
    df = df.sort_values("model", key=lambda s: s.map(order), kind="stable").reset_index(drop=True)
    preds = {**{k: v for k, v in old_preds.items() if k.split("__")[0] in kept}, **preds}
    histories = {**{k: v for k, v in old_hist.items() if k in kept}, **histories}
    if importance is None and "LSTM_XGBoost" in kept:
        importance = old_imp
    return df, preds, histories, importance


def load_results(paths):
    df = pd.read_csv(paths["csv"])
    npz = np.load(paths["preds"])
    preds = {k: npz[k] for k in npz.files if "__" in k}
    histories = {}
    if os.path.exists(paths["histories"]):
        with open(paths["histories"], encoding="utf-8") as f:
            histories = json.load(f)
    importance = None
    if os.path.exists(paths["importance"]):
        importance = pd.read_csv(paths["importance"], index_col=0)["gain"]
    return df, npz["y_test"], pd.to_datetime(npz["test_index"]), preds, histories, importance


# ---------------------------------------------------------------------
# Raporlama
# ---------------------------------------------------------------------
def _print_dm(y_test, preds, seeds, a, b):
    """Seed başına Diebold-Mariano: |e| (MAE ile tutarlı) ve e² (RMSE ile tutarlı) kaybı."""
    for seed in seeds:
        pa, pb = preds.get(f"{a}__{seed}"), preds.get(f"{b}__{seed}")
        if pa is None or pb is None:
            continue
        parts = []
        for power, loss in ((1, "|e|"), (2, "e²")):
            dm, p = cmp.diebold_mariano(y_test, pa, pb, power=power)
            better = a if dm < 0 else b
            sig = "p<0.05" if p < 0.05 else "anlamlı değil"
            parts.append(f"{loss}: DM={dm:+.2f}, p={p:.3f} → {better} ({sig})")
        print(f"    DM seed={seed}  " + "  |  ".join(parts))

def report(df, y_test, test_index, preds, histories, importance, paths, grid):
    seeds = sorted(df["seed"].unique())
    first_seed = int(df["seed"].iloc[0])
    summary = cmp.summarize(df)
    flat = summary.copy()
    flat.columns = [f"{m}_{s}" for m, s in flat.columns]
    flat.to_csv(paths["summary_csv"])

    print(f"\n{'=' * 90}\n  ÖZET: Grid {grid}, {len(seeds)} seed ({seeds}) — ortalama ± std\n{'=' * 90}")
    print(cmp.write_markdown_table(summary, paths["summary_md"], len(seeds)))

    for title, exp_models, comparisons in EXPERIMENTS:
        present = [m for m in exp_models if m in summary.index]
        if not present:
            continue
        print(f"\n--- {title} ---")
        evaluate.print_comparison_table(
            {n: {m: summary.loc[n, (m, "mean")] for m in cmp.METRICS} for n in present})
        for name, ref in comparisons:
            if name in summary.index and ref in summary.index:
                print(cmp.improvement_report(summary, name, ref))
                _print_dm(y_test, preds, seeds, name, ref)

    a, b = FINAL_PAIR
    winner = None
    if a in summary.index and b in summary.index:
        print(f"\n{'=' * 70}\n  FİNAL: {a} vs {b}\n{'=' * 70}")
        print(cmp.improvement_report(summary, a, b))
        _print_dm(y_test, preds, seeds, a, b)
        winner = cmp.best_model(summary, "MAE", candidates=FINAL_PAIR)
        print(f"\n  >>> Ortalama MAE'ye göre kazanan mimari: {winner}")
    print(f"  >>> Tüm modeller içinde en düşük MAE: {cmp.best_model(summary, 'MAE')}")

    regimes = None
    if f"Persistence__{first_seed}" in preds:   # persistence tahmini = y(t) → rejim maskesi
        regimes = cmp.regime_mae(y_test, preds[f"Persistence__{first_seed}"], preds,
                                 list(summary.index), seeds)
        regime_md = cmp.regime_markdown(regimes)
        print(f"\n--- Rejime göre MAE (seed ortalaması) ---\n{regime_md}")
        with open(paths["summary_md"], "a", encoding="utf-8") as f:
            f.write("\n### Rejime göre MAE\n\n" + regime_md)

    # --- Grafikler (ilk seed'in tahminleriyle; tablolar tüm seed'lerin ortalaması) ---
    out = lambda n: os.path.join(paths["dir"], f"{n}_grid{grid}.png")
    seed_preds = {n: preds[f"{n}__{first_seed}"] for n in summary.index if f"{n}__{first_seed}" in preds}
    final_preds = {n: seed_preds[n] for n in FINAL_PAIR if n in seed_preds}
    suffix = f"Grid {grid}"

    cmp.plot_metric_bars(summary, out("metrics_bar"),
                         f"Test metrikleri — {suffix} ({len(seeds)} seed ortalaması ± std)")
    cmp.plot_efficiency(summary, out("efficiency"), f"Doğruluk vs eğitim maliyeti — {suffix}")
    cmp.plot_scatter_true_vs_pred(y_test, seed_preds, out("scatter"),
                                  f"Gerçek vs tahmin — {suffix} (seed={first_seed})")
    cmp.plot_error_distribution(y_test, seed_preds, out("error_distribution"),
                                f"Hata dağılımları — {suffix} (seed={first_seed})")
    cmp.plot_hourly_mae(y_test, seed_preds, test_index, out("hourly_mae"),
                        f"Saatlik MAE — {suffix} (seed={first_seed})")
    if regimes is not None:
        cmp.plot_regime_mae(regimes, out("regime_mae"),
                            f"Rejime göre MAE — {suffix} ({len(seeds)} seed ortalaması)")
    if final_preds:
        cmp.plot_prediction_overlay(y_test, final_preds, test_index, out("predictions_final"),
                                    f"{' vs '.join(final_preds)} — {suffix} (seed={first_seed})")
    if importance is not None:
        cmp.plot_xgb_importance(importance, out("xgb_importance"),
                                f"LSTM_XGBoost — XGBoost özellik önemi — {suffix}")
    curves = {n: SimpleNamespace(history=h) for n, h in histories.items() if n not in LEARNING_CURVE_SKIP}
    if curves:
        evaluate.plot_learning_curves(curves, out("learning_curves_hybrid"), colors=cmp.MODEL_COLORS)
    return winner


def run_pricing(data, y_pred, model_name):
    print(f"\n=== Dinamik fiyatlandırma simülasyonu ({model_name} tahminiyle) ===")
    result = pricing.evaluate_pricing_policy(
        predicted_occupancy=y_pred,
        actual_volume=data.test_df.loc[data.test_index, "volume"].values,
        base_price=data.test_df.loc[data.test_index, "price"].values,
        capacity=data.capacity,
    )
    for k, v in result.items():
        print(f"  {k:20s}: {v:.3f}")


def main():
    args = parse_args()
    os.makedirs(TARGET_DIRS[args.target], exist_ok=True)

    if args.plots_only:
        grid = args.grid
        if grid is None:
            raise SystemExit("--plots-only için --grid verilmelidir.")
        paths = _paths(grid, args.target)
        report(*load_results(paths), paths, grid)
        return

    data = pipeline.prepare_data(args.grid, args.window)
    grid = data.target_grid
    paths = _paths(grid, args.target)
    print(f"Girdi şekli (input_shape): {data.X_train.shape[1:]}  |  "
          f"TCN receptive field: {hybrid.tcn_receptive_field()} adım  |  hedef: {args.target}")

    run_experiments(args, data, paths)
    df, y_test, test_index, preds, histories, importance = load_results(paths)
    winner = report(df, y_test, test_index, preds, histories, importance, paths, grid)
    if winner:
        run_pricing(data, preds[f"{winner}__{args.seeds[0]}"], winner)
    print(f"\nTüm çıktılar '{paths['dir']}' klasörüne kaydedildi.")


if __name__ == "__main__":
    main()
