"""
src/evaluate.py
----------------
Regresyon metrikleri (MAE, RMSE, MAPE, R^2) ve karşılaştırma grafikleri.
"""

 
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
 
 
def _to_flat_array(x, name: str) -> np.ndarray:
    """Girdiyi kesin olarak 1 boyutlu (n,) numpy dizisine çevirir."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1)
        if arr.shape[1] != 1:
            raise ValueError(
                f"'{name}' beklenenden fazla boyutlu: {arr.shape}. "
                "Tek adımlı (HORIZON=1) tahmin bekleniyor; çok adımlıysa "
                "her horizon adımını ayrı ayrı değerlendirin."
            )
        arr = arr.reshape(-1)
    return arr
 
 
def regression_metrics(y_true, y_pred) -> dict:
    y_true = _to_flat_array(y_true, "y_true")
    y_pred = _to_flat_array(y_pred, "y_pred")
 
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true {y_true.shape} ve y_pred {y_pred.shape} boyutları uyuşmuyor. "
        )
 
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
 
    # MAPE: y_true == 0 olan noktalarda bölme hatası olmasın diye dışarıda bırak
    nonzero = y_true != 0
    if nonzero.sum() == 0:
        mape = float("nan")
    else:
        mape = float(
            np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
        )
 
    # sklearn r2_score DAİMA verilen y_true'nun varyansına göre hesaplar --
    # train setinin ya da başka bir dizinin varyansını ASLA kullanmaz.
    r2 = float(r2_score(y_true, y_pred))
 
    metrics = {"MAE": mae, "RMSE": rmse, "MAPE_%": mape, "R2": r2}
    _sanity_check(y_true, metrics)
    return metrics
 
 
def _sanity_check(y_true: np.ndarray, metrics: dict) -> None:
    r2 = metrics["R2"]
    rmse = metrics["RMSE"]
    real_std = float(np.std(y_true))
 
    if r2 < 1.0 and real_std > 0:
        implied_var = (rmse ** 2) / (1 - r2)
        implied_std = np.sqrt(max(implied_var, 0))
        rel_diff = abs(implied_std - real_std) / real_std
        if rel_diff > 0.5:
            print(
                "  [UYARI] R2 ve RMSE birbiriyle TUTARSIZ görünüyor!\n"
                f"           Gerçek std(y_true)        = {real_std:.3f}\n"
                f"           R2/RMSE'den çıkarılan std = {implied_std:.3f}\n"
                "           Bu fonksiyona geçirdiğiniz y_true/y_pred'in, "
                "grafiği çizdiğiniz dizilerle AYNI obje olduğundan emin olun."
            )
 
 
def print_metrics_table(name: str, metrics: dict) -> None:
    print(f"--- {name} ---")
    print(f"  MAE      : {metrics['MAE']:.4f}")
    print(f"  RMSE     : {metrics['RMSE']:.4f}")
    print(f"  MAPE_%   : {metrics['MAPE_%']:.4f}")
    print(f"  R2       : {metrics['R2']:.4f}")


# =====================================================================
# Karşılaştırma tablosu
# =====================================================================
def print_comparison_table(results: dict) -> None:
    """Tüm modellerin metriklerini yan yana tablo olarak yazdırır.
    
    results: {"model_adı": {"MAE": ..., "RMSE": ..., ...}, ...}
    """
    print("\n" + "=" * 65)
    print(f"{'Model':25s} {'MAE':>8s} {'RMSE':>8s} {'MAPE_%':>8s} {'R2':>8s}")
    print("-" * 65)
    
    # En iyi değerleri bul (min MAE/RMSE/MAPE, max R2)
    best = {
        "MAE":    min(m["MAE"]    for m in results.values()),
        "RMSE":   min(m["RMSE"]   for m in results.values()),
        "MAPE_%": min(m["MAPE_%"] for m in results.values()),
        "R2":     max(m["R2"]     for m in results.values()),
    }
    
    for name, metrics in results.items():
        markers = []
        for key in ["MAE", "RMSE", "MAPE_%", "R2"]:
            if metrics[key] == best[key]:
                markers.append(f"  *{metrics[key]:.4f}")
            else:
                markers.append(f"   {metrics[key]:.4f}")
        print(f"  {name:23s}" + "".join(markers))
    
    print("-" * 65)
    print("  * = bu metrikte en iyi model")
    print("=" * 65)

 
def plot_predictions(y_true, y_pred, title: str, save_path: str) -> None:
    """Grafiği, metriklerle TAM OLARAK AYNI (y_true, y_pred) dizilerinden çizer."""
    y_true = _to_flat_array(y_true, "y_true")
    y_pred = _to_flat_array(y_pred, "y_pred")
 
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(y_true, label="Gerçek")
    ax.plot(y_pred, label="Tahmin")
    ax.set_title(title)
    ax.set_xlabel("Zaman adımı (5 dk)")
    ax.set_ylabel("Değer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# =====================================================================
# Learning Curve (eğitim eğrisi) grafikleri
# =====================================================================
def plot_learning_curves(histories: dict, save_path: str, colors: dict = None) -> None:
    """Her modelin eğitim ve doğrulama kayıp eğrilerini karşılaştırmalı çizer.

    histories: {"model_adı": keras History objesi, ...}
    save_path: grafik kayıt yolu
    colors   : isteğe bağlı {"model_adı": renk}; verilmezse varsayılan eşleme
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    colors = colors or {
        "LSTM_baseline": "#1f77b4",
        "CNN_BiLSTM": "#ff7f0e",
        "CNN_LSTM_v2": "#ff7f0e",
        "CNN_LSTM": "#2ca02c",
        "CNN_BiLSTM_Attention": "#d62728",
        "CNN_LSTM_Attention": "#9467bd",
    }
    
    for name, history in histories.items():
        color = colors.get(name, None)
        epochs = range(1, len(history.history["loss"]) + 1)
        
        # Sol: Loss eğrileri
        axes[0].plot(epochs, history.history["loss"],
                     label=f"{name} (train)", color=color, linestyle="-")
        axes[0].plot(epochs, history.history["val_loss"],
                     label=f"{name} (val)", color=color, linestyle="--")
        
        # Sağ: MAE eğrileri
        axes[1].plot(epochs, history.history["mae"],
                     label=f"{name} (train)", color=color, linestyle="-")
        axes[1].plot(epochs, history.history["val_mae"],
                     label=f"{name} (val)", color=color, linestyle="--")
    
    axes[0].set_title("Loss (Huber)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title("MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle("Eğitim Eğrileri Karşılaştırması", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  Learning curves kaydedildi: {save_path}")


# =====================================================================
# Attention ağırlıkları görselleştirmesi
# =====================================================================
def plot_attention_weights(weights, title: str, save_path: str,
                           n_heatmap: int = 200) -> None:
    """Attention modelinin zaman adımlarına verdiği ağırlıkları çizer.

    weights : (n_samples, window) veya (n_samples, window, 1) dizisi
    Üst: test seti boyunca ortalama ağırlık (± std), x ekseni t-window ... t-1
    Alt: ilk n_heatmap test örneği için ısı haritası
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim == 3:
        w = w[..., 0]
    window = w.shape[1]
    lags = np.arange(-window, 0)           # -48 ... -1 (5 dk adımlar)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                             gridspec_kw={"height_ratios": [1, 1.4]})

    mean, std = w.mean(axis=0), w.std(axis=0)
    axes[0].bar(lags, mean, color="#d62728", alpha=0.8)
    axes[0].errorbar(lags, mean, yerr=std, fmt="none", ecolor="gray", alpha=0.5)
    axes[0].axhline(1 / window, color="black", linestyle="--", linewidth=1,
                    label=f"Uniform (1/{window})")
    axes[0].set_title("Ortalama attention ağırlığı (test seti)")
    axes[0].set_xlabel("Geçmiş zaman adımı (5 dk)")
    axes[0].set_ylabel("Ağırlık")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    hm = w[:n_heatmap]
    im = axes[1].imshow(hm, aspect="auto", cmap="viridis",
                        extent=[-window - 0.5, -0.5, len(hm), 0])
    axes[1].set_title(f"Örnek bazında attention ağırlıkları (ilk {len(hm)} test örneği)")
    axes[1].set_xlabel("Geçmiş zaman adımı (5 dk)")
    axes[1].set_ylabel("Test örneği")
    fig.colorbar(im, ax=axes[1], label="Ağırlık")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  Attention grafiği kaydedildi: {save_path}")
