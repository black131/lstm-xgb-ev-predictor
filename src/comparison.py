"""
src/comparison.py
-----------------
Karşılaştırma tabloları, grafikler ve Diebold-Mariano istatistiksel anlamlılık testi.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

METRICS = ["MAE", "RMSE", "MAPE_%", "R2"]
LOWER_IS_BETTER = {"MAE": True, "RMSE": True, "MAPE_%": True, "R2": False}

MODEL_COLORS = {
    "LSTM_XGBoost": "#2a78d6",    # 1 mavi
    "TCN_LSTM": "#eb6834",        # 2 turuncu
    "LSTM_baseline": "#1baf7a",   # 3 aqua
    "XGBoost_only": "#eda100",    # 4 sarı
    "TCN_only": "#e87ba4",        # 5 magenta
    "Persistence": "#8a8984",     # nötr gri: kategorik değil, referans çizgisi
}
ACTUAL_COLOR = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
DPI = 120


def _color(name):
    return MODEL_COLORS.get(name, INK_SECONDARY)


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=11, loc="left", color=ACTUAL_COLOR)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SECONDARY)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY)


def _save(fig, save_path):
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"  Grafik kaydedildi: {save_path}")


# =====================================================================
# Tablolar
# =====================================================================
def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Seed bazlı sonuç tablosundan model başına ortalama ve std üretir."""
    cols = METRICS + ["params", "n_trees", "epochs_run", "train_sec"]
    summary = df.groupby("model", sort=False)[cols].agg(["mean", "std"])
    return summary.fillna({c: 0 for c in summary.columns if c[1] == "std"})


def best_model(summary: pd.DataFrame, metric="MAE", candidates=None):
    means = summary[(metric, "mean")]
    if candidates is not None:
        means = means[means.index.isin(candidates)]
    return means.idxmin() if LOWER_IS_BETTER[metric] else means.idxmax()


def write_markdown_table(summary: pd.DataFrame, save_path: str, n_seeds: int) -> str:
    """Ortalama ± std özet tablosunu Markdown formatında kaydeder."""
    best = {m: best_model(summary, m) for m in METRICS}
    header = ("| Model | MAE | RMSE | MAPE (%) | R² | NN param | Ağaç | Süre (s) |\n"
              "|---|---|---|---|---|---|---|---|")
    lines = [header]
    for name in summary.index:
        s = summary.loc[name]
        cells = []
        for m in METRICS:
            txt = f"{s[(m, 'mean')]:.4f} ± {s[(m, 'std')]:.4f}"
            cells.append(f"**{txt}**" if best[m] == name else txt)
        trees = s[("n_trees", "mean")]
        trees_txt = "—" if np.isnan(trees) else f"{trees:.0f}"
        lines.append(f"| {name} | " + " | ".join(cells)
                     + f" | {int(s[('params', 'mean')]):,} | {trees_txt} | {s[('train_sec', 'mean')]:.1f} |")
    table = "\n".join(lines) + f"\n\n_{n_seeds} seed ortalaması ± std; kalın = o metrikte en iyi._\n"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"  Markdown tablo kaydedildi: {save_path}")
    return table


def improvement_report(summary: pd.DataFrame, name: str, ref: str) -> str:
    """name modelinin ref'e göre metrik bazında göreli iyileşmesi (pozitif = daha iyi)."""
    parts = []
    for m in METRICS:
        a, b = summary.loc[name, (m, "mean")], summary.loc[ref, (m, "mean")]
        if m == "R2":
            parts.append(f"R2 {a - b:+.4f}")
        else:
            parts.append(f"{m} {(b - a) / b * 100:+.1f}%")
    return f"  {name} vs {ref}:  " + ",  ".join(parts)


REGIMES = ["Değişmeyen adımlar", "Değişen adımlar", "Genel"]


def regime_mae(y_true, y_last, preds: dict, names, seeds) -> pd.DataFrame:
    """Doluluğun değişmediği (y(t+1) = y(t)) ve değiştiği adımlarda ayrı MAE,
    seed ortalaması. Persistence ilk rejimde tanım gereği 0 hata yapar; modellerin
    asıl katkısı ikinci rejimdedir (fiyatlandırma için kritik geçişler)."""
    changed = np.abs(y_true - y_last) > 1e-6
    rows = {}
    for name in names:
        errs = [np.abs(preds[f"{name}__{s}"] - y_true) for s in seeds if f"{name}__{s}" in preds]
        if errs:
            e = np.mean(errs, axis=0)
            rows[name] = [e[~changed].mean(), e[changed].mean(), e.mean()]
    df = pd.DataFrame.from_dict(rows, orient="index", columns=REGIMES)
    df.attrs["changed_share"] = float(changed.mean())
    return df


def regime_markdown(df: pd.DataFrame) -> str:
    share = df.attrs.get("changed_share", np.nan)
    lines = [f"| Model | MAE: değişmeyen (%{(1 - share) * 100:.0f}) | MAE: değişen (%{share * 100:.0f}) | MAE: genel |",
             "|---|---|---|---|"]
    best = df.idxmin()
    for name, r in df.iterrows():
        cells = [f"**{v:.4f}**" if best[c] == name else f"{v:.4f}" for c, v in r.items()]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# =====================================================================
# İstatistiksel test
# =====================================================================
def diebold_mariano(y_true, pred_a, pred_b, power=2):
    """Diebold-Mariano testi (Harvey-Leybourne-Newbold küçük örnek düzeltmesiyle).

    H0: iki modelin beklenen kaybı eşit. d_t = |e_a|^p - |e_b|^p.
    5 dakikalık seride hatalar güçlü otokorelasyonludur; bu yüzden d_t'nin
    varyansı Newey-West (Bartlett çekirdekli HAC) ile, gecikme L = n^(1/3)
    alınarak tahmin edilir — aksi hâlde anlamlılık abartılır.

    Dönen: (DM istatistiği, iki yönlü p-değeri). DM < 0 → model A daha iyi.
    """
    y_true = np.asarray(y_true, float)
    d = np.abs(y_true - pred_a) ** power - np.abs(y_true - pred_b) ** power
    n = len(d)
    lag = int(np.floor(n ** (1 / 3)))
    d_c = d - d.mean()
    gamma = [np.dot(d_c[k:], d_c[:n - k]) / n for k in range(lag + 1)]
    var = gamma[0] + 2 * sum((1 - k / (lag + 1)) * gamma[k] for k in range(1, lag + 1))
    dm = d.mean() / np.sqrt(var / n)
    h = 1
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(dm), df=n - 1)
    return float(dm), float(p)


# =====================================================================
# Grafikler
# =====================================================================
def plot_metric_bars(summary: pd.DataFrame, save_path: str, title: str):
    """2×2: her metrik için model başına ortalama, seed std hata çubuklu."""
    names = list(summary.index)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    labels = {"MAE": "MAE (soket)", "RMSE": "RMSE (soket)", "MAPE_%": "MAPE (%)", "R2": "R²"}
    for ax, m in zip(axes.flat, METRICS):
        mean = summary[(m, "mean")].values
        std = summary[(m, "std")].values
        x = np.arange(len(names))
        ax.bar(x, mean, width=0.6, color=[_color(n) for n in names],
               edgecolor="white", linewidth=2)
        ax.errorbar(x, mean, yerr=std, fmt="none", ecolor=INK_SECONDARY, capsize=4, linewidth=1)
        best = best_model(summary, m)
        for xi, (n, v, s) in enumerate(zip(names, mean, std)):
            ax.text(xi, v + s, f"{v:.3f}", ha="center", va="bottom", fontsize=9,
                    color=ACTUAL_COLOR, fontweight="bold" if n == best else "normal")
        ax.set_xticks(x, names, rotation=15, fontsize=9)
        if m == "R2":
            lo = max(0.0, (mean - std).min() - 0.02)
            ax.set_ylim(lo, 1.0)
        _style(ax, f"{labels[m]}  ({'düşük' if LOWER_IS_BETTER[m] else 'yüksek'} = iyi)")
    fig.suptitle(title, fontsize=14, fontweight="bold", x=0.01, ha="left")
    _save(fig, save_path)


def _most_dynamic_day(y_true, steps_per_day=288):
    n_days = len(y_true) // steps_per_day
    if n_days == 0:
        return 0, len(y_true)
    stds = [y_true[i * steps_per_day:(i + 1) * steps_per_day].std() for i in range(n_days)]
    i = int(np.argmax(stds))
    return i * steps_per_day, (i + 1) * steps_per_day


def plot_prediction_overlay(y_true, preds: dict, test_index, save_path: str, title: str):
    """Üst: tüm test dönemi; alt: en dalgalı günün yakın görünümü."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), gridspec_kw={"height_ratios": [1, 1.2]})
    s, e = _most_dynamic_day(y_true)
    for ax, sl, sub in [(axes[0], slice(None), "Tüm test dönemi"),
                        (axes[1], slice(s, e), f"Yakın görünüm: en dalgalı gün "
                                               f"({pd.Timestamp(test_index[s]).date()})")]:
        ax.plot(test_index[sl], y_true[sl], color=ACTUAL_COLOR, linewidth=2, label="Gerçek")
        for name, p in preds.items():
            ax.plot(test_index[sl], p[sl], color=_color(name), linewidth=1.6, alpha=0.9, label=name)
        _style(ax, sub, None, "Doluluk (soket)")
    axes[0].legend(loc="upper left", ncol=len(preds) + 1, frameon=False)
    fig.suptitle(title, fontsize=14, fontweight="bold", x=0.01, ha="left")
    _save(fig, save_path)


def plot_scatter_true_vs_pred(y_true, preds: dict, save_path: str, title: str):
    names = list(preds)
    ncols = min(3, len(names))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.4 * nrows + 0.5),
                             sharex=True, sharey=True, squeeze=False)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    lo, hi = float(np.min(y_true)), float(np.max(y_true))
    pad = (hi - lo) * 0.05
    for i, (ax, name) in enumerate(zip(axes.flat, names)):
        p = preds[name]
        ax.scatter(y_true, p, s=8, alpha=0.35, color=_color(name), linewidths=0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK_SECONDARY,
                linestyle="--", linewidth=1)
        r2 = 1 - np.sum((y_true - p) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
        _style(ax, f"{name}  (R²={r2:.3f})",
               "Gerçek" if i >= len(names) - ncols else None,
               "Tahmin" if i % ncols == 0 else None)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
    fig.suptitle(title, fontsize=14, fontweight="bold", x=0.01, ha="left")
    _save(fig, save_path)


def plot_error_distribution(y_true, preds: dict, save_path: str, title: str):
    """Sol: hata (tahmin − gerçek) yoğunluğu; sağ: mutlak hata kutu grafiği."""
    names = list(preds)
    errors = {n: preds[n] - y_true for n in names}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1.3, 1]})

    allerr = np.concatenate(list(errors.values()))
    q = np.quantile(np.abs(allerr), 0.995)
    grid = np.linspace(-q, q, 400)
    for n in names:
        kde = stats.gaussian_kde(errors[n])
        axes[0].plot(grid, kde(grid), color=_color(n), linewidth=2, label=n)
    axes[0].axvline(0, color=INK_SECONDARY, linewidth=1, linestyle="--")
    axes[0].legend(frameon=False)
    _style(axes[0], "Hata dağılımı (tahmin − gerçek)", "Hata (soket)", "Yoğunluk")

    bp = axes[1].boxplot([np.abs(errors[n]) for n in names], vert=False, widths=0.55,
                         patch_artist=True, showfliers=False,
                         medianprops={"color": ACTUAL_COLOR, "linewidth": 1.5})
    for patch, n in zip(bp["boxes"], names):
        patch.set_facecolor(_color(n))
        patch.set_alpha(0.75)
        patch.set_edgecolor(INK_SECONDARY)
    axes[1].set_yticks(range(1, len(names) + 1), names)
    axes[1].invert_yaxis()
    _style(axes[1], "Mutlak hata (aykırı değerler gizli)", "|Hata| (soket)")
    fig.suptitle(title, fontsize=14, fontweight="bold", x=0.01, ha="left")
    _save(fig, save_path)


def plot_hourly_mae(y_true, preds: dict, test_index, save_path: str, title: str):
    """Günün saatine göre MAE — dinamik fiyatlandırmanın kritik saatlerinde hangi
    modelin daha güvenilir olduğunu gösterir."""
    hours = pd.DatetimeIndex(test_index).hour
    fig, ax = plt.subplots(figsize=(13, 5))
    for name, p in preds.items():
        mae = pd.Series(np.abs(p - y_true)).groupby(hours).mean()
        ax.plot(mae.index, mae.values, color=_color(name), linewidth=2, marker="o",
                markersize=5, label=name)
    ax.set_xticks(range(0, 24))
    ax.legend(frameon=False, ncol=len(preds), loc="lower left", bbox_to_anchor=(0, 1.0))
    _style(ax, None, "Saat", "MAE (soket)")
    fig.suptitle(title, fontsize=14, fontweight="bold", x=0.01, ha="left")
    _save(fig, save_path)


def plot_regime_mae(regimes: pd.DataFrame, save_path: str, title: str):
    """Rejim başına MAE (seed ortalaması): değişmeyen vs değişen adımlar."""
    names = list(regimes.index)
    share = regimes.attrs.get("changed_share", np.nan)
    subtitles = [f"Doluluğun değişmediği adımlar (%{(1 - share) * 100:.0f})",
                 f"Doluluğun değiştiği adımlar (%{share * 100:.0f})"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, sub in zip(axes, REGIMES[:2], subtitles):
        vals = regimes[col].values
        ax.bar(names, vals, width=0.6, color=[_color(n) for n in names],
               edgecolor="white", linewidth=2)
        best = int(np.argmin(vals))
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=ACTUAL_COLOR,
                    fontweight="bold" if i == best else "normal")
        _style(ax, sub + "  (düşük = iyi)", None, "MAE (soket)")
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle(title, fontsize=14, fontweight="bold", x=0.01, ha="left")
    _save(fig, save_path)


def plot_xgb_importance(importance: pd.Series, save_path: str, title: str, top=20):
    """Sol: en önemli top-N özellik (gain); sağ: LSTM kaynaklı vs tabular özellik
    toplam gain payı — XGBoost'un LSTM'i ne kadar kullandığını gösterir."""
    is_lstm = lambda n: n.startswith("lstm_")      # lstm_pred (OOF) veya eski lstm_emb_*
    is_emb = importance.index.map(is_lstm).to_numpy(bool)
    top_imp = importance.head(top)[::-1]
    emb_color, tab_color = MODEL_COLORS["LSTM_XGBoost"], MODEL_COLORS["LSTM_baseline"]
    lstm_label = "LSTM embedding" if importance.index.str.startswith("lstm_emb_").any() \
        else "LSTM tahmini (OOF)"

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [2.2, 1]})
    axes[0].barh(top_imp.index, top_imp.values, height=0.7, edgecolor="white", linewidth=2,
                 color=[emb_color if is_lstm(n) else tab_color for n in top_imp.index])
    # gain birkaç özellikte yoğunlaşıyorsa (≥2 mertebe) kuyruk ancak log ölçekte görünür
    log = top_imp.max() / max(top_imp.min(), 1e-12) >= 100
    if log:
        axes[0].set_xscale("log")
    _style(axes[0], f"En önemli {top} özellik (toplam gain{', log ölçek' if log else ''})",
           "Toplam gain")
    axes[0].tick_params(axis="y", labelsize=8)
    axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=emb_color),
                            plt.Rectangle((0, 0), 1, 1, color=tab_color)],
                   labels=[lstm_label, "Tabular özellik"], frameon=False, loc="lower right")

    total = importance.sum()
    shares = [importance[is_emb].sum() / total * 100, importance[~is_emb].sum() / total * 100]
    axes[1].bar([lstm_label.replace(" ", "\n", 1), "Tabular\nözellik"], shares, width=0.55,
                color=[emb_color, tab_color], edgecolor="white", linewidth=2)
    for i, v in enumerate(shares):
        axes[1].text(i, v, f"%{v:.1f}", ha="center", va="bottom", fontsize=11, color=ACTUAL_COLOR)
    axes[1].set_ylim(0, 105)
    _style(axes[1], "Toplam gain payı", None, "%")
    fig.suptitle(title, fontsize=14, fontweight="bold", x=0.01, ha="left")
    _save(fig, save_path)


def plot_efficiency(summary: pd.DataFrame, save_path: str, title: str):
    """Eğitim süresi vs MAE (sol-alt köşe = hızlı ve doğru)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    xs = summary[("train_sec", "mean")].to_numpy(float)
    ys = summary[("MAE", "mean")].to_numpy(float)
    # En yakın komşu (eksen aralığına normalize) üstteyse etiket alta yazılır → çakışma önlenir
    nx = (xs - xs.min()) / (np.ptp(xs) or 1)
    ny = (ys - ys.min()) / (np.ptp(ys) or 1)
    for i, name in enumerate(summary.index):
        s = summary.loc[name]
        x, y = xs[i], ys[i]
        ax.errorbar(x, y, xerr=s[("train_sec", "std")], yerr=s[("MAE", "std")],
                    fmt="o", markersize=10, color=_color(name), ecolor=_color(name),
                    elinewidth=1, capsize=3, markeredgecolor="white", markeredgewidth=2)
        dist = np.hypot(nx - nx[i], ny - ny[i])
        dist[i] = np.inf
        below = ny[dist.argmin()] > ny[i]
        params = int(s[("params", "mean")])
        ax.annotate(f"{name}\n{params:,} param", (x, y), textcoords="offset points",
                    xytext=(10, -8 if below else 6), va="top" if below else "bottom",
                    fontsize=9, color=ACTUAL_COLOR)
    ax.margins(y=0.12)
    _style(ax, None, "Eğitim süresi (s, seed ortalaması)", "Test MAE (soket)")
    fig.suptitle(title, fontsize=14, fontweight="bold", x=0.01, ha="left")
    _save(fig, save_path)
