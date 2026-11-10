"""
src/pricing.py
---------------
Talep (doluluk) tahminine dayalı dinamik fiyatlandırma ve talep esnekliği simülasyonu.
"""

import numpy as np


def congestion_price(predicted_occupancy: np.ndarray,
                      capacity: float,
                      base_price: np.ndarray,
                      target_utilization: float = 0.6,
                      alpha: float = 0.8,
                      price_band: float = 0.5) -> np.ndarray:
    """Tahmini doluluk oranına göre dinamik tarife hesaplar."""
    utilization = np.clip(predicted_occupancy / max(capacity, 1e-6), 0, 1.5)
    adjustment = 1 + alpha * (utilization - target_utilization)

    dynamic_price = base_price * adjustment
    lower_bound = base_price * (1 - price_band)
    upper_bound = base_price * (1 + price_band)
    return np.clip(dynamic_price, lower_bound, upper_bound)


def simulate_demand_response(volume: np.ndarray,
                              old_price: np.ndarray,
                              new_price: np.ndarray,
                              elasticity: float = -0.3) -> np.ndarray:
    """Doğrusal fiyat-talep esnekliği ile yeni fiyat altındaki şarj hacmini hesaplar."""
    price_change_pct = (new_price - old_price) / np.where(old_price == 0, 1e-6, old_price)
    volume_adjusted = volume * (1 + elasticity * price_change_pct)
    return np.clip(volume_adjusted, 0, None)


def evaluate_pricing_policy(predicted_occupancy: np.ndarray,
                             actual_volume: np.ndarray,
                             base_price: np.ndarray,
                             capacity: float,
                             **congestion_kwargs) -> dict:
    """Statik ToU (baseline) ile dinamik fiyatlandırmayı gelir ve tepe yük
    açısından karşılaştırır."""
    dynamic_price = congestion_price(predicted_occupancy, capacity, base_price,
                                      **congestion_kwargs)
    adjusted_volume = simulate_demand_response(actual_volume, base_price, dynamic_price)

    static_revenue = float(np.sum(base_price * actual_volume))
    dynamic_revenue = float(np.sum(dynamic_price * adjusted_volume))

    static_peak = float(np.max(actual_volume))
    dynamic_peak = float(np.max(adjusted_volume))

    return {
        "static_revenue": static_revenue,
        "dynamic_revenue": dynamic_revenue,
        "revenue_change_%": (dynamic_revenue - static_revenue) / static_revenue * 100
        if static_revenue > 0 else float("nan"),
        "static_peak_volume": static_peak,
        "dynamic_peak_volume": dynamic_peak,
        "peak_reduction_%": (static_peak - dynamic_peak) / static_peak * 100
        if static_peak > 0 else float("nan"),
    }
