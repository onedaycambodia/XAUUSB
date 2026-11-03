"""ការកំណត់ទាំងអស់ (អាចកែតាម Environment Variables)។"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

TIMEFRAMES = ["M5", "M15", "M30", "H1", "H4", "D1"]
TF_SECONDS = {"M5": 300, "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400, "D1": 86400}

DEFAULT_WEIGHTS = {
    "trend_price": 20,   # តម្លៃលើ/ក្រោម EMA20 និង EMA50
    "ema_cross": 15,     # EMA20 ធៀប EMA50
    "rsi": 15,           # RSI ធៀប 50
    "macd": 15,          # MACD ធៀប Signal
    "adx": 15,           # កម្លាំងនិន្នាការ ADX + ទិសដៅ DI
    "price_action": 20,  # Price action បញ្ជាក់ម៉ូមង់ទឹម
}
DEFAULT_MTF_WEIGHTS = {"M5": 1, "M15": 1, "M30": 2, "H1": 3, "H4": 4, "D1": 5}
DEFAULT_TTL = {"M5": 60, "M15": 120, "M30": 180, "H1": 300, "H4": 600, "D1": 900}


def _env(name, default, cast=float):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError):
        return default


def _env_bool(name, default):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_json(name, default):
    raw = os.getenv(name)
    if not raw:
        return dict(default)
    try:
        val = json.loads(raw)
        if isinstance(val, dict):
            merged = dict(default)
            merged.update({k: float(v) for k, v in val.items() if k in default})
            return merged
    except (ValueError, TypeError):
        pass
    return dict(default)


def normalise_weights(weights: dict) -> dict:
    """ធ្វើឲ្យផលបូកទម្ងន់ = 100 ដើម្បីពិន្ទុនៅចន្លោះ 0-100។"""
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {k: max(0.0, float(v)) * 100.0 / total for k, v in weights.items()}


@dataclass
class Settings:
    # ---- ទិន្នន័យទីផ្សារ ----
    twelvedata_key: str = ""
    twelvedata_symbol: str = "XAU/USD"
    yahoo_symbol: str = "XAUUSD=X"      # Spot (មិនមែន GC=F Futures)
    http_timeout: float = 10.0
    http_retries: int = 2
    max_stale_sec: int = 6 * 3600
    fail_backoff_sec: int = 30
    candles_wanted: int = 600
    min_candles: int = 100               # តិចជាងនេះ => NEUTRAL
    stale_extra_sec: int = 900
    cache_ttl: dict = field(default_factory=lambda: dict(DEFAULT_TTL))
    news_ttl: int = 600
    # ---- Signal ----
    primary_tf: str = "H1"
    weights: dict = field(default_factory=lambda: normalise_weights(DEFAULT_WEIGHTS))
    buy_threshold: float = 70.0
    min_conditions: int = 4
    adx_min: float = 20.0
    adx_strong: float = 25.0
    pa_body_frac: float = 0.30
    pa_lookback: int = 3
    mtf_weights: dict = field(default_factory=lambda: dict(DEFAULT_MTF_WEIGHTS))
    mtf_agree_ratio: float = 0.60
    require_mtf_agreement: bool = True
    # ---- Entry / SL / TP ----
    sl_atr_mult: float = 1.5
    rr_list: tuple = (1.0, 2.0, 3.0)
    drift_atr: float = 0.5
    # ---- Support / Resistance ----
    swing_window: int = 5
    sr_lookback: int = 300
    sr_cluster_atr: float = 0.5
    sr_min_touches: int = 2
    breakout_confirm_bars: int = 2
    breakout_buffer_atr: float = 0.25
    # ---- Volatility ----
    vol_high: float = 1.3
    vol_low: float = 0.7
    # ---- Backtest ----
    bt_spread: float = 0.30          # USD ក្នុង 1 oz
    bt_slippage: float = 0.10        # USD ក្នុង 1 oz ក្នុងមួយចំហៀង
    bt_commission_per_lot: float = 7.0   # USD ក្នុង 1 lot (round turn)
    bt_lot: float = 0.01
    bt_contract: float = 100.0       # oz ក្នុង 1 lot
    bt_rr: float = 2.0
    bt_max_hold: int = 48
    bt_train_ratio: float = 0.70
    bt_balance: float = 1000.0
    bt_ttl: int = 1800

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        s.twelvedata_key = os.getenv("TWELVEDATA_API_KEY", "").strip()
        s.twelvedata_symbol = os.getenv("TWELVEDATA_SYMBOL", s.twelvedata_symbol)
        s.yahoo_symbol = os.getenv("GOLD_SYMBOL", s.yahoo_symbol)
        s.http_timeout = _env("HTTP_TIMEOUT", s.http_timeout)
        s.http_retries = _env("HTTP_RETRIES", s.http_retries, int)
        s.primary_tf = os.getenv("PRIMARY_TIMEFRAME", s.primary_tf).upper()
        if s.primary_tf not in TIMEFRAMES:
            s.primary_tf = "H1"
        s.weights = normalise_weights(_env_json("SCORE_WEIGHTS", DEFAULT_WEIGHTS))
        s.mtf_weights = _env_json("MTF_WEIGHTS", DEFAULT_MTF_WEIGHTS)
        s.buy_threshold = _env("SIGNAL_THRESHOLD", s.buy_threshold)
        s.min_conditions = _env("MIN_CONDITIONS", s.min_conditions, int)
        s.adx_min = _env("ADX_MIN", s.adx_min)
        s.adx_strong = _env("ADX_STRONG", s.adx_strong)
        s.require_mtf_agreement = _env_bool("REQUIRE_MTF_AGREEMENT", s.require_mtf_agreement)
        s.sl_atr_mult = _env("SL_ATR_MULT", s.sl_atr_mult)
        s.bt_spread = _env("BT_SPREAD", s.bt_spread)
        s.bt_slippage = _env("BT_SLIPPAGE", s.bt_slippage)
        s.bt_commission_per_lot = _env("BT_COMMISSION_PER_LOT", s.bt_commission_per_lot)
        s.bt_lot = _env("BT_LOT", s.bt_lot)
        s.bt_rr = _env("BT_RR", s.bt_rr)
        s.bt_max_hold = _env("BT_MAX_HOLD", s.bt_max_hold, int)
        s.bt_train_ratio = min(0.9, max(0.5, _env("BT_TRAIN_RATIO", s.bt_train_ratio)))
        s.bt_balance = _env("BT_BALANCE", s.bt_balance)
        return s


def public_config(s: Settings) -> dict:
    """ការកំណត់ដែលអាចបង្ហាញលើ Frontend (គ្មាន API key)។"""
    return {
        "primary_tf": s.primary_tf,
        "weights": {k: round(v, 2) for k, v in s.weights.items()},
        "signal_threshold": s.buy_threshold,
        "min_conditions": s.min_conditions,
        "adx_min": s.adx_min,
        "adx_strong": s.adx_strong,
        "sl_atr_mult": s.sl_atr_mult,
        "rr_list": list(s.rr_list),
        "mtf_weights": s.mtf_weights,
        "require_mtf_agreement": s.require_mtf_agreement,
        "backtest": {
            "spread": s.bt_spread, "slippage": s.bt_slippage,
            "commission_per_lot": s.bt_commission_per_lot, "lot": s.bt_lot,
            "rr": s.bt_rr, "max_hold": s.bt_max_hold, "train_ratio": s.bt_train_ratio,
        },
        "data_source": "twelvedata" if s.twelvedata_key else "yahoo",
    }
