"""ការរកឃើញស្ថានភាពទីផ្សារដោយស្វ័យប្រវត្តិ។"""
from __future__ import annotations

import pandas as pd

from . import i18n
from .config import Settings
from .util import num


def detect(ind: pd.DataFrame, lv: dict | None, cfg: Settings) -> dict:
    r = ind.iloc[-1]
    need = ["close", "ema20", "ema50", "adx", "plus_di", "minus_di", "atr"]
    if any(pd.isna(r.get(k)) for k in need):
        return {"trend": "sideways", "trend_label": "មិនអាចកំណត់ (ទិន្នន័យមិនគ្រប់)",
                "volatility": "normal", "volatility_label": i18n.VOL_LABEL["normal"],
                "structure": None, "structure_label": None, "atr_ratio": None,
                "tags": [], "available": False}
    bull_al = r.close > r.ema20 > r.ema50
    bear_al = r.close < r.ema20 < r.ema50
    if r.adx < cfg.adx_min:
        trend = "sideways"             # ADX ទាប = គ្មាននិន្នាការច្បាស់ ទោះ EMA តម្រៀបតាមឱកាស
    elif bull_al:
        trend = "strong_bull" if (r.adx >= cfg.adx_strong and r.plus_di > r.minus_di) else "weak_bull"
    elif bear_al:
        trend = "strong_bear" if (r.adx >= cfg.adx_strong and r.minus_di > r.plus_di) else "weak_bear"
    else:
        trend = "sideways"

    atr_hist = ind["atr"].dropna().tail(51)
    ratio = None
    vol = "normal"
    if len(atr_hist) >= 20:
        base = atr_hist.iloc[:-1].mean()
        if base > 0:
            ratio = float(atr_hist.iloc[-1] / base)
            vol = "high" if ratio >= cfg.vol_high else ("low" if ratio <= cfg.vol_low else "normal")

    struct = None
    if lv:
        if lv.get("breakout"):
            struct = "breakout"
        elif lv.get("breakdown"):
            struct = "breakdown"
    tags = [i18n.TREND_LABEL[trend]]
    if vol != "normal":
        tags.append(i18n.VOL_LABEL[vol])
    if struct:
        tags.append(i18n.STRUCT_LABEL[struct])
    return {"trend": trend, "trend_label": i18n.TREND_LABEL[trend],
            "volatility": vol, "volatility_label": i18n.VOL_LABEL[vol],
            "atr_ratio": num(ratio, 2), "structure": struct,
            "structure_label": i18n.STRUCT_LABEL.get(struct), "tags": tags, "available": True}
