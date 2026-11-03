"""Trading Signal Engine (deterministic, rule-based)។

ពិន្ទុ 0-100 គឺ "ពិន្ទុលក្ខខណ្ឌបច្ចេកទេស" (ផលបូកទម្ងន់នៃលក្ខខណ្ឌដែលពេញ)
មិនមែនប្រូបាប៊ីលីតេឈ្នះទេ។ Signal BUY/SELL ត្រូវការ៖
  ពិន្ទុ ≥ កម្រិត + លក្ខខណ្ឌពេញ ≥ min_conditions + ADX បញ្ជាក់ + Price action បញ្ជាក់
ដូច្នេះមិនអាចកើតពី indicator តែមួយទេ។
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import i18n
from .config import Settings
from .util import num

KEYS = ["trend_price", "ema_cross", "rsi", "macd", "adx", "price_action"]
NEEDED = ["ema20", "ema50", "rsi", "macd", "macd_signal", "adx", "plus_di", "minus_di", "atr"]


def signal_frame(ind: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    """គណនា signal គ្រប់ candle (causal: ជួរ i ប្រើតែទិន្នន័យ ≤ i)។"""
    o, h, l, c = ind["open"], ind["high"], ind["low"], ind["close"]
    rng = (h - l).replace(0, np.nan)
    body = c - o
    frac = body.abs() / rng
    prev = c.shift(cfg.pa_lookback)
    bull = {
        "trend_price": (c > ind["ema20"]) & (c > ind["ema50"]),
        "ema_cross": ind["ema20"] > ind["ema50"],
        "rsi": ind["rsi"] > 50,
        "macd": ind["macd"] > ind["macd_signal"],
        "adx": (ind["adx"] >= cfg.adx_min) & (ind["plus_di"] > ind["minus_di"]),
        "price_action": (body > 0) & (frac >= cfg.pa_body_frac) & (c > prev),
    }
    bear = {
        "trend_price": (c < ind["ema20"]) & (c < ind["ema50"]),
        "ema_cross": ind["ema20"] < ind["ema50"],
        "rsi": ind["rsi"] < 50,
        "macd": ind["macd"] < ind["macd_signal"],
        "adx": (ind["adx"] >= cfg.adx_min) & (ind["minus_di"] > ind["plus_di"]),
        "price_action": (body < 0) & (frac >= cfg.pa_body_frac) & (c < prev),
    }
    out = pd.DataFrame(index=ind.index)
    valid = ind[NEEDED].notna().all(axis=1)
    bs = np.zeros(len(ind))
    es = np.zeros(len(ind))
    nb = np.zeros(len(ind), dtype=int)
    ne = np.zeros(len(ind), dtype=int)
    for k in KEYS:
        b = bull[k].fillna(False).to_numpy() & valid.to_numpy()
        e = bear[k].fillna(False).to_numpy() & valid.to_numpy()
        out[f"bull_{k}"], out[f"bear_{k}"] = b, e
        bs += cfg.weights.get(k, 0) * b
        es += cfg.weights.get(k, 0) * e
        nb += b
        ne += e
    out["valid"] = valid.to_numpy()
    out["bull_score"], out["bear_score"] = np.round(bs, 1), np.round(es, 1)
    out["n_bull"], out["n_bear"] = nb, ne
    buy = (out["valid"] & (bs >= cfg.buy_threshold) & (nb >= cfg.min_conditions)
           & out["bull_adx"] & out["bull_price_action"] & (bs > es))
    sell = (out["valid"] & (es >= cfg.buy_threshold) & (ne >= cfg.min_conditions)
            & out["bear_adx"] & out["bear_price_action"] & (es > bs))
    out["signal"] = np.select([buy.to_numpy(), sell.to_numpy()], ["BUY", "SELL"], "NEUTRAL")
    return out


def trend_dir(row) -> str:
    """ទិសដៅនិន្នាការតាម EMA alignment (ប្រើក្នុង Multi-Timeframe)។"""
    if any(pd.isna(row.get(k)) for k in ("close", "ema20", "ema50")):
        return "mixed"
    if row["close"] > row["ema50"] and row["ema20"] > row["ema50"]:
        return "bull"
    if row["close"] < row["ema50"] and row["ema20"] < row["ema50"]:
        return "bear"
    return "mixed"


def evaluate_latest(ind: pd.DataFrame, frame: pd.DataFrame, cfg: Settings,
                    gate_reasons: list | None = None) -> dict:
    """លទ្ធផលសម្រាប់ candle បិទចុងក្រោយ + ការពន្យល់ជាខ្មែរ។"""
    if len(frame) == 0:
        return _empty(["មិនមានទិន្នន័យ"], cfg)
    fr, ir = frame.iloc[-1], ind.iloc[-1]
    sig = fr["signal"]
    reasons = list(gate_reasons or [])
    if gate_reasons:
        sig = "NEUTRAL"
    bs, es = float(fr["bull_score"]), float(fr["bear_score"])
    side = "bull" if bs >= es else "bear"
    if sig == "BUY":
        side = "bull"
    elif sig == "SELL":
        side = "bear"
    score = bs if side == "bull" else es

    conds = []
    for k in KEYS:
        conds.append({
            "key": k, "name": i18n.COND_NAME[k], "weight": round(cfg.weights.get(k, 0), 1),
            "bull_ok": bool(fr[f"bull_{k}"]), "bear_ok": bool(fr[f"bear_{k}"]),
            "bull_text": i18n.COND_BULL[k].format(adx_min=cfg.adx_min),
            "bear_text": i18n.COND_BEAR[k].format(adx_min=cfg.adx_min),
        })

    if not fr["valid"] and not gate_reasons:
        reasons.append("Indicator មិនទាន់គណនាគ្រប់ (candle មិនគ្រប់គ្រាន់ ឬមាន NaN)")
    elif sig == "NEUTRAL" and not gate_reasons:
        adx_v = ir.get("adx")
        if pd.notna(adx_v) and adx_v < cfg.adx_min:
            reasons.append(f"និន្នាការខ្សោយ: ADX {adx_v:.1f} ទាបជាង {cfg.adx_min:g}")
        if fr["n_bull"] >= 2 and fr["n_bear"] >= 2:
            reasons.append(f"Indicator ប្រឆាំងគ្នា (ឡើង {int(fr['n_bull'])} / ចុះ {int(fr['n_bear'])})")
        if not (fr["bull_price_action"] or fr["bear_price_action"]):
            reasons.append("Price action មិនទាន់បញ្ជាក់ទិសដៅ")
        if score < cfg.buy_threshold:
            reasons.append(f"ពិន្ទុ {score:.0f} ទាបជាងកម្រិតសញ្ញា {cfg.buy_threshold:g}")
        if not reasons:
            reasons.append("លក្ខខណ្ឌចាំបាច់ (ADX + Price action) មិនគ្រប់")
    elif sig in ("BUY", "SELL"):
        ok_txt = [c["bull_text"] if sig == "BUY" else c["bear_text"]
                  for c in conds if (c["bull_ok"] if sig == "BUY" else c["bear_ok"])]
        n = int(fr["n_bull"] if sig == "BUY" else fr["n_bear"])
        reasons.append(f"លក្ខខណ្ឌពេញ {n}/6 ពិន្ទុ {score:.0f} ≥ {cfg.buy_threshold:g}")
        reasons.extend(ok_txt)

    return {
        "signal": sig, "signal_label": i18n.SIGNAL_LABEL[sig],
        "raw_signal": fr["signal"], "score": round(score, 1), "side": side,
        "bull_score": round(bs, 1), "bear_score": round(es, 1),
        "n_bull": int(fr["n_bull"]), "n_bear": int(fr["n_bear"]),
        "valid": bool(fr["valid"]), "conditions": conds, "reasons": reasons,
        "threshold": cfg.buy_threshold, "min_conditions": cfg.min_conditions,
        "trend_dir": trend_dir(ir),
        "candle_time": ind.index[-1],
    }


def _empty(reasons, cfg):
    return {"signal": "NEUTRAL", "signal_label": i18n.SIGNAL_LABEL["NEUTRAL"],
            "raw_signal": "NEUTRAL", "score": 0.0, "side": "bull", "bull_score": 0.0,
            "bear_score": 0.0, "n_bull": 0, "n_bear": 0, "valid": False, "conditions": [],
            "reasons": reasons, "threshold": cfg.buy_threshold,
            "min_conditions": cfg.min_conditions, "trend_dir": "mixed", "candle_time": None}


def history(ind: pd.DataFrame, frame: pd.DataFrame, limit: int = 15) -> list:
    """ប្រវត្តិការប្តូរ signal (គណនាពីទិន្នន័យពិត មិនបង្កើតឡើងវិញ)។"""
    if len(frame) == 0:
        return []
    s = frame["signal"]
    chg = s.ne(s.shift())
    rows = []
    for ts in frame.index[chg.to_numpy()][-limit:][::-1]:
        sg = s.loc[ts]
        sc = frame.loc[ts, "bull_score"] if sg == "BUY" else (
            frame.loc[ts, "bear_score"] if sg == "SELL" else
            max(frame.loc[ts, "bull_score"], frame.loc[ts, "bear_score"]))
        rows.append({"time": ts, "signal": sg, "label": i18n.SIGNAL_LABEL[sg],
                     "score": num(sc, 1), "price": num(ind.loc[ts, "close"])})
    return rows
