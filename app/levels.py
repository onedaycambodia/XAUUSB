"""Support / Resistance: swing high-low, pivot points, ចង្កោម (cluster) និងការបញ្ជាក់ Breakout។
មិនបង្កើតកម្រិតតម្លៃដោយការស្មានទេ ‒ កម្រិតទាំងអស់មកពីទិន្នន័យ candle ពិត។"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings
from .util import iso, num


def swing_points(df: pd.DataFrame, k: int):
    """Swing ដែលបានបញ្ជាក់ (ត្រូវការ k candle ទាំងសងខាង => គ្មាន look-ahead)។"""
    w = 2 * k + 1
    hi_max = df["high"].rolling(w, center=True, min_periods=w).max()
    lo_min = df["low"].rolling(w, center=True, min_periods=w).min()
    sh = df["high"][(df["high"] == hi_max)]
    sl = df["low"][(df["low"] == lo_min)]
    return sh, sl


def cluster_levels(points: list, tol: float) -> list:
    """points = [(price, time, kind)] → ចង្កោមកម្រិតជិតគ្នា។"""
    if not points:
        return []
    pts = sorted(points, key=lambda x: x[0])
    groups, cur = [], [pts[0]]
    for p in pts[1:]:
        if p[0] - cur[-1][0] <= tol:
            cur.append(p)
        else:
            groups.append(cur)
            cur = [p]
    groups.append(cur)
    out = []
    for g in groups:
        out.append({"price": float(np.mean([x[0] for x in g])), "touches": len(g),
                    "last_touch": max(x[1] for x in g),
                    "highs": sum(1 for x in g if x[2] == "H"), "lows": sum(1 for x in g if x[2] == "L")})
    return out


def pivots_from_daily(daily: pd.DataFrame | None):
    """Classic pivot ពី candle D1 ដែលបិទរួចចុងក្រោយ។"""
    if daily is None or len(daily) == 0:
        return None
    r = daily.iloc[-1]
    h, l, c = float(r["high"]), float(r["low"]), float(r["close"])
    p = (h + l + c) / 3
    return {"P": p, "R1": 2 * p - l, "S1": 2 * p - h, "R2": p + (h - l), "S2": p - (h - l),
            "basis_time": daily.index[-1]}


def _breakouts(closes: np.ndarray, levels: list, atr: float, cfg: Settings):
    n = cfg.breakout_confirm_bars
    buf = cfg.breakout_buffer_atr * atr
    res = {"breakout": None, "breakdown": None}
    if len(closes) < n + 8:
        return res
    for lv in sorted(levels, key=lambda x: -x["touches"]):
        L = lv["price"]
        for s in range(0, 6):
            end = len(closes) - s
            seg, prior = closes[end - n:end], closes[end - n - 1]
            if res["breakout"] is None and (seg > L + buf).all() and prior <= L + buf \
                    and closes[-1] > L:
                res["breakout"] = {"level": num(L), "bars_ago": s, "confirm_bars": n}
            if res["breakdown"] is None and (seg < L - buf).all() and prior >= L - buf \
                    and closes[-1] < L:
                res["breakdown"] = {"level": num(L), "bars_ago": s, "confirm_bars": n}
    return res


def analyse(ind: pd.DataFrame, price: float, daily: pd.DataFrame | None, cfg: Settings) -> dict:
    df = ind.tail(cfg.sr_lookback)
    atr_v = float(ind["atr"].dropna().iloc[-1]) if ind["atr"].notna().any() else float("nan")
    out = {"price": num(price), "swing_levels": [], "pivots": None, "support": None,
           "resistance": None, "breakout": None, "breakdown": None, "notes": []}
    if len(df) < 2 * cfg.swing_window + 5 or np.isnan(atr_v):
        out["notes"].append("ទិន្នន័យមិនគ្រប់គ្រាន់សម្រាប់គណនា Support/Resistance")
        return out
    sh, sl = swing_points(df, cfg.swing_window)
    pts = [(float(v), t, "H") for t, v in sh.items()] + [(float(v), t, "L") for t, v in sl.items()]
    levels = cluster_levels(pts, cfg.sr_cluster_atr * atr_v)
    confirmed = [x for x in levels if x["touches"] >= cfg.sr_min_touches]
    piv = pivots_from_daily(daily)

    cands = [{"price": x["price"], "source": "swing", "touches": x["touches"]} for x in confirmed]
    if piv:
        for name in ("R2", "R1", "P", "S1", "S2"):
            cands.append({"price": piv[name], "source": f"pivot {name}", "touches": None})
    below = [x for x in cands if x["price"] < price]
    above = [x for x in cands if x["price"] > price]

    def pack(x):
        if x is None:
            return None
        d = abs(price - x["price"])
        return {"price": num(x["price"]), "source": x["source"], "touches": x["touches"],
                "distance": num(d), "distance_pct": num(d / price * 100, 3),
                "distance_atr": num(d / atr_v, 2)}
    out["support"] = pack(max(below, key=lambda x: x["price"]) if below else None)
    out["resistance"] = pack(min(above, key=lambda x: x["price"]) if above else None)
    out["swing_levels"] = [
        {"price": num(x["price"]), "touches": x["touches"],
         "kind": "resistance" if x["price"] > price else "support",
         "distance": num(abs(price - x["price"]))}
        for x in sorted(confirmed, key=lambda x: -x["price"])]
    if piv:
        out["pivots"] = {k: num(v) for k, v in piv.items() if k != "basis_time"}
        out["pivots"]["basis_time"] = iso(piv["basis_time"])
    else:
        out["notes"].append("មិនមានទិន្នន័យ D1 សម្រាប់គណនា Pivot")
    if not confirmed:
        out["notes"].append("មិនមានកម្រិត swing ដែលមានការប៉ះច្រើនដង (ទិន្នន័យតិច ឬ ទីផ្សារមិនទាន់បង្កើតរចនាសម្ព័ន្ធ)")
    bo = _breakouts(df["close"].to_numpy(), confirmed, atr_v, cfg)
    out["breakout"], out["breakdown"] = bo["breakout"], bo["breakdown"]
    return out
