"""Backtesting engine ‒ គ្មាន look-ahead bias។

- Signal គណនានៅពេល candle i បិទ (indicator causal)។ Entry = តម្លៃ open នៃ candle i+1។
- SL/TP គណនាពី ATR ត្រឹម candle i (ព័ត៌មានដែលស្គាល់ពេលនោះ)។
- ប្រសិន SL និង TP ប៉ះក្នុង candle តែមួយ => សន្មត់ថា SL មុន (អភិរក្ស)។
- ថ្លៃ៖ spread + slippage (ម្ខាងៗ) + commission ក្នុងមួយ lot។
- បែងចែកពេល Train/Test តាមលំដាប់ពេល។ មិនមាន parameter optimisation => មិន overfit ដោយស្វ័យប្រវត្តិ។
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings
from .util import iso, num


def simulate(df: pd.DataFrame, signal, atr, cfg: Settings, rr: float | None = None) -> list:
    rr = rr or cfg.bt_rr
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    sig = np.asarray(signal)
    atr = np.asarray(atr, dtype=float)
    idx = df.index
    n = len(df)
    half = cfg.bt_spread / 2.0 + cfg.bt_slippage
    qty = cfg.bt_lot * cfg.bt_contract
    comm = cfg.bt_commission_per_lot * cfg.bt_lot
    trades, i = [], 0
    while i < n - 1:
        s = sig[i]
        if s not in ("BUY", "SELL") or np.isnan(atr[i]) or atr[i] <= 0:
            i += 1
            continue
        d = 1 if s == "BUY" else -1
        entry = o[i + 1] + d * half
        risk = cfg.sl_atr_mult * atr[i]
        sl, tp = entry - d * risk, entry + d * risk * rr
        j_end = min(n - 1, i + cfg.bt_max_hold)
        exit_raw, reason, j = None, None, None
        for j in range(i + 1, j_end + 1):
            hit_sl = l[j] <= sl if d == 1 else h[j] >= sl
            hit_tp = h[j] >= tp if d == 1 else l[j] <= tp
            if hit_sl:
                gap = (o[j] < sl) if d == 1 else (o[j] > sl)
                exit_raw, reason = (o[j] if gap else sl), "SL"
                break
            if hit_tp:
                exit_raw, reason = tp, "TP"
                break
        if exit_raw is None:
            j = j_end
            exit_raw = c[j]
            reason = "END" if (j == n - 1 and i + cfg.bt_max_hold > n - 1) else "TIME"
        exit_fill = exit_raw - d * half
        pnl = (exit_fill - entry) * d * qty - comm
        trades.append({"entry_idx": i + 1, "entry_time": idx[i + 1], "exit_time": idx[j],
                       "side": s, "entry": entry, "exit": exit_fill, "sl": sl, "tp": tp,
                       "pnl": pnl, "reason": reason, "bars": j - i})
        i = j
    return trades


def metrics(trades: list, balance: float) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": None, "profit_factor": None,
                "net_profit": 0.0, "gross_profit": 0.0, "gross_loss": 0.0,
                "max_drawdown": 0.0, "max_drawdown_pct": 0.0, "avg_win": None,
                "avg_loss": None, "expectancy": None}
    pnl = np.array([t["pnl"] for t in trades])
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gp, gl = float(wins.sum()), float(-losses.sum())
    eq = balance + np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[balance], eq]))[1:]
    dd = peak - eq
    k = int(dd.argmax())
    return {
        "trades": n, "wins": int(len(wins)), "losses": int(len(losses)),
        "win_rate": num(len(wins) / n * 100, 1),
        "profit_factor": num(gp / gl, 2) if gl > 0 else None,
        "net_profit": num(pnl.sum()), "gross_profit": num(gp), "gross_loss": num(gl),
        "max_drawdown": num(dd.max()), "max_drawdown_pct": num(dd.max() / peak[k] * 100, 2),
        "avg_win": num(wins.mean()) if len(wins) else None,
        "avg_loss": num(losses.mean()) if len(losses) else None,
        "expectancy": num(pnl.mean()),
    }


def run(ind: pd.DataFrame, frame: pd.DataFrame, cfg: Settings, rr: float | None = None,
        tf: str = "") -> dict:
    rr = rr or cfg.bt_rr
    valid_idx = np.flatnonzero(frame["valid"].to_numpy())
    if len(ind) < cfg.min_candles + 50 or len(valid_idx) < 50:
        return {"ok": False, "error": "ទិន្នន័យប្រវត្តិមិនគ្រប់គ្រាន់សម្រាប់ Backtest",
                "candles": int(len(ind)), "timeframe": tf}
    start = int(valid_idx[0])
    d = ind.iloc[start:]
    f = frame.iloc[start:]
    n = len(d)
    split = int(n * cfg.bt_train_ratio)
    trades = simulate(d, f["signal"].to_numpy(), d["atr"].to_numpy(), cfg, rr)
    train = [t for t in trades if t["entry_idx"] < split]
    test = [t for t in trades if t["entry_idx"] >= split]

    def block(tr, a, b):
        return {"from": iso(d.index[a]), "to": iso(d.index[b - 1]), "candles": b - a,
                **metrics(tr, cfg.bt_balance)}
    notes = ["លទ្ធផលអតីតកាលមិនធានាអនាគតទេ។",
             "Train/Test បែងចែកតាមពេល: ប្រព័ន្ធមិនបាន optimise parameter លើ Train ទេ "
             "(កំណត់ដោយ rule ថេរ) ដូច្នេះ Test គឺជាការពិនិត្យ out-of-sample ត្រង់ៗ។",
             "Candle ដែល SL និង TP ប៉ះព្រមគ្នា ត្រូវរាប់ជា SL (អភិរក្ស)។"]
    for name, blk in (("Train", train), ("Test", test)):
        if len(blk) < 30:
            notes.append(f"{name}: មាន {len(blk)} trade (<30) ‒ ស្ថិតិមិនទាន់គ្រប់គ្រាន់សម្រាប់សន្និដ្ឋាន")
    return {
        "ok": True, "timeframe": tf, "rr": rr,
        "params": {"spread": cfg.bt_spread, "slippage_per_side": cfg.bt_slippage,
                   "commission_per_lot": cfg.bt_commission_per_lot, "lot": cfg.bt_lot,
                   "contract_oz": cfg.bt_contract, "sl_atr_mult": cfg.sl_atr_mult,
                   "max_hold_bars": cfg.bt_max_hold, "initial_balance": cfg.bt_balance,
                   "threshold": cfg.buy_threshold},
        "period": {"from": iso(d.index[0]), "to": iso(d.index[-1]), "candles": n,
                   "split_time": iso(d.index[split])},
        "train": block(train, 0, split), "test": block(test, split, n),
        "all": metrics(trades, cfg.bt_balance),
        "recent_trades": [{"entry_time": iso(t["entry_time"]), "exit_time": iso(t["exit_time"]),
                           "side": t["side"], "entry": num(t["entry"]), "exit": num(t["exit"]),
                           "pnl": num(t["pnl"]), "reason": t["reason"], "bars": t["bars"]}
                          for t in trades[-25:][::-1]],
        "notes": notes,
    }
