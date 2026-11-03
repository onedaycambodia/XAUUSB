"""Entry / Stop Loss / Take Profit តាមរូបមន្តកំណត់ទុក (ATR-based)។
ជាកម្រិតយោងសម្រាប់វិភាគ មិនមែនការណែនាំ trade ទេ។"""
from __future__ import annotations

import math

from .config import Settings
from .util import num


def build_setup(signal: str, entry: float, atr: float, current: float | None,
                levels: dict | None, cfg: Settings) -> dict | None:
    if signal not in ("BUY", "SELL") or not atr or math.isnan(atr) or atr <= 0 or not entry:
        return None
    d = 1 if signal == "BUY" else -1
    risk = cfg.sl_atr_mult * atr
    sl = entry - d * risk
    targets = [{"rr": rr, "price": num(entry + d * risk * rr), "reward": num(risk * rr)}
               for rr in cfg.rr_list]
    cur = current if current is not None else entry
    status, status_label, invalid = "active", "ដំណើរការ", []
    if (d == 1 and cur <= sl) or (d == -1 and cur >= sl):
        status, status_label = "invalidated", "អស់សុពលភាព: តម្លៃប៉ះ Stop Loss"
        invalid.append("តម្លៃបច្ចុប្បន្នបានប៉ះ/ឆ្លងកាត់ Stop Loss")
    elif (d == 1 and cur >= targets[-1]["price"]) or (d == -1 and cur <= targets[-1]["price"]):
        status, status_label = "target_hit", "បានដល់ TP ចុងក្រោយ"
    elif abs(cur - entry) > cfg.drift_atr * atr:
        status, status_label = "drifted", "តម្លៃឆ្ងាយពីចំណុចចូលលើសកំណត់"
        invalid.append(f"តម្លៃបច្ចុប្បន្នឆ្ងាយពី Entry លើស {cfg.drift_atr:g}×ATR")
    notes = []
    if levels:
        opp = levels.get("resistance") if d == 1 else levels.get("support")
        if opp and opp.get("price") is not None:
            for t in targets:
                if (d == 1 and opp["price"] < t["price"]) or (d == -1 and opp["price"] > t["price"]):
                    notes.append(f"ប្រុងប្រយ័ត្ន: {'Resistance' if d == 1 else 'Support'} "
                                 f"{opp['price']} នៅមុន TP 1:{t['rr']:g}")
                    break
    return {
        "side": signal, "entry": num(entry), "stop_loss": num(sl), "risk": num(risk),
        "atr": num(atr), "sl_atr_mult": cfg.sl_atr_mult, "targets": targets,
        "rr_text": " / ".join(f"1:{t['rr']:g}" for t in targets),
        "status": status, "status_label": status_label,
        "valid": status == "active", "invalidation": invalid,
        "invalidation_rules": [
            "Signal ប្តូរទៅទិសផ្ទុយ ឬ NEUTRAL នៅ candle បិទបន្ទាប់",
            "តម្លៃប៉ះ Stop Loss",
            f"តម្លៃឆ្ងាយពី Entry លើស {cfg.drift_atr:g}×ATR",
        ],
        "notes": notes,
    }
