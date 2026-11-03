"""ភ្ជាប់ម៉ូឌុលទាំងអស់: data → indicators → signal → MTF → levels → setup → regime។
គ្មាន HTTP/Framework នៅទីនេះ => សាកល្បងបានដោយឯករាជ្យពី Frontend។"""
from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd

from . import backtest as bt
from . import i18n, indicators, levels, regime, signals, trade
from .config import TF_SECONDS, TIMEFRAMES, Settings
from .provider import Feed, MarketDataError, market_open
from .util import iso, num


class Analyzer:
    def __init__(self, market, cfg: Settings | None = None, clock=time.time):
        self.market, self.cfg, self.clock = market, cfg or Settings(), clock
        self._prep_cache: dict = {}
        self._bt_cache: dict = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------ prepare
    def _prep(self, tf: str):
        feed: Feed = self.market.get(tf)
        key = (feed.fetched_at, feed.stale, len(feed.df))
        with self._lock:
            hit = self._prep_cache.get(tf)
            if hit and hit[0] == key:
                return hit[1]
            ind = indicators.compute_all(feed.df) if len(feed.df) else feed.df
            frame = signals.signal_frame(ind, self.cfg) if len(ind) else pd.DataFrame()
            val = (feed, ind, frame)
            self._prep_cache[tf] = (key, val)
            return val

    def _gate(self, feed: Feed, ind: pd.DataFrame) -> list:
        """លក្ខខណ្ឌទិន្នន័យ: មិនគ្រប់/យឺត => បង្ខំ NEUTRAL។"""
        why = []
        if len(ind) < self.cfg.min_candles:
            why.append(f"ទិន្នន័យមិនគ្រប់: មាន {len(ind)} candle (ត្រូវការ ≥ {self.cfg.min_candles})")
        if len(ind) and market_open(self.clock()):
            age = self.clock() - ind.index[-1].timestamp() - TF_SECONDS[feed.tf]
            if age > TF_SECONDS[feed.tf] * 2 + self.cfg.stale_extra_sec:
                why.append(f"ទិន្នន័យយឺត: candle ចុងក្រោយបិទមុន {int(age // 60)} នាទី")
        return why

    # ------------------------------------------------------------ per TF
    def _tf_row(self, tf: str) -> dict:
        try:
            feed, ind, frame = self._prep(tf)
        except MarketDataError as e:
            return {"tf": tf, "available": False, "error": str(e)}
        gate = self._gate(feed, ind)
        ev = signals.evaluate_latest(ind, frame, self.cfg, gate) if len(ind) else \
            signals._empty(gate or ["មិនមានទិន្នន័យ"], self.cfg)
        r = ind.iloc[-1] if len(ind) else None
        row = {"tf": tf, "available": True, "signal": ev["signal"],
               "signal_label": ev["signal_label"], "score": ev["score"],
               "bull_score": ev["bull_score"], "bear_score": ev["bear_score"],
               "trend": ev["trend_dir"], "trend_label": i18n.DIR_LABEL[ev["trend_dir"]],
               "candles": int(len(ind)), "gated": bool(gate), "gate_reasons": gate,
               "candle_time": iso(ind.index[-1]) if len(ind) else None}
        if r is not None:
            ema_state = "bull" if (r.ema20 > r.ema50) else ("bear" if r.ema20 < r.ema50 else "mixed") \
                if pd.notna(r.ema20) and pd.notna(r.ema50) else "mixed"
            row.update({
                "rsi": num(r.rsi, 1), "macd": num(r.macd, 3), "macd_signal": num(r.macd_signal, 3),
                "macd_state": ("bull" if r.macd > r.macd_signal else "bear")
                if pd.notna(r.macd) and pd.notna(r.macd_signal) else None,
                "ema_alignment": ema_state, "ema_alignment_label": i18n.DIR_LABEL[ema_state],
                "adx": num(r.adx, 1)})
        return row

    def _overall(self, rows: list) -> dict:
        w = self.cfg.mtf_weights
        bull = bear = tot = 0.0
        buys = sells = neut = 0
        for r in rows:
            if not r.get("available") or r.get("gated") or r.get("candles", 0) < self.cfg.min_candles:
                continue
            wt = float(w.get(r["tf"], 1))
            tot += wt
            bull += wt * (r["trend"] == "bull")
            bear += wt * (r["trend"] == "bear")
            buys += r["signal"] == "BUY"
            sells += r["signal"] == "SELL"
            neut += r["signal"] == "NEUTRAL"
        used = buys + sells + neut
        if tot == 0:
            return {"direction": "mixed", "label": "មិនអាចសន្និដ្ឋាន (ទិន្នន័យ timeframe មិនគ្រប់)",
                    "confirmed": False, "bull_pct": None, "bear_pct": None,
                    "buy_count": 0, "sell_count": 0, "neutral_count": 0, "used": 0,
                    "text": "គ្មាន timeframe ដែលមានទិន្នន័យគ្រប់គ្រាន់"}
        bp, ep = bull / tot, bear / tot
        thr = self.cfg.mtf_agree_ratio
        d = "bull" if bp >= thr else ("bear" if ep >= thr else "mixed")
        label = {"bull": "និន្នាការរួមឡើង (បានបញ្ជាក់ដោយ Timeframe)",
                 "bear": "និន្នាការរួមចុះ (បានបញ្ជាក់ដោយ Timeframe)",
                 "mixed": "និន្នាការរួមមិនច្បាស់ (Timeframe ប្រឆាំងគ្នា)"}[d]
        text = (f"ក្នុង {used} timeframe: BUY {buys}, SELL {sells}, NEUTRAL {neut}។ "
                f"ទម្ងន់ឡើង {bp * 100:.0f}% / ចុះ {ep * 100:.0f}% (កម្រិតបញ្ជាក់ {thr * 100:.0f}%)")
        return {"direction": d, "label": label, "confirmed": d != "mixed",
                "bull_pct": num(bp * 100, 0), "bear_pct": num(ep * 100, 0),
                "buy_count": int(buys), "sell_count": int(sells), "neutral_count": int(neut),
                "used": int(used), "text": text}

    # ------------------------------------------------------------ price
    def _price(self, primary: Feed) -> dict:
        """តម្លៃបច្ចុប្បន្ន (M5 បើមាន) + % ប្តូរ 24 ម៉ោងជាក់ស្តែង។"""
        feed = primary
        try:
            feed = self.market.get("M5")
        except MarketDataError:
            pass
        full = feed.full
        if len(full) == 0:
            return {"last": None, "time": None, "change_24h_pct": None, "change_24h_abs": None,
                    "note": "មិនមានទិន្នន័យតម្លៃ"}
        last_t, last = full.index[-1], float(full["close"].iloc[-1])
        target = last_t - pd.Timedelta(hours=24)
        ref = full[full.index <= target]
        if len(ref) == 0:
            return {"last": num(last), "time": iso(last_t), "change_24h_pct": None,
                    "change_24h_abs": None, "note": "ទិន្នន័យមិនវែងគ្រប់ 24 ម៉ោង"}
        ref_t, ref_p = ref.index[-1], float(ref["close"].iloc[-1])
        span = (last_t - ref_t).total_seconds() / 3600
        note = None
        if span > 30:
            note = f"ចន្លោះទិន្នន័យ {span:.0f} ម៉ោង (ទីផ្សារបិទ/ខ្វះ candle) ‒ % ប្តូរគិតពី {iso(ref_t)}"
        return {"last": num(last), "time": iso(last_t),
                "change_24h_pct": num((last / ref_p - 1) * 100, 2),
                "change_24h_abs": num(last - ref_p), "reference_time": iso(ref_t),
                "reference_price": num(ref_p), "note": note, "source_tf": feed.tf}

    # ------------------------------------------------------------ indicators panel
    @staticmethod
    def _panel(ind: pd.DataFrame, feed: Feed) -> list:
        if len(ind) == 0:
            return []
        r = ind.iloc[-1]
        c = r.close

        def st(cond_up, up="bull", dn="bear"):
            return up if cond_up else dn
        p = []
        if pd.notna(r.rsi):
            s = "bear" if r.rsi >= 70 else ("bull" if r.rsi <= 30 else st(r.rsi > 50))
            txt = "Overbought (≥70)" if r.rsi >= 70 else ("Oversold (≤30)" if r.rsi <= 30 else
                  ("ម៉ូមង់ឡើង (>50)" if r.rsi > 50 else "ម៉ូមង់ចុះ (<50)"))
            p.append({"key": "rsi", "name": "RSI (14)", "value": num(r.rsi, 1), "state": s, "text": txt})
        if pd.notna(r.macd):
            p.append({"key": "macd", "name": "MACD (12,26,9)", "value": num(r.macd, 3),
                      "extra": {"Signal": num(r.macd_signal, 3), "Histogram": num(r.macd_hist, 3)},
                      "state": st(r.macd > r.macd_signal),
                      "text": "MACD លើ Signal" if r.macd > r.macd_signal else "MACD ក្រោម Signal"})
        for n in (9, 20, 50, 200):
            v = r.get(f"ema{n}")
            if pd.notna(v):
                p.append({"key": f"ema{n}", "name": f"EMA {n}", "value": num(v),
                          "state": st(c > v), "text": "តម្លៃនៅលើ" if c > v else "តម្លៃនៅក្រោម"})
            else:
                p.append({"key": f"ema{n}", "name": f"EMA {n}", "value": None, "state": None,
                          "text": f"ទិន្នន័យមិនគ្រប់ (ត្រូវការ ≥ {n} candle)"})
        if pd.notna(r.atr):
            p.append({"key": "atr", "name": "ATR (14)", "value": num(r.atr),
                      "extra": {"% នៃតម្លៃ": num(r.atr / c * 100, 3)}, "state": None,
                      "text": "ជួរប្រែប្រួលជាមធ្យមក្នុងមួយ candle"})
        if pd.notna(r.bb_up):
            pb = r.bb_pctb
            p.append({"key": "bb", "name": "Bollinger (20,2)", "value": num(r.bb_mid),
                      "extra": {"ខាងលើ": num(r.bb_up), "ខាងក្រោម": num(r.bb_low),
                                "%B": num(pb, 2), "ទទឹង": num(r.bb_width * 100, 2)},
                      "state": None if pd.isna(pb) else ("bear" if pb > 1 else "bull" if pb < 0 else None),
                      "text": "តម្លៃលើ band ខាងលើ" if pb > 1 else ("តម្លៃក្រោម band ខាងក្រោម" if pb < 0
                              else "តម្លៃនៅក្នុង band")})
        if pd.notna(r.adx):
            dirn = "bull" if r.plus_di > r.minus_di else "bear"
            p.append({"key": "adx", "name": "ADX (14)", "value": num(r.adx, 1),
                      "extra": {"+DI": num(r.plus_di, 1), "−DI": num(r.minus_di, 1)},
                      "state": dirn if r.adx >= 20 else None,
                      "text": "និន្នាការខ្លាំង" if r.adx >= 25 else ("និន្នាការមធ្យម" if r.adx >= 20 else "និន្នាការខ្សោយ/ដើរចំហៀង")})
        if pd.notna(r.stoch_k):
            s = "bear" if r.stoch_k >= 80 else ("bull" if r.stoch_k <= 20 else None)
            p.append({"key": "stochrsi", "name": "Stochastic RSI", "value": num(r.stoch_k, 1),
                      "extra": {"%D": num(r.stoch_d, 1)}, "state": s,
                      "text": "Overbought (≥80)" if r.stoch_k >= 80 else ("Oversold (≤20)" if r.stoch_k <= 20 else "កណ្តាល")})
        if pd.notna(r.rel_vol):
            p.append({"key": "volume", "name": "Volume", "value": num(r.rel_vol, 2), "state": None,
                      "text": "Volume ខ្ពស់ជាងមធ្យម" if r.rel_vol >= 1.5 else "Volume ធម្មតា/ទាប"})
        else:
            p.append({"key": "volume", "name": "Volume", "value": None, "state": None,
                      "text": "គ្មាន Volume ដែលទុកចិត្តបាន (Spot Forex ភាគច្រើនមិនផ្តល់)"})
        return p

    # ------------------------------------------------------------ public API
    def analysis(self, tf: str | None = None) -> dict:
        tf = (tf or self.cfg.primary_tf).upper()
        feed, ind, frame = self._prep(tf)         # បរាជ័យ => MarketDataError
        rows = [self._tf_row(t) for t in TIMEFRAMES]
        overall = self._overall(rows)
        gate = self._gate(feed, ind)
        ev = signals.evaluate_latest(ind, frame, self.cfg, gate) if len(ind) else \
            signals._empty(gate or ["មិនមានទិន្នន័យ"], self.cfg)

        tf_signal, override = ev["signal"], False
        if self.cfg.require_mtf_agreement and ev["signal"] in ("BUY", "SELL"):
            opposite = (ev["signal"] == "BUY" and overall["direction"] == "bear") or \
                       (ev["signal"] == "SELL" and overall["direction"] == "bull")
            if opposite:
                override = True
                ev["signal"], ev["signal_label"] = "NEUTRAL", i18n.SIGNAL_LABEL["NEUTRAL"]
                ev["reasons"] = [f"Signal {tf} = {tf_signal} ប៉ុន្តែ Multi-Timeframe ប្រឆាំង "
                                 f"({overall['label']}) ‒ ដូច្នេះប្តូរជា NEUTRAL"] + ev["reasons"]

        daily = None
        try:
            daily = self._prep("D1")[0].df if tf != "D1" else feed.df
        except MarketDataError:
            pass
        price = self._price(feed)
        cur = price["last"] if price["last"] is not None else feed.latest_price
        lv = levels.analyse(ind, cur, daily, self.cfg) if (len(ind) and cur) else None
        reg = regime.detect(ind, lv, self.cfg) if len(ind) else None
        entry = float(ind["close"].iloc[-1]) if len(ind) else None
        atr_v = float(ind["atr"].iloc[-1]) if len(ind) and pd.notna(ind["atr"].iloc[-1]) else None
        setup = trade.build_setup(ev["signal"], entry, atr_v, cur, lv, self.cfg) if entry and atr_v else None

        warnings = list(feed.warnings)
        if not market_open(self.clock()):
            warnings.append("ទីផ្សារ XAUUSD បិទ (ចុងសប្តាហ៍) ‒ តម្លៃជាតម្លៃបិទចុងក្រោយ")
        warnings += ev["reasons"][:0]
        age = (self.clock() - feed.latest_time.timestamp()) if feed.latest_time is not None else None
        summary = self._summary(tf, ev, overall, reg, setup)

        tail = ind.tail(120)
        return {
            "meta": {"provider": feed.provider, "symbol": feed.symbol, "instrument": feed.instrument,
                     "timeframe": tf, "last_candle_time": iso(ind.index[-1]) if len(ind) else None,
                     "fetched_at": iso(feed.fetched_at), "generated_at": iso(self.clock()),
                     "market_open": market_open(self.clock()), "data_age_sec": num(age, 0),
                     "stale": feed.stale, "fallback": feed.fallback, "candles": int(len(ind)),
                     "warnings": warnings, "engine": "rule-based (គ្មាន AI)"},
            "price": price,
            "engine": {**{k: v for k, v in ev.items() if k != "candle_time"},
                       "tf_signal": tf_signal, "mtf_override": override, "summary": summary,
                       "score_note": "ពិន្ទុនេះជាពិន្ទុលក្ខខណ្ឌបច្ចេកទេស មិនមែនប្រូបាប៊ីលីតេឈ្នះទេ"},
            "trend": reg,
            "mtf": {"rows": rows, "overall": overall},
            "levels": lv, "setup": setup,
            "indicators": self._panel(ind, feed),
            "history": [{**h, "time": iso(h["time"])} for h in
                        (signals.history(ind, frame) if len(frame) else [])],
            "chart": {
                "labels": [t.strftime("%m-%d %H:%M") for t in tail.index],
                "close": [num(v) for v in tail["close"]], "ema20": [num(v) for v in tail["ema20"]],
                "ema50": [num(v) for v in tail["ema50"]], "bb_up": [num(v) for v in tail["bb_up"]],
                "bb_low": [num(v) for v in tail["bb_low"]]} if len(ind) else None,
        }

    def _summary(self, tf, ev, overall, reg, setup) -> str:
        sg = ev["signal"]
        head = {"BUY": "លក្ខខណ្ឌបច្ចេកទេសគាំទ្រការទិញ", "SELL": "លក្ខខណ្ឌបច្ចេកទេសគាំទ្រការលក់",
                "NEUTRAL": "លក្ខខណ្ឌមិនទាន់ច្បាស់ ‒ អព្យាក្រឹត"}[sg]
        parts = [f"{head} លើ {tf} (ពិន្ទុ {ev['score']:.0f}/100)។"]
        if reg and reg.get("available"):
            parts.append(f"ស្ថានភាព: {reg['trend_label']}, {reg['volatility_label']}។")
        parts.append(overall["text"] + "។")
        if setup:
            parts.append(f"កម្រិតយោង: Entry {setup['entry']}, SL {setup['stop_loss']} (ATR×{setup['sl_atr_mult']:g})។")
        return " ".join(parts)

    def mtf(self) -> dict:
        rows = [self._tf_row(t) for t in TIMEFRAMES]
        return {"rows": rows, "overall": self._overall(rows)}

    def history(self, tf: str | None = None) -> list:
        _, ind, frame = self._prep((tf or self.cfg.primary_tf).upper())
        return [{**h, "time": iso(h["time"])} for h in signals.history(ind, frame, 30)] if len(frame) else []

    def backtest(self, tf: str | None = None, rr: float | None = None) -> dict:
        tf = (tf or self.cfg.primary_tf).upper()
        rr = rr or self.cfg.bt_rr
        feed, ind, frame = self._prep(tf)
        key = (tf, rr, feed.fetched_at // self.cfg.bt_ttl if self.cfg.bt_ttl else feed.fetched_at)
        with self._lock:
            if key in self._bt_cache:
                return self._bt_cache[key]
            res = bt.run(ind, frame, self.cfg, rr, tf) if len(ind) else {
                "ok": False, "error": "មិនមានទិន្នន័យ", "timeframe": tf}
            res["data"] = {"provider": feed.provider, "instrument": feed.instrument,
                           "candles_available": int(len(ind))}
            self._bt_cache = {k: v for k, v in self._bt_cache.items() if k[0] != tf or k == key}
            self._bt_cache[key] = res
            return res

    def legacy_market(self) -> dict:
        """ទម្រង់ដូច /api/market ចាស់ (រក្សាភាពឆបគ្នា)។"""
        a = self.analysis()
        feed, ind, _ = self._prep(a["meta"]["timeframe"])
        r = ind.iloc[-1]
        return {"snapshot": {"symbol": feed.symbol, "price": a["price"]["last"],
                             "change_24h_pct": a["price"]["change_24h_pct"],
                             "rsi": num(r.rsi, 1), "macd": num(r.macd, 3),
                             "macd_signal": num(r.macd_signal, 3), "ema20": num(r.ema20),
                             "ema50": num(r.ema50), "atr": num(r.atr)},
                "chart": {k: a["chart"][k] for k in ("labels", "close", "ema20", "ema50")}}
