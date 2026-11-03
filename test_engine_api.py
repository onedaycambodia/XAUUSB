import json
import pathlib
import re
import unittest

import pandas as pd

from app.config import TIMEFRAMES, public_config
from app.engine import Analyzer
from app.provider import MarketDataError
from tests.synth import NOW, FakeMarket, cfg, make_candles

ROOT = pathlib.Path(__file__).resolve().parent.parent


def analyzer(**kw):
    fm = kw.pop("market", None) or FakeMarket()
    return Analyzer(fm, kw.pop("c", None) or cfg(), clock=lambda: NOW)


def strict_json(obj):
    return json.dumps(obj, allow_nan=False)      # NaN/Inf => ValueError


class TestEngine(unittest.TestCase):
    def test_every_timeframe_analysis(self):
        a = analyzer()
        for tf in TIMEFRAMES:
            r = a.analysis(tf)
            self.assertEqual(r["meta"]["timeframe"], tf)
            self.assertIn(r["engine"]["signal"], ("BUY", "SELL", "NEUTRAL"))
            self.assertEqual(len(r["mtf"]["rows"]), 6)
            self.assertEqual(len(r["indicators"]) >= 10, True)
            strict_json(r)                                   # JSON សុវត្ថិភាព
            self.assertEqual(r["meta"]["engine"], "rule-based (គ្មាន AI)")

    def test_mtf_rows_fields(self):
        r = analyzer().mtf()
        for row in r["rows"]:
            for k in ("tf", "trend", "rsi", "macd", "ema_alignment", "adx", "score", "signal"):
                self.assertIn(k, row)
        self.assertEqual(r["overall"]["direction"], "bull")
        self.assertTrue(r["overall"]["confirmed"])

    def test_buy_setup_present_when_confirmed(self):
        a = analyzer(market=FakeMarket(pa={"H1": "bull"}))
        r = a.analysis("H1")
        self.assertEqual(r["engine"]["signal"], "BUY")
        s = r["setup"]
        self.assertEqual(s["side"], "BUY")
        self.assertLess(s["stop_loss"], s["entry"])
        self.assertEqual(len(s["targets"]), 3)
        risk = s["entry"] - s["stop_loss"]
        for t in s["targets"]:
            self.assertAlmostEqual(t["price"] - s["entry"], risk * t["rr"], delta=0.02)
        self.assertTrue(any("EMA" in x for x in r["engine"]["reasons"]))

    def test_sell_setup(self):
        a = analyzer(market=FakeMarket(default="down", pa={"H1": "bear"}))
        r = a.analysis("H1")
        self.assertEqual(r["engine"]["signal"], "SELL")
        self.assertGreater(r["setup"]["stop_loss"], r["setup"]["entry"])

    def test_neutral_has_no_setup(self):
        r = analyzer(market=FakeMarket(default="flat")).analysis("H1")
        self.assertEqual(r["engine"]["signal"], "NEUTRAL")
        self.assertIsNone(r["setup"])

    def test_mtf_opposition_overrides_to_neutral(self):
        kinds = {t: "down" for t in TIMEFRAMES}
        kinds["H1"] = "up"
        a = analyzer(market=FakeMarket(kinds=kinds, pa={"H1": "bull"}))
        r = a.analysis("H1")
        self.assertEqual(r["engine"]["tf_signal"], "BUY")
        self.assertEqual(r["engine"]["signal"], "NEUTRAL")
        self.assertTrue(r["engine"]["mtf_override"])
        self.assertIsNone(r["setup"])
        off = cfg(require_mtf_agreement=False)
        r2 = analyzer(market=FakeMarket(kinds=kinds, pa={"H1": "bull"}), c=off).analysis("H1")
        self.assertEqual(r2["engine"]["signal"], "BUY")

    def test_insufficient_candles_neutral(self):
        r = analyzer(market=FakeMarket(n=60)).analysis("H1")
        self.assertEqual(r["engine"]["signal"], "NEUTRAL")
        self.assertTrue(any("ទិន្នន័យមិនគ្រប់" in x for x in r["engine"]["reasons"]))
        strict_json(r)

    def test_missing_other_timeframe_is_handled(self):
        r = analyzer(market=FakeMarket(fail=("H4", "D1"))).analysis("H1")
        bad = [x for x in r["mtf"]["rows"] if not x["available"]]
        self.assertEqual({x["tf"] for x in bad}, {"H4", "D1"})
        self.assertEqual(r["meta"]["timeframe"], "H1")
        self.assertIn("pivots", r["levels"])             # ធ្លាក់មកវិញ: គ្មាន D1 => pivots None
        self.assertIsNone(r["levels"]["pivots"])
        strict_json(r)

    def test_primary_failure_raises_marketdataerror(self):
        with self.assertRaises(MarketDataError):
            analyzer(market=FakeMarket(fail=("H1",))).analysis("H1")

    def test_stale_data_forces_neutral_when_market_open(self):
        class OldMarket(FakeMarket):
            def get(self, tf):
                f = super().get(tf)
                f.df = f.df.iloc[:-30]             # ចុងក្រោយយឺត 30 candle
                return f
        r = analyzer(market=OldMarket(pa={"H1": "bull"})).analysis("H1")
        self.assertEqual(r["engine"]["signal"], "NEUTRAL")
        self.assertTrue(any("យឺត" in x for x in r["engine"]["reasons"]))

    def test_change_24h_matches_manual_calc(self):
        fm = FakeMarket()
        r = analyzer(market=fm).analysis("H1")
        full = fm.get("M5").full
        last = full.close.iloc[-1]
        ref = full[full.index <= full.index[-1] - pd.Timedelta(hours=24)].close.iloc[-1]
        self.assertAlmostEqual(r["price"]["change_24h_pct"], (last / ref - 1) * 100, places=2)
        self.assertAlmostEqual(r["price"]["last"], last, places=2)

    def test_short_history_has_no_24h_value(self):
        r = analyzer(market=FakeMarket(n=120)).analysis("H1")     # M5 120 candle = 10 ម៉ោង
        self.assertIsNone(r["price"]["change_24h_pct"])

    def test_history_and_regime_fields(self):
        r = analyzer().analysis("H1")
        self.assertTrue(r["history"])
        self.assertIn(r["trend"]["trend"], ("strong_bull", "weak_bull", "strong_bear", "weak_bear", "sideways"))
        self.assertIn(r["trend"]["volatility"], ("high", "low", "normal"))
        up = analyzer().analysis("H1")["trend"]["trend"]
        dn = analyzer(market=FakeMarket(default="down")).analysis("H1")["trend"]["trend"]
        nz = analyzer(market=FakeMarket(default="noise")).analysis("H1")
        fl = nz["trend"]["trend"]
        self.assertLess([i for i in nz["indicators"] if i["key"] == "adx"][0]["value"], 25)
        self.assertEqual(up, "strong_bull")
        self.assertEqual(dn, "strong_bear")
        self.assertEqual(fl, "sideways")

    def test_backtest_endpoint_logic_and_cache(self):
        a = analyzer(market=FakeMarket(n=900))
        r1 = a.backtest("H1")
        self.assertTrue(r1["ok"])
        self.assertIs(a.backtest("H1"), r1)               # cache
        strict_json(r1)

    def test_legacy_market_shape(self):
        d = analyzer().legacy_market()
        for k in ("price", "rsi", "macd", "macd_signal", "ema20", "ema50", "atr", "change_24h_pct"):
            self.assertIn(k, d["snapshot"])
        for k in ("labels", "close", "ema20", "ema50"):
            self.assertIn(k, d["chart"])

    def test_public_config_hides_secrets(self):
        c = cfg(twelvedata_key="SECRET123")
        self.assertNotIn("SECRET123", json.dumps(public_config(c)))


class TestNoAI(unittest.TestCase):
    BAD = re.compile(r"anthropic|openai|gemini|claude|chatgpt|ANTHROPIC_API_KEY|CLAUDE_MODEL", re.I)

    def test_no_ai_references_in_runtime_files(self):
        files = list((ROOT / "app").glob("*.py")) + [ROOT / "requirements.txt", ROOT / "Procfile",
                                                     ROOT / ".env.example", ROOT / "static" / "index.html"]
        for f in files:
            self.assertIsNone(self.BAD.search(f.read_text(encoding="utf-8")), str(f))

    def test_no_api_keys_in_frontend(self):
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"apikey|api_key|TWELVEDATA|secret", html, re.I))


try:
    from fastapi.testclient import TestClient
    HAVE_API = True
except Exception:  # noqa: BLE001
    HAVE_API = False


@unittest.skipUnless(HAVE_API, "ត្រូវការ fastapi + httpx (pip install -r requirements-dev.txt)")
class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import main
        main.ENGINE = Analyzer(FakeMarket(n=900, pa={"H1": "bull"}), main.CFG, clock=lambda: NOW)
        main.news.get_news = lambda **k: [{"title": "t", "link": "", "published": "", "source": ""}]
        cls.c = TestClient(main.app)

    def test_endpoints(self):
        c = self.c
        self.assertEqual(c.get("/health").json()["ok"], True)
        self.assertEqual(c.get("/").status_code, 200)
        for tf in TIMEFRAMES:
            r = c.get(f"/api/analysis?tf={tf}")
            self.assertEqual(r.status_code, 200, tf)
        self.assertEqual(c.get("/api/analysis?tf=H2").status_code, 400)
        self.assertEqual(len(c.get("/api/mtf").json()["rows"]), 6)
        self.assertIsInstance(c.get("/api/history?tf=H1").json(), list)
        self.assertTrue(c.get("/api/backtest?tf=H1&rr=2").json()["ok"])
        self.assertEqual(c.get("/api/backtest?tf=H1&rr=99").status_code, 422)
        self.assertNotIn("api", json.dumps(c.get("/api/config").json()).lower().replace("data_source", ""))
        self.assertIn("snapshot", c.get("/api/market").json())
        self.assertIn(c.post("/api/analyze").json()["signal"], ("BUY", "SELL", "NEUTRAL"))


if __name__ == "__main__":
    unittest.main()
