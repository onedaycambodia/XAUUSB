import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from . import news  # noqa: E402
from .config import TIMEFRAMES, Settings, public_config  # noqa: E402
from .engine import Analyzer  # noqa: E402
from .provider import MarketData, MarketDataError  # noqa: E402

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("xau")

BASE = Path(__file__).resolve().parent.parent
CFG = Settings.from_env()
MARKET = MarketData(CFG)
ENGINE = Analyzer(MARKET, CFG)

app = FastAPI(title="XAUUSD Analyzer — ម៉ាស៊ីនវិភាគបច្ចេកទេស (Rule-Based)")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


def _tf(tf: str | None) -> str:
    tf = (tf or CFG.primary_tf).upper()
    if tf not in TIMEFRAMES:
        raise HTTPException(400, f"Timeframe មិនត្រឹមត្រូវ។ ជ្រើសរើសមួយក្នុង: {', '.join(TIMEFRAMES)}")
    return tf


def _guard(fn, *a, **k):
    try:
        return fn(*a, **k)
    except MarketDataError as e:
        raise HTTPException(503, f"បញ្ហាទិន្នន័យទីផ្សារ: {e}")
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001
        log.exception("internal error")
        raise HTTPException(500, "កំហុសក្នុងប្រព័ន្ធវិភាគ (មើល log)")


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/api/analysis")
def analysis(tf: str | None = None):
    return _guard(ENGINE.analysis, _tf(tf))


@app.get("/api/mtf")
def mtf():
    return _guard(ENGINE.mtf)


@app.get("/api/history")
def history(tf: str | None = None):
    return _guard(ENGINE.history, _tf(tf))


@app.get("/api/backtest")
def backtest(tf: str | None = None, rr: float = Query(None, ge=0.5, le=5)):
    return _guard(ENGINE.backtest, _tf(tf), rr)


@app.get("/api/news")
def get_news():
    return news.get_news(ttl=CFG.news_ttl, timeout=CFG.http_timeout)


@app.get("/api/config")
def config():
    return public_config(CFG)


# ---- ឆបគ្នាជាមួយ Frontend/API ចាស់ ----
@app.get("/api/market")
def market():
    d = _guard(ENGINE.legacy_market)
    d["news"] = news.get_news(ttl=CFG.news_ttl, timeout=CFG.http_timeout)
    return d


@app.post("/api/analyze")
def analyze(lang: str = "Khmer"):
    """endpoint ចាស់ (ធ្លាប់ហៅ AI) ‒ ឥឡូវឆ្លើយដោយ rule engine ក្នុងទម្រង់ដូចដើម។"""
    a = _guard(ENGINE.analysis)
    e, lv = a["engine"], a["levels"] or {}
    return {
        "signal": e["signal"], "confidence": e["score"], "summary": e["summary"],
        "drivers": e["reasons"][:4], "risks": [e["score_note"]] + a["meta"]["warnings"][:2],
        "levels": {"support": (lv.get("support") or {}).get("price"),
                   "resistance": (lv.get("resistance") or {}).get("price")},
        "generated_at": int(time.time()), "engine": "rule-based",
    }


@app.get("/health")
def health():
    return {"ok": True, "engine": "rule-based", "provider": public_config(CFG)["data_source"]}
