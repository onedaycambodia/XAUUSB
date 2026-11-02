import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from . import analyze as ai  # noqa: E402
from . import data, news  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
app = FastAPI(title="XAUUSD Analyzer")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

_cache: dict = {}
CACHE_SECONDS = 300  # protect API credits


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/api/market")
def market():
    try:
        df = data.add_indicators(data.get_ohlc())
    except Exception as e:
        raise HTTPException(502, f"Market data error: {e}")
    if df.empty:
        raise HTTPException(502, "No market data returned")
    return {
        "snapshot": data.snapshot(df),
        "chart": data.chart_series(df),
        "news": news.get_news(),
    }


@app.post("/api/analyze")
def analyze(lang: str = "Khmer"):
    now = time.time()
    hit = _cache.get(lang)
    if hit and now - hit["t"] < CACHE_SECONDS:
        return hit["v"]
    try:
        df = data.add_indicators(data.get_ohlc())
        result = ai.analyze(data.snapshot(df), news.get_news(), lang)
    except Exception as e:
        raise HTTPException(502, f"Analysis error: {e}")
    result["generated_at"] = int(now)
    _cache[lang] = {"t": now, "v": result}
    return result


@app.get("/health")
def health():
    return {"ok": True}
