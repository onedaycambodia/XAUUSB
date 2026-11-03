"""ព័ត៌មាន (Google News RSS) ‒ មាន timeout, cache, និងមិនធ្វើឲ្យ API ដួលពេលបរាជ័យ។"""
import time

import feedparser
import requests

FEEDS = [
    "https://news.google.com/rss/search?q=gold+price+OR+XAUUSD&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=Federal+Reserve+OR+inflation+OR+US+dollar&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=geopolitics+OR+safe+haven+gold&hl=en-US&gl=US&ceid=US:en",
]
_cache = {"t": 0.0, "v": []}


def get_news(limit: int = 12, ttl: int = 600, timeout: float = 8.0) -> list[dict]:
    now = time.time()
    if _cache["v"] and now - _cache["t"] < ttl:
        return _cache["v"][:limit]
    seen, items = set(), []
    for url in FEEDS:
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            feed = feedparser.parse(r.content)
        except Exception:
            continue
        for e in feed.entries[:8]:
            title = e.get("title", "").strip()
            if not title or title in seen:
                continue
            seen.add(title)
            src = e.get("source", {})
            items.append({"title": title, "link": e.get("link", ""),
                          "published": e.get("published", ""),
                          "source": src.get("title", "") if hasattr(src, "get") else ""})
    if items:
        _cache.update(t=now, v=items)
        return items[:limit]
    return _cache["v"][:limit]  # បរាជ័យ => ប្រើ cache ចាស់ (ឬ បញ្ជីទទេ)
