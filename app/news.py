import feedparser

FEEDS = [
    "https://news.google.com/rss/search?q=gold+price+OR+XAUUSD&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=Federal+Reserve+OR+inflation+OR+US+dollar&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=geopolitics+OR+safe+haven+gold&hl=en-US&gl=US&ceid=US:en",
]


def get_news(limit: int = 12) -> list[dict]:
    seen, items = set(), []
    for url in FEEDS:
        try:
            feed = feedparser.parse(url)
        except Exception:
            continue
        for e in feed.entries[:8]:
            title = e.get("title", "").strip()
            if not title or title in seen:
                continue
            seen.add(title)
            items.append(
                {
                    "title": title,
                    "link": e.get("link", ""),
                    "published": e.get("published", ""),
                    "source": getattr(e.get("source", {}), "get", lambda *_: "")(
                        "title", ""
                    ),
                }
            )
    return items[:limit]
