import json
import os
import re

import anthropic

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")

SYSTEM = """You are a cautious gold (XAUUSD) market analyst.
You receive technical indicators and recent headlines. Weigh both and reply with ONLY a JSON object:
{
  "signal": "BUY" | "SELL" | "NEUTRAL",
  "confidence": 0-100,
  "summary": "2-3 sentences",
  "drivers": ["key factors, max 4"],
  "risks": ["key risks, max 3"],
  "levels": {"support": number, "resistance": number}
}
Be honest about uncertainty. Prefer NEUTRAL when evidence conflicts. Reply in the language requested."""


def analyze(tech: dict, news: list[dict], lang: str = "Khmer") -> dict:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    headlines = "\n".join(f"- {n['title']}" for n in news)
    user = (
        f"Language for text fields: {lang}\n\n"
        f"Technical snapshot (1h candles):\n{json.dumps(tech, indent=2)}\n\n"
        f"Recent headlines:\n{headlines}"
    )
    msg = client.messages.create(
        model=MODEL,
        max_tokens=800,
        system=SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    text = msg.content[0].text
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("Model did not return JSON")
    return json.loads(match.group(0))
