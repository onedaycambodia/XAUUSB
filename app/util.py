import math
from datetime import datetime, timezone


def num(x, nd=2):
    """បម្លែងទៅជា float សុវត្ថិភាព JSON: NaN/Inf => None។"""
    try:
        if x is None:
            return None
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, nd)


def iso(ts):
    if ts is None:
        return None
    try:
        t = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        if isinstance(t, (int, float)):
            t = datetime.fromtimestamp(t, tz=timezone.utc)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None
