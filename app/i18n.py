"""អត្ថបទខ្មែរទាំងអស់សម្រាប់ Backend (Frontend ប្រើអត្ថបទដែលបានបញ្ជូនមក)។"""

SIGNAL_LABEL = {"BUY": "ទិញ", "SELL": "លក់", "NEUTRAL": "អព្យាក្រឹត"}
DIR_LABEL = {"bull": "ឡើង", "bear": "ចុះ", "mixed": "ច្របូកច្របល់"}

COND_BULL = {
    "trend_price": "តម្លៃនៅលើ EMA20 និង EMA50",
    "ema_cross": "EMA20 នៅលើ EMA50",
    "rsi": "RSI លើ 50",
    "macd": "MACD នៅលើ Signal",
    "adx": "ADX ≥ {adx_min:g} (និន្នាការមានកម្លាំង) ហើយ +DI > −DI",
    "price_action": "Price action បញ្ជាក់ម៉ូមង់ទឹមឡើង",
}
COND_BEAR = {
    "trend_price": "តម្លៃនៅក្រោម EMA20 និង EMA50",
    "ema_cross": "EMA20 នៅក្រោម EMA50",
    "rsi": "RSI ក្រោម 50",
    "macd": "MACD នៅក្រោម Signal",
    "adx": "ADX ≥ {adx_min:g} (និន្នាការមានកម្លាំង) ហើយ −DI > +DI",
    "price_action": "Price action បញ្ជាក់ម៉ូមង់ទឹមចុះ",
}
COND_NAME = {
    "trend_price": "តម្លៃធៀប EMA", "ema_cross": "EMA20/EMA50", "rsi": "RSI",
    "macd": "MACD", "adx": "ADX", "price_action": "Price action",
}

TREND_LABEL = {
    "strong_bull": "និន្នាការឡើងខ្លាំង",
    "weak_bull": "និន្នាការឡើងខ្សោយ",
    "strong_bear": "និន្នាការចុះខ្លាំង",
    "weak_bear": "និន្នាការចុះខ្សោយ",
    "sideways": "ទីផ្សារដើរចំហៀង",
}
VOL_LABEL = {"high": "ភាពប្រែប្រួលខ្ពស់", "low": "ភាពប្រែប្រួលទាប", "normal": "ភាពប្រែប្រួលធម្មតា"}
STRUCT_LABEL = {"breakout": "ការទម្លុះឡើង (Breakout)", "breakdown": "ការទម្លុះចុះ (Breakdown)"}
