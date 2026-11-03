# XAUUSD Analyzer — ម៉ាស៊ីនវិភាគបច្ចេកទេស (Rule-Based, គ្មាន AI)

ប្រព័ន្ធវិភាគមាស XAU/USD ដែលគណនាទាំងអស់ដោយ **Python + រូបមន្តគណិតវិទ្យា + ច្បាប់កំណត់ទុក**។
មិនមាន Claude / ChatGPT / Gemini / LLM ឬ API ខាងក្រៅណាមួយសម្រាប់វិភាគឡើយ។

## លក្ខណៈពិសេស
- Indicator: RSI(14), MACD(12,26,9), EMA 9/20/50/200, ATR(14), Bollinger(20,2), ADX(14), Stochastic RSI, Volume (តែពេលទិន្នន័យ Volume ទុកចិត្តបាន)
- Signal Engine: ពិន្ទុ 0-100 តាមទម្ងន់ដែលកែបាន។ **ពិន្ទុជា "ពិន្ទុលក្ខខណ្ឌបច្ចេកទេស" មិនមែនប្រូបាប៊ីលីតេឈ្នះទេ**
  - BUY/SELL ត្រូវការ៖ ពិន្ទុ ≥ កម្រិត, លក្ខខណ្ឌពេញ ≥ 4 ក្នុង 6, **ADX បញ្ជាក់** និង **Price action បញ្ជាក់** (មិនអាចចេញពី indicator តែមួយ)
  - ទិន្នន័យមិនគ្រប់ / យឺត / ប្រឆាំងគ្នា / និន្នាការខ្សោយ => NEUTRAL ព្រមទាំងបង្ហាញមូលហេតុ
- Multi-Timeframe: M5, M15, M30, H1, H4, D1 (ទម្ងន់ TF ខ្ពស់ធំជាង) ‒ បើ MTF ប្រឆាំង signal ចម្បង => NEUTRAL (បិទបានដោយ `REQUIRE_MTF_AGREEMENT=false`)
- Support/Resistance: swing high/low ដែលបានបញ្ជាក់ (គ្មាន look-ahead), ចង្កោមតាម ATR, ត្រូវប៉ះ ≥ 2 ដង, Pivot Classic ពី D1, Breakout/Breakdown (បិទ 2 candle)
- Entry/SL/TP: SL = ATR×1.5, TP = R:R 1:1, 1:2, 1:3; Setup អស់សុពលភាពពេលប៉ះ SL ឬតម្លៃឆ្ងាយពី Entry
- ស្ថានភាពទីផ្សារ: ឡើង/ចុះ ខ្លាំង/ខ្សោយ, ដើរចំហៀង, ភាពប្រែប្រួលខ្ពស់/ទាប, Breakout/Breakdown
- Backtest: Entry ត្រង់ open របស់ candle បន្ទាប់ (គ្មាន look-ahead), spread + slippage + commission, Train/Test តាមពេល, Win Rate, Profit Factor, Max Drawdown, Net Profit
- Dashboard ខ្មៅ-មាស ភាសាខ្មែរទាំងស្រុង

## ប្រភពទិន្នន័យ (Spot)
| ប្រភព | ការកំណត់ | ចំណាំ |
|---|---|---|
| **Twelve Data** (ណែនាំ) | `TWELVEDATA_API_KEY=...` | XAU/USD Spot, គាំទ្រ M5–D1 ដោយផ្ទាល់។ Free plan ~800 request/ថ្ងៃ, 8/នាទី ‒ ប្រព័ន្ធមាន cache ដើម្បីកុំលើស |
| Yahoo Finance | (លំនាំដើម បើគ្មាន key) | `XAUUSD=X` តម្លៃ Spot បែប indicative; H4 បង្កើតពី H1; M5–M30 ទិន្នន័យតែ ~30 ថ្ងៃចុងក្រោយ |

- **មិនប្រើ GC=F (Futures) ជា Spot ទេ**។ បើកំណត់ `GOLD_SYMBOL=GC=F` Dashboard នឹងព្រមានថាជា Futures។
- Dashboard បង្ហាញ ប្រភពទិន្នន័យ, Timeframe, ពេល candle ចុងក្រោយ, ពេលទាញទិន្នន័យ, ភាពចាស់នៃទិន្នន័យ, ទីផ្សារបើក/បិទ។
- `%` ប្តូរ 24 ម៉ោង គណនាពីតម្លៃ M5 ចុងក្រោយ ធៀប candle មុន 24 ម៉ោងពិត (បើមានចន្លោះទីផ្សារបិទ នឹងបញ្ជាក់ក្នុង banner)។
- Volume: Spot Forex ភាគច្រើនគ្មាន Volume => Dashboard ប្រាប់ថា "គ្មាន Volume ដែលទុកចិត្តបាន" ហើយមិនប្រើវា។
- ប្រភពបរាជ័យ: ប្រើ cache ចាស់ (មានស្លាក stale) រហូត 6 ម៉ោង; បើបរាជ័យជាប់ => API ឆ្លើយ 503 ជាភាសាខ្មែរ។

## API Endpoints
| Endpoint | ការពិពណ៌នា |
|---|---|
| `GET /api/analysis?tf=H1` | ការវិភាគពេញ (signal, MTF, S/R, setup, indicator, history, chart) |
| `GET /api/mtf` | តារាង Multi-Timeframe |
| `GET /api/history?tf=H1` | ប្រវត្តិការប្តូរ signal |
| `GET /api/backtest?tf=H1&rr=2` | Backtest (cache 30 នាទី) |
| `GET /api/news`, `GET /api/config`, `GET /health` | ព័ត៌មាន / ការកំណត់ (គ្មាន key) / health check |
| `GET /api/market`, `POST /api/analyze` | endpoint ចាស់ (រក្សាភាពឆបគ្នា, ឥឡូវឆ្លើយដោយ rule engine) |

## ដំណើរការក្នុងម៉ាស៊ីន
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # ដាក់ TWELVEDATA_API_KEY (ជម្រើស)
uvicorn app.main:app --reload
# បើក http://localhost:8000
```

## Deploy លើ Railway
1. Push ឯកសារទាំងអស់ទៅ GitHub (មាន `Procfile`, `railway.json`, `requirements.txt` រួចហើយ)។
2. Railway → **New Project → Deploy from GitHub repo**។
3. **Variables** ដាក់៖ `TWELVEDATA_API_KEY` (ណែនាំ) — ផ្សេងៗជាជម្រើស (មើល `.env.example`)។
   - **លែងត្រូវការ** `ANTHROPIC_API_KEY` និង `CLAUDE_MODEL` — លុបចេញបាន។
4. Railway ប្រើ start command `uvicorn app.main:app --host 0.0.0.0 --port $PORT` និង health check `/health` ដោយស្វ័យប្រវត្តិ។
5. **Settings → Networking → Generate Domain** ដើម្បីបើក Dashboard។

## Environment Variables
| ឈ្មោះ | លំនាំដើម | ន័យ |
|---|---|---|
| `TWELVEDATA_API_KEY` | (ទទេ) | ប្រភព Spot សំខាន់; ទទេ => Yahoo |
| `GOLD_SYMBOL` | `XAUUSD=X` | Symbol Yahoo (កុំប្រើ `GC=F`) |
| `PRIMARY_TIMEFRAME` | `H1` | Timeframe ចម្បង |
| `HTTP_TIMEOUT` / `HTTP_RETRIES` | `10` / `2` | timeout និងចំនួន retry |
| `SIGNAL_THRESHOLD` | `70` | ពិន្ទុអប្បបរមាសម្រាប់ BUY/SELL |
| `MIN_CONDITIONS` | `4` | លក្ខខណ្ឌអប្បបរមា (ក្នុង 6) |
| `ADX_MIN` / `ADX_STRONG` | `20` / `25` | កម្រិតកម្លាំងនិន្នាការ |
| `SL_ATR_MULT` | `1.5` | មេគុណ ATR សម្រាប់ Stop Loss |
| `REQUIRE_MTF_AGREEMENT` | `true` | MTF ប្រឆាំង => NEUTRAL |
| `SCORE_WEIGHTS`, `MTF_WEIGHTS` | JSON | ទម្ងន់ (ធ្វើឲ្យផលបូក 100 ដោយស្វ័យប្រវត្តិ) |
| `BT_*` | មើល `.env.example` | spread, slippage, commission, lot, R:R, ... |

## សាកល្បង (Test)
```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests -t .     # ឬ pytest
```
Test ប្រើទិន្នន័យ **សំយោគ (synthetic)** ដែលបង្កើតដោយកូដ (`tests/synth.py`) សម្រាប់ផ្ទៀងផ្ទាត់តក្កវិជ្ជា ‒ មិនមែនទិន្នន័យទីផ្សារ ហើយមិនត្រូវបានប្រើក្នុង production ទេ។
`TestAPI` ត្រូវការ fastapi + httpx (ដំឡើងដោយ requirements-dev)។

## ការកំណត់ដែលត្រូវដឹង
- Railway filesystem មិនអចិន្ត្រៃយ៍ ‒ ប្រវត្តិសញ្ញាគណនាឡើងវិញពីទិន្នន័យ candle ពិតរាល់ពេល (មិនរក្សាទុកក្នុង DB)។
- Backtest ត្រូវការប្រវត្តិគ្រប់គ្រាន់: M5–M30 តិច ដូច្នេះ trade ក៏តិច; Dashboard ព្រមានពេល Train/Test < 30 trade។
- ព័ត៌មានជាចំណងជើងអង់គ្លេសពី Google News (មិនបកប្រែ ព្រោះមិនប្រើ AI)។
- លទ្ធផលអតីតកាលមិនធានាអនាគត។ ទាំងអស់ជាឯកសារយោងវិភាគ មិនមែនដំបូន្មានហិរញ្ញវត្ថុទេ។
