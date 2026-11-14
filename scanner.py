"""
GitHub Actions Trading Scanner - Enhanced Version v3.0
Filters & Accuracy improvements:
  1. HTF Trend Gate (1h EMA50 direction filter)
  2. ADX regime filter (reject signals in choppy market)
  3. Distance-weighted kNN + softmax calibration
  4. Volume filter (current candle vs 20-bar avg)
  5. Feedback loop: track signal outcomes (CSV) + auto-tune thresholds
  6. Per-symbol cooldown (avoid duplicate alerts)
Plus previous v2.0 features:
  - Count >= 2 (two-stage confirmation)
  - Min R:R = 1:1.8 (reject signal if not met)
  - Min S/R distance = max(0.6% price, 0.8*ATR)
  - Min confidence = 0.62 (calibrated)
  - 3 TPs based on real S/R
  - Trailing TP guidance in Persian Telegram message
  - Leverage suggestion based on SL distance
"""

import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
import time
import logging
import sys
import os
import json
import base64
from datetime import datetime, timezone, timedelta
import math
import aiohttp

try:
    import joblib
except Exception:
    joblib = None

# ---------------- Config ----------------
GITHUB_REPO = os.environ.get('GITHUB_REPOSITORY', '')
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')
GITHUB_ACTOR = os.environ.get('GITHUB_ACTOR', 'scanner-bot')

COIN_LIST = [
    'DOGE/USDT','SHIB/USDT','PEPE/USDT','WIF/USDT','BONK/USDT','FLOKI/USDT','MEME/USDT',
    'KOMA/USDT','DOGS/USDT','NEIROETH/USDT','1000RATS/USDT','ORDI/USDT','PIPPIN/USDT',
    'BAN/USDT','1000SHIB/USDT','OM/USDT','CHILLGUY/USDT','PONKE/USDT','BOME/USDT',
    'MYRO/USDT','PEOPLE/USDT','PENGU/USDT','SPX/USDT','1000BONK/USDT','PNUT/USDT',
    'FARTCOIN/USDT','HIPPO/USDT','AIXBT/USDT','BRETT/USDT','VINE/USDT','MOODENG/USDT',
    'MUBARAK/USDT','MEW/USDT','POPCAT/USDT','1000FLOKI/USDT','1000CAT/USDT','ACT/USDT',
    'SLERF/USDT','DEGEN/USDT','1000PEPE/USDT'
]

EXCHANGE_ORDER = ['binance','bybit','kucoin','mexc','gateio','okx','bitget','huobi']
SR_TIMEFRAMES = ['1d', '4h', '1h', '15m', '5m']
TIMEFRAME_WEIGHTS = {'1d':5,'4h':3,'1h':2,'15m':1.5,'5m':1}

PIVOT_WINDOW = 5
MIN_TOUCHES = 2
SR_TOLERANCE_ATR = 0.5
PROXIMITY_THRESHOLD_ATR = 1.5
CSV_LOG_PATH = 'confirmed_signals_log.csv'
OUTCOME_LOG_PATH = 'signal_outcomes.csv'
PERSIST_PATH = 'prev_signals.json'
TUNE_PATH = 'tuner_state.json'

# ---- Base tuning parameters (may be overridden by tuner) ----
DEFAULT_TUNING = {
    'MIN_RR': 1.8,
    'MIN_CONFIDENCE': 0.62,
    'MIN_ADX': 20.0,
    'MIN_VOLUME_MULT': 1.1,
    'MIN_SL_PCT': 0.012,
    'MIN_SR_DISTANCE_PCT': 0.006,
    'MIN_SR_DISTANCE_ATR_MULT': 0.8,
}

# ---- HTF trend gate ----
HTF_TIMEFRAME = '1h'
HTF_EMA_PERIOD = 50

# ---- Confirmation / dedup ----
SIGNAL_STALE_MINUTES = 30
ALERT_COOLDOWN_MINUTES = 120
MAX_LEVERAGE_RISK_PCT = 4.5

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("ERROR: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required!")
    sys.exit(1)

RATE_LIMIT_DELAY = 0.5
MAX_RETRIES = 3
RETRY_DELAY = 2.0

SYMBOL_MAP = {
    '1000PEPE/USDT': {'mexc':'PEPE1000/USDT','gateio':'PEPE1000/USDT','bitget':'PEPE1000/USDT'},
    '1000BONK/USDT': {'mexc':'BONK1000/USDT'},
    '1000FLOKI/USDT': {'mexc':'FLOKI1000/USDT'},
    '1000SHIB/USDT': {'mexc':'SHIB1000/USDT'},
    '1000CAT/USDT': {'mexc':'CAT1000/USDT'},
    'WIF/USDT': {'okx': 'WIF-USDT'},
    'NEIROETH/USDT': {'okx': 'NEIRO-USDT'}
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

exchanges = {}
sr_cache = {}
prev_signals = {}
TUNING = dict(DEFAULT_TUNING)


# ---------------- GitHub helpers ----------------
async def read_file_from_github(filename):
    if not GITHUB_TOKEN or not GITHUB_REPO:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            return None
    try:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"
        headers = {'Authorization': f'token {GITHUB_TOKEN}', 'Accept': 'application/vnd.github.v3+json'}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return base64.b64decode(data['content']).decode('utf-8')
                return None
    except Exception as e:
        logger.error(f"GitHub read error: {e}")
        return None


async def write_file_to_github(filename, content, commit_message):
    if not GITHUB_TOKEN or not GITHUB_REPO:
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except:
            return False
    try:
        get_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"
        headers = {'Authorization': f'token {GITHUB_TOKEN}', 'Accept': 'application/vnd.github.v3+json'}
        sha = None
        async with aiohttp.ClientSession() as session:
            async with session.get(get_url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    sha = data['sha']
        encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
        commit_data = {
            'message': commit_message,
            'content': encoded_content,
            'committer': {'name': GITHUB_ACTOR, 'email': f'{GITHUB_ACTOR}@github.com'}
        }
        if sha:
            commit_data['sha'] = sha
        put_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"
        async with aiohttp.ClientSession() as session:
            async with session.put(put_url, headers=headers, json=commit_data) as response:
                return response.status in [200, 201]
    except Exception as e:
        logger.error(f"GitHub write error: {e}")
        return False


# ---------------- Telegram ----------------
async def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'HTML'}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=15) as response:
                if response.status != 200:
                    body = await response.text()
                    logger.error(f"Telegram non-200: {body[:200]}")
                return response.status == 200
    except Exception as e:
        logger.error(f"Telegram error: {e}")
        return False


async def send_error_notification(error_msg):
    message = f"🚨 <b>خطا در اسکنر</b>\n\n{error_msg[:500]}\n\n⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    await send_telegram_message(message)


def format_price(price):
    if price < 0.0001:
        return f"{price:.10f}"
    elif price < 0.01:
        return f"{price:.8f}"
    elif price < 1:
        return f"{price:.6f}"
    else:
        return f"{price:.4f}"


async def format_and_send_signals(confirmed_list):
    if not confirmed_list:
        return

    for signal in confirmed_list:
        direction = signal['direction']
        emoji = "🟢" if direction == "LONG" else "🔴"
        dir_text = "لانگ (خرید)" if direction == "LONG" else "شورت (فروش)"

        entry = signal['price']
        sl = signal.get('sl', 0)
        tp1 = signal.get('tp1', 0)
        tp2 = signal.get('tp2', 0)
        tp3 = signal.get('tp3', 0)
        leverage = signal.get('leverage', '۵x - ۱۰x')

        if direction == "LONG":
            risk_pct = ((entry - sl) / entry) * 100 if entry else 0
            rr1 = (tp1 - entry) / (entry - sl) if (entry - sl) > 0 else 0
            rr2 = (tp2 - entry) / (entry - sl) if (entry - sl) > 0 else 0
            rr3 = (tp3 - entry) / (entry - sl) if (entry - sl) > 0 else 0
            tp1_pct = ((tp1 - entry) / entry) * 100 if entry else 0
            tp2_pct = ((tp2 - entry) / entry) * 100 if entry else 0
            tp3_pct = ((tp3 - entry) / entry) * 100 if entry else 0
        else:
            risk_pct = ((sl - entry) / entry) * 100 if entry else 0
            rr1 = (entry - tp1) / (sl - entry) if (sl - entry) > 0 else 0
            rr2 = (entry - tp2) / (sl - entry) if (sl - entry) > 0 else 0
            rr3 = (entry - tp3) / (sl - entry) if (sl - entry) > 0 else 0
            tp1_pct = ((entry - tp1) / entry) * 100 if entry else 0
            tp2_pct = ((entry - tp2) / entry) * 100 if entry else 0
            tp3_pct = ((entry - tp3) / entry) * 100 if entry else 0

        htf_txt = "موافق" if signal.get('htf_ok') else "—"

        message = f"""
{emoji} <b>سیگنال تأیید شده</b> {emoji}

📌 <b>{signal['symbol']}</b>
📍 جهت: <b>{dir_text}</b>
🏦 صرافی: {signal['exchange'].upper()}

━━━━━━━━━━━━━━━━━━
💰 <b>قیمت ورود:</b> <code>{format_price(entry)}</code>

🛑 <b>حد ضرر:</b> <code>{format_price(sl)}</code>
  ریسک: <b>{risk_pct:.2f}٪</b>

🎯 <b>تارگت‌ها:</b>
• هدف ۱: <code>{format_price(tp1)}</code>
   +{tp1_pct:.2f}٪   | R:R ≈ 1:{rr1:.1f}
• هدف ۲: <code>{format_price(tp2)}</code>
   +{tp2_pct:.2f}٪   | R:R ≈ 1:{rr2:.1f}
• هدف ۳: <code>{format_price(tp3)}</code>
   +{tp3_pct:.2f}٪   | R:R ≈ 1:{rr3:.1f}

━━━━━━━━━━━━━━━━━━
⚡ <b>اهرم پیشنهادی:</b> {leverage}
📊 اطمینان مدل: <b>{signal['confidence']*100:.0f}٪</b>
📉 RSI: {signal['rsi']:.1f}
📈 ADX: {signal.get('adx', 0):.1f}
🧭 روند HTF: {htf_txt}
🔄 تعداد تأیید: {signal['count']}

📋 <b>مدیریت پوزیشن (Trailing):</b>
• پس از TP1 → حد ضرر را به نقطه ورود منتقل کنید
• پس از TP2 → حد ضرر را به TP1 منتقل کنید
• در TP3 → ۷۰٪ پوزیشن را ببندید، باقی را با trailing ادامه دهید

⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d  %H:%M')} UTC
"""
        await send_telegram_message(message.strip())


# ---------------- Persistence ----------------
async def load_prev_signals():
    global prev_signals
    try:
        content = await read_file_from_github(PERSIST_PATH)
        prev_signals = json.loads(content) if content else {}
    except Exception as e:
        logger.error(f"Load prev_signals error: {e}")
        prev_signals = {}


async def save_prev_signals():
    try:
        content = json.dumps(prev_signals, ensure_ascii=False, indent=2)
        await write_file_to_github(PERSIST_PATH, content, f"Update signals - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}")
    except Exception as e:
        logger.error(f"Save error: {e}")


async def append_csv_row(path, header, row):
    try:
        existing = await read_file_from_github(path)
        new_row = ','.join(str(row.get(col, '')) for col in header)
        content = (existing.rstrip() + '\n' + new_row) if existing else (','.join(header) + '\n' + new_row)
        await write_file_to_github(path, content, f"Signal log: {row.get('symbol','?')}")
    except Exception as e:
        logger.error(f"CSV {path} error: {e}")


SIGNAL_HEADER = ['datetime_utc','symbol','direction','exchange','confidence','price','rsi','adx','count','sl','tp1','tp2','tp3','leverage','htf_ok']
OUTCOME_HEADER = ['signal_id','datetime_utc','symbol','direction','entry','sl','tp1','tp2','tp3','outcome','exit_price','resolved_at_utc']


# ---------------- Exchange helpers ----------------
def get_symbol_for_exchange(symbol, exchange_id):
    if symbol in SYMBOL_MAP and exchange_id in SYMBOL_MAP[symbol]:
        return SYMBOL_MAP[symbol][exchange_id]
    return symbol


async def create_exchange_client(exchange_id):
    try:
        cls = getattr(ccxt, exchange_id)
        client = cls({'enableRateLimit': True, 'timeout': 30000, 'rateLimit': 1200})
        try:
            await client.load_markets()
        except:
            pass
        return client
    except Exception as e:
        logger.error(f"Init {exchange_id} failed: {e}")
        return None


async def init_exchanges():
    for eid in EXCHANGE_ORDER:
        client = await create_exchange_client(eid)
        if client:
            exchanges[eid] = client
        await asyncio.sleep(0.4)


async def close_exchanges():
    for c in exchanges.values():
        try:
            await c.close()
        except:
            pass


# ---------------- Indicators & S/R ----------------
async def fetch_ohlcv_with_retry(exchange, symbol, timeframe='15m', limit=300):
    for attempt in range(MAX_RETRIES):
        try:
            mapped = get_symbol_for_exchange(symbol, exchange.id)
            await asyncio.sleep(RATE_LIMIT_DELAY)
            ohlcv = await exchange.fetch_ohlcv(mapped, timeframe=timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < 30:
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            return df.sort_values('timestamp').reset_index(drop=True)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                return None
    return None


def calculate_indicators(df):
    if df is None or len(df) < 30:
        return df
    def rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta).where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    def wavetrend(df_in, n1=10, n2=11):
        ap = (df_in['high'] + df_in['low'] + df_in['close']) / 3
        esa = ap.ewm(span=n1).mean()
        d = (ap - esa).abs().ewm(span=n1).mean()
        ci = (ap - esa) / (0.015 * d + 1e-10)
        return ci.ewm(span=n2).mean()
    def cci(df_in, period=20):
        tp = (df_in['high'] + df_in['low'] + df_in['close']) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (tp - sma) / (0.015 * mad + 1e-10)
    def adx(df_in, period=14):
        high, low, close = df_in['high'], df_in['low'], df_in['close']
        up = high.diff()
        down = -low.diff()
        plus_dm = up.where((up > down) & (up > 0), 0)
        minus_dm = down.where((down > up) & (down > 0), 0)
        tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
        plus_di = (plus_dm.rolling(window=period).mean() / (tr.rolling(window=period).mean() + 1e-10)) * 100
        minus_di = (minus_dm.rolling(window=period).mean() / (tr.rolling(window=period).mean() + 1e-10)) * 100
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10) * 100
        return dx.rolling(window=period).mean()
    try:
        df['rsi'] = rsi(df['close'])
        df['rsi_short'] = rsi(df['close'], 9)
        df['wavetrend'] = wavetrend(df)
        df['cci'] = cci(df)
        df['adx'] = adx(df)
    except:
        for col in ['rsi','rsi_short','wavetrend','cci','adx']:
            df[col] = 50
    return df


def normalize_features(df):
    for c in ['rsi','wavetrend','cci','adx','rsi_short']:
        if c in df.columns:
            std = df[c].std()
            df[c + '_norm'] = np.tanh(df[c] / std) if std > 1e-6 else 0
    return df


def detect_pivots(highs, lows, window=PIVOT_WINDOW):
    ph, pl = [], []
    for i in range(window, len(highs)-window):
        if all(highs[i] >= highs[j] for j in range(i-window, i+window+1) if j != i):
            ph.append((i, highs[i]))
        if all(lows[i] <= lows[j] for j in range(i-window, i+window+1) if j != i):
            pl.append((i, lows[i]))
    return ph, pl


def cluster_levels(levels, atr_value, tolerance_multiplier=SR_TOLERANCE_ATR):
    if not levels or atr_value <= 0:
        return []
    tol = atr_value * tolerance_multiplier
    levels = sorted(levels, key=lambda x: x[1])
    clusters, cur = [], [levels[0]]
    for lvl in levels[1:]:
        if abs(lvl[1] - cur[-1][1]) <= tol:
            cur.append(lvl)
        else:
            clusters.append((np.mean([x[1] for x in cur]), len(cur)))
            cur = [lvl]
    if cur:
        clusters.append((np.mean([x[1] for x in cur]), len(cur)))
    return clusters


def calculate_sr_levels(df, timeframe):
    if df is None or len(df) < 30:
        return [], []
    df = df.copy()
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()))
    atr = df['tr'].rolling(14).mean().iloc[-1]
    if pd.isna(atr) or atr <= 0:
        atr = (df['high'] - df['low']).mean()
    ph, pl = detect_pivots(df['high'].values, df['low'].values)
    resist = cluster_levels(ph, atr)
    support = cluster_levels(pl, atr)
    weight = TIMEFRAME_WEIGHTS.get(timeframe, 1)
    valid_resist = [(p, touches * weight, timeframe) for p, touches in resist if touches >= MIN_TOUCHES]
    valid_support = [(p, touches * weight, timeframe) for p, touches in support if touches >= MIN_TOUCHES]
    return valid_resist, valid_support


async def get_multi_timeframe_sr(exchange, symbol):
    all_r, all_s = [], []
    for tf in SR_TIMEFRAMES:
        limit = {'1d':100,'4h':200,'1h':250,'15m':300,'5m':300}.get(tf, 300)
        df = await fetch_ohlcv_with_retry(exchange, symbol, timeframe=tf, limit=limit)
        if df is not None:
            r, s = calculate_sr_levels(df, tf)
            all_r.extend(r)
            all_s.extend(s)
    all_r.sort(key=lambda x: x[1], reverse=True)
    all_s.sort(key=lambda x: x[1], reverse=True)
    return all_r, all_s


def check_sr_proximity(current_price, resistance_levels, support_levels, atr_value, signal_type):
    if atr_value is None or atr_value <= 0 or math.isnan(atr_value):
        return True
    min_dist_pct = current_price * TUNING['MIN_SR_DISTANCE_PCT']
    min_dist_atr = atr_value * TUNING['MIN_SR_DISTANCE_ATR_MULT']
    threshold = max(min_dist_pct, min_dist_atr)

    if signal_type == 'LONG':
        nearby = [r[0] for r in resistance_levels if r[0] > current_price]
        if nearby and (min(nearby) - current_price) < threshold:
            return False
    else:
        nearby = [s[0] for s in support_levels if s[0] < current_price]
        if nearby and (current_price - max(nearby)) < threshold:
            return False
    return True


# ---- HTF trend gate ----
async def check_htf_trend(exchange, symbol, direction):
    df = await fetch_ohlcv_with_retry(exchange, symbol, timeframe=HTF_TIMEFRAME, limit=HTF_EMA_PERIOD + 50)
    if df is None or len(df) < HTF_EMA_PERIOD + 2:
        return False
    ema = df['close'].ewm(span=HTF_EMA_PERIOD, adjust=False).mean().iloc[-1]
    last_close = df['close'].iloc[-1]
    if direction == 'LONG':
        return last_close > ema
    else:
        return last_close < ema


# ---- Volume filter ----
def check_volume(df, lookback=20, mult=None):
    if df is None or len(df) < lookback + 1:
        return False
    mult = mult if mult is not None else TUNING['MIN_VOLUME_MULT']
    avg = df['volume'].iloc[-lookback-1:-1].mean()
    if avg <= 0:
        return False
    return df['volume'].iloc[-1] >= avg * mult


def _round_price(p, ref_price):
    decimals = 8 if ref_price < 1 else 6
    return round(float(p), decimals)


def calculate_sl_tp_and_leverage(price, direction, atr, support_levels, resistance_levels):
    if atr is None or atr <= 0 or math.isnan(atr):
        atr = price * 0.025
    min_rr = TUNING['MIN_RR']
    min_sl_pct = TUNING['MIN_SL_PCT']

    if direction == 'LONG':
        nearby_supports = [s[0] for s in support_levels if s[0] < price]
        if nearby_supports:
            sl = max(nearby_supports) - (0.25 * atr)
        else:
            sl = price - (2.2 * atr)
        if (price - sl) < price * min_sl_pct:
            sl = price - price * min_sl_pct
        risk = price - sl
        if risk <= 0:
            return None

        resist_targets = sorted([r[0] for r in resistance_levels if r[0] > price + risk])
        atr_tp1 = price + max(min_rr * risk, 1.8 * risk)
        atr_tp2 = price + max((min_rr + 0.9) * risk, 2.8 * risk)
        atr_tp3 = price + max((min_rr + 2.0) * risk, 4.0 * risk)

        valid_r = [r for r in resist_targets if (r - price) / risk >= min_rr]
        tp1 = valid_r[0] if valid_r else atr_tp1
        valid_r2 = [r for r in valid_r if (r - price) / risk >= (min_rr + 0.9)]
        tp2 = valid_r2[0] if valid_r2 else max(atr_tp2, tp1 + 0.9 * risk)
        valid_r3 = [r for r in valid_r2 if (r - price) / risk >= (min_rr + 2.0)]
        tp3 = valid_r3[0] if valid_r3 else max(atr_tp3, tp2 + 1.1 * risk)

        if (tp1 - price) / risk < min_rr - 1e-9:
            return None
    else:
        nearby_resists = [r[0] for r in resistance_levels if r[0] > price]
        if nearby_resists:
            sl = min(nearby_resists) + (0.25 * atr)
        else:
            sl = price + (2.2 * atr)
        if (sl - price) < price * min_sl_pct:
            sl = price + price * min_sl_pct
        risk = sl - price
        if risk <= 0:
            return None

        support_targets = sorted([s[0] for s in support_levels if s[0] < price - risk], reverse=True)
        atr_tp1 = price - max(min_rr * risk, 1.8 * risk)
        atr_tp2 = price - max((min_rr + 0.9) * risk, 2.8 * risk)
        atr_tp3 = price - max((min_rr + 2.0) * risk, 4.0 * risk)

        valid_s = [s for s in support_targets if (price - s) / risk >= min_rr]
        tp1 = valid_s[0] if valid_s else atr_tp1
        valid_s2 = [s for s in valid_s if (price - s) / risk >= (min_rr + 0.9)]
        tp2 = valid_s2[0] if valid_s2 else min(atr_tp2, tp1 - 0.9 * risk)
        valid_s3 = [s for s in valid_s2 if (price - s) / risk >= (min_rr + 2.0)]
        tp3 = valid_s3[0] if valid_s3 else min(atr_tp3, tp2 - 1.1 * risk)

        if (price - tp1) / risk < min_rr - 1e-9:
            return None

    sl = _round_price(sl, price)
    tp1 = _round_price(tp1, price)
    tp2 = _round_price(tp2, price)
    tp3 = _round_price(tp3, price)

    risk_pct = abs(price - sl) / price * 100
    if risk_pct < 1.3:
        leverage = "۱۵x – ۳۰x"
    elif risk_pct < 2.0:
        leverage = "۱۰x – ۲۰x"
    elif risk_pct < 3.0:
        leverage = "۷x – ۱۵x"
    elif risk_pct < MAX_LEVERAGE_RISK_PCT:
        leverage = "۵x – ۱۰x"
    else:
        leverage = "۳x – ۷x"

    return sl, tp1, tp2, tp3, leverage


# ---------------- ML: distance-weighted kNN + softmax ----------------
def create_labels(df, forward_bars=4, threshold=0.002):
    returns = df['close'].shift(-forward_bars) / df['close'] - 1
    return np.where(returns >= threshold, 1, np.where(returns <= -threshold, -1, 0))


def lorentzian_knn_predict(df, k=8):
    """
    Distance-weighted kNN. Returns (prediction_sum, calibrated_confidence 0..1).
    Confidence = softmax(weighted vote) => sharper than raw mean.
    """
    feature_cols = ['rsi_norm','wavetrend_norm','cci_norm','adx_norm','rsi_short_norm']
    if df is None or len(df) < 60 or not all(c in df.columns for c in feature_cols):
        return 0, 0
    labels = create_labels(df)
    recent_df = df.iloc[-180:].copy()
    recent_labels = labels[-180:]
    current = recent_df[feature_cols].iloc[-1].values
    distances, valid_idx = [], []
    for i in range(0, len(recent_df)-6, 3):
        if pd.notna(recent_labels[i]) and recent_labels[i] != 0:
            hist = recent_df[feature_cols].iloc[i].values
            if not np.any(np.isnan(hist)) and not np.any(np.isnan(current)):
                d = float(np.sum(np.log(1 + np.abs(current - hist))))
                distances.append(d)
                valid_idx.append(i)
    if len(distances) < k:
        return 0, 0

    order = np.argsort(distances)[:k]
    top_d = np.array([distances[i] for i in order])
    top_labels = np.array([recent_labels[valid_idx[i]] for i in order], dtype=float)

    # Inverse-distance weights (guard against zero)
    weights = 1.0 / (top_d + 1e-6)
    weights = weights / weights.sum()

    weighted_vote = float(np.sum(weights * top_labels))  # in [-1, 1]

    # Softmax-style calibration: map |vote| in [0,1] to a smooth confidence
    # Using 2*x/(1+x) gives values approaching 1 for strong agreement,
    # and near 0 for weak agreement. Then floor at raw agreement.
    raw_agree = abs(np.mean(top_labels))
    weighted_agree = abs(weighted_vote)
    calibrated = max(raw_agree, (2.0 * weighted_agree) / (1.0 + weighted_agree))
    calibrated = min(calibrated, 0.99)

    prediction_sum = float(np.sum(top_labels))
    return prediction_sum, calibrated


# ---------------- Tuner (feedback loop) ----------------
async def load_tuning():
    global TUNING
    try:
        content = await read_file_from_github(TUNE_PATH)
        if content:
            saved = json.loads(content)
            TUNING = {**DEFAULT_TUNING, **saved}
            logger.info(f"Loaded tuning: {TUNING}")
    except Exception as e:
        logger.error(f"Load tuning error: {e}")
        TUNING = dict(DEFAULT_TUNING)


async def save_tuning():
    try:
        content = json.dumps(TUNING, ensure_ascii=False, indent=2)
        await write_file_to_github(TUNE_PATH, content, f"Tuner state - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}")
    except Exception as e:
        logger.error(f"Save tuning error: {e}")


async def resolve_pending_outcomes():
    """
    Check past confirmed signals older than 6h but without outcome.
    Fetch price history and decide whether TP1 or SL was hit first.
    Append to OUTCOME_LOG_PATH.
    """
    try:
        existing = await read_file_from_github(OUTCOME_LOG_PATH)
        if not existing:
            return
        lines = existing.strip().split('\n')
        if len(lines) < 2:
            return
        header = lines[0].split(',')
        rows = [dict(zip(header, l.split(','))) for l in lines[1:] if l]
        resolved_ids = {r['signal_id'] for r in rows if r.get('outcome')}

        # Load confirmed signals
        sig_csv = await read_file_from_github(CSV_LOG_PATH)
        if not sig_csv:
            return
        sig_lines = sig_csv.strip().split('\n')
        if len(sig_lines) < 2:
            return
        sig_header = sig_lines[0].split(',')
        sig_rows = [dict(zip(sig_header, l.split(','))) for l in sig_lines[1:] if l]

        now = datetime.now(timezone.utc)
        newly_resolved = []
        for s in sig_rows[-30:]:  # only recent ones
            sid = f"{s.get('datetime_utc','')}_{s.get('symbol','')}_{s.get('direction','')}"
            if sid in resolved_ids:
                continue
            try:
                sig_time = datetime.fromisoformat(s['datetime_utc'])
                if sig_time.tzinfo is None:
                    sig_time = sig_time.replace(tzinfo=timezone.utc)
            except:
                continue
            age = (now - sig_time).total_seconds() / 3600
            if age < 6:
                continue
            symbol = s['symbol']
            direction = s['direction']
            entry = float(s['price'])
            sl = float(s['sl'])
            tp1 = float(s['tp1'])

            # find exchange client
            eid = s.get('exchange','binance')
            ex = exchanges.get(eid)
            if not ex:
                continue
            resolved_sym = _resolve_swap_symbol(ex, eid, symbol)
            if not resolved_sym:
                continue
            df = await fetch_ohlcv_with_retry(ex, resolved_sym, timeframe='15m', limit=200)
            if df is None:
                continue
            # filter candles after signal time
            df = df[df['timestamp'] > sig_time]
            if df.empty:
                continue

            outcome = None
            exit_price = entry
            resolved_at = now.isoformat()
            for _, c in df.iterrows():
                hi, lo = c['high'], c['low']
                if direction == 'LONG':
                    if lo <= sl:
                        outcome, exit_price = 'SL', sl
                        resolved_at = c['timestamp'].isoformat()
                        break
                    if hi >= tp1:
                        outcome, exit_price = 'TP1', tp1
                        resolved_at = c['timestamp'].isoformat()
                        break
                else:
                    if hi >= sl:
                        outcome, exit_price = 'SL', sl
                        resolved_at = c['timestamp'].isoformat()
                        break
                    if lo <= tp1:
                        outcome, exit_price = 'TP1', tp1
                        resolved_at = c['timestamp'].isoformat()
                        break

            newly_resolved.append({
                'signal_id': sid,
                'datetime_utc': s.get('datetime_utc',''),
                'symbol': symbol,
                'direction': direction,
                'entry': entry,
                'sl': sl,
                'tp1': tp1,
                'tp2': s.get('tp2',''),
                'tp3': s.get('tp3',''),
                'outcome': outcome or 'OPEN',
                'exit_price': exit_price,
                'resolved_at_utc': resolved_at,
            })

        for row in newly_resolved:
            await append_csv_row(OUTCOME_LOG_PATH, OUTCOME_HEADER, row)
    except Exception as e:
        logger.error(f"resolve_pending_outcomes error: {e}")


async def auto_tune_thresholds():
    """
    Read OUTCOME_LOG_PATH, compute win-rate.
    Adjust TUNING slightly (bounded) every run:
      - win-rate < 0.40 => raise MIN_CONFIDENCE and MIN_RR
      - win-rate > 0.65 => lower MIN_CONFIDENCE slightly (more signals)
    """
    try:
        content = await read_file_from_github(OUTCOME_LOG_PATH)
        if not content:
            return
        lines = content.strip().split('\n')
        if len(lines) < 2:
            return
        header = lines[0].split(',')
        rows = [dict(zip(header, l.split(','))) for l in lines[1:] if l]
        closed = [r for r in rows if r.get('outcome') in ('TP1','TP2','TP3','SL')]
        if len(closed) < 20:
            logger.info(f"Tuner: not enough closed trades ({len(closed)}<20), skipping")
            return
        wins = sum(1 for r in closed if r['outcome'].startswith('TP'))
        win_rate = wins / len(closed)
        logger.info(f"Tuner: closed={len(closed)} win_rate={win_rate:.2%}")

        changed = False
        if win_rate < 0.40:
            new_conf = min(TUNING['MIN_CONFIDENCE'] + 0.01, 0.75)
            new_rr = min(TUNING['MIN_RR'] + 0.05, 2.5)
            if new_conf != TUNING['MIN_CONFIDENCE'] or new_rr != TUNING['MIN_RR']:
                TUNING['MIN_CONFIDENCE'] = round(new_conf, 3)
                TUNING['MIN_RR'] = round(new_rr, 3)
                changed = True
        elif win_rate > 0.65:
            new_conf = max(TUNING['MIN_CONFIDENCE'] - 0.005, 0.58)
            if new_conf != TUNING['MIN_CONFIDENCE']:
                TUNING['MIN_CONFIDENCE'] = round(new_conf, 3)
                changed = True

        if changed:
            await save_tuning()
            logger.info(f"Tuner updated: {TUNING}")
    except Exception as e:
        logger.error(f"auto_tune_thresholds error: {e}")


# ---------------- Scan helpers ----------------
def _base_quote_from_unified(sym: str):
    try:
        base, right = sym.split('/')
        return base, right.split(':')[0]
    except:
        return sym, ''


async def _ensure_swap_options(exchange, eid: str):
    try:
        opts = getattr(exchange, 'options', {}) or {}
        if opts.get('defaultType') != 'swap':
            exchange.options = {**opts, 'defaultType': 'swap'}
            try:
                await exchange.load_markets()
            except:
                pass
    except:
        pass


def _resolve_swap_symbol(exchange, eid: str, requested_symbol: str):
    try:
        mapped = get_symbol_for_exchange(requested_symbol, eid)
        markets = getattr(exchange, 'markets', {}) or {}
        if mapped in markets and markets[mapped].get('swap'):
            return mapped
        if '/' in mapped:
            b, q = _base_quote_from_unified(mapped)
            colon = f"{b}/{q}:{q}"
            if colon in markets and markets[colon].get('swap'):
                return colon
        b, q = _base_quote_from_unified(mapped)
        for m in markets.values():
            if m.get('swap') and m.get('base') == b and m.get('quote') == q:
                return m.get('symbol')
    except:
        pass
    return None


def _pick_supported_timeframe(exchange):
    tfs = getattr(exchange, 'timeframes', {}) or {}
    for tf in ['15m', '5m', '1h', '4h']:
        if tf in tfs:
            return tf
    return '15m'


def _is_signal_stale(last_seen_iso: str, minutes: int) -> bool:
    try:
        last = datetime.fromisoformat(last_seen_iso)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - last).total_seconds() > minutes * 60
    except:
        return True


def _is_in_cooldown(sym, cooldown_minutes):
    prev = prev_signals.get(sym, {})
    last_alert = prev.get('last_alert')
    if not last_alert:
        return False
    try:
        dt = datetime.fromisoformat(last_alert)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() < cooldown_minutes * 60
    except:
        return False


async def scan_market_single_run():
    logger.info(f"Starting market scan (v3.0) | TUNING={TUNING}")
    sr_cache.clear()
    for eid, ex in exchanges.items():
        await _ensure_swap_options(ex, eid)

    current_map = {}
    confirmed_list = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for i, requested_symbol in enumerate(COIN_LIST):
        logger.info(f"Processing {requested_symbol} ({i+1}/{len(COIN_LIST)})")
        for eid in EXCHANGE_ORDER:
            exchange = exchanges.get(eid)
            if not exchange:
                continue
            resolved = _resolve_swap_symbol(exchange, eid, requested_symbol)
            if not resolved:
                continue
            tf = _pick_supported_timeframe(exchange)
            df = await fetch_ohlcv_with_retry(exchange, resolved, timeframe=tf, limit=300)
            if df is None:
                continue

            df = calculate_indicators(df)
            df = normalize_features(df)

            # ---- ADX regime filter ----
            current_adx = float(df['adx'].iloc[-1]) if 'adx' in df.columns and not pd.isna(df['adx'].iloc[-1]) else 0.0
            if current_adx < TUNING['MIN_ADX']:
                logger.info(f"  Skipped {requested_symbol}: ADX {current_adx:.1f} < {TUNING['MIN_ADX']}")
                continue

            # ---- Volume filter ----
            if not check_volume(df):
                logger.info(f"  Skipped {requested_symbol}: low volume")
                continue

            prediction_sum, confidence = lorentzian_knn_predict(df)
            if confidence < TUNING['MIN_CONFIDENCE'] or abs(prediction_sum) < 2:
                continue

            current_price = float(df['close'].iloc[-1])
            current_rsi = float(df['rsi'].iloc[-1]) if 'rsi' in df.columns else 50.0
            signal_type = 'LONG' if prediction_sum > 0 else 'SHORT'
            rsi_agrees = (current_rsi < 72) if signal_type == 'LONG' else (current_rsi > 28)
            if not rsi_agrees:
                continue

            # ---- HTF trend gate ----
            if not await check_htf_trend(exchange, resolved, signal_type):
                logger.info(f"  Skipped {requested_symbol}: HTF trend mismatch")
                continue

            cache_key = f"{requested_symbol}_{eid}"
            if cache_key not in sr_cache:
                resist, support = await get_multi_timeframe_sr(exchange, resolved)
                sr_cache[cache_key] = (resist, support)
            else:
                resist, support = sr_cache[cache_key]

            df['tr'] = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()))
            atr = df['tr'].rolling(14).mean().iloc[-1]
            if pd.isna(atr):
                atr = (df['high'] - df['low']).mean()
            atr = float(atr)

            if not check_sr_proximity(current_price, resist, support, atr, signal_type):
                logger.info(f"  Skipped {requested_symbol} {signal_type}: too close to S/R")
                continue

            result = calculate_sl_tp_and_leverage(current_price, signal_type, atr, support, resist)
            if result is None:
                logger.info(f"  Skipped {requested_symbol} {signal_type}: cannot satisfy MIN_RR={TUNING['MIN_RR']}")
                continue

            sl, tp1, tp2, tp3, leverage = result

            current_map[requested_symbol] = {
                'symbol': requested_symbol,
                'exchange': eid,
                'direction': signal_type,
                'rsi': current_rsi,
                'adx': current_adx,
                'confidence': float(confidence),
                'price': current_price,
                'sl': sl,
                'tp1': tp1,
                'tp2': tp2,
                'tp3': tp3,
                'leverage': leverage,
                'atr': atr,
                'htf_ok': True,
                'ts': now_iso
            }
            logger.info(f"Signal found: {requested_symbol} {signal_type} conf={confidence:.2f} adx={current_adx:.1f}")
            break
        await asyncio.sleep(0.08)

    # Confirmation + cooldown
    alerts_to_send = []
    for sym, cur in current_map.items():
        prev = prev_signals.get(sym)
        if prev and prev.get('direction') == cur['direction'] and not _is_signal_stale(prev.get('last_seen', ''), SIGNAL_STALE_MINUTES):
            new_count = prev.get('count', 1) + 1
            prev_signals[sym] = {**cur, 'count': new_count, 'last_seen': cur['ts'], 'last_alert': prev.get('last_alert')}
        else:
            prev_signals[sym] = {**cur, 'count': 1, 'last_seen': cur['ts'], 'last_alert': (prev or {}).get('last_alert')}

    for sym, cur in current_map.items():
        data = prev_signals.get(sym)
        if not data or data.get('count', 0) < 2:
            continue
        # cooldown check
        if _is_in_cooldown(sym, ALERT_COOLDOWN_MINUTES):
            logger.info(f"  Cooldown active for {sym}, skipping alert")
            continue

        confirmed_list.append({
            'symbol': sym,
            'exchange': data['exchange'],
            'direction': data['direction'],
            'rsi': data.get('rsi', 50),
            'adx': data.get('adx', 0),
            'confidence': data['confidence'],
            'price': data['price'],
            'sl': data.get('sl'),
            'tp1': data.get('tp1'),
            'tp2': data.get('tp2'),
            'tp3': data.get('tp3'),
            'leverage': data.get('leverage', '۵x - ۱۰x'),
            'atr': data.get('atr', 0),
            'htf_ok': data.get('htf_ok', False),
            'count': data['count']
        })
        prev_signals[sym]['last_alert'] = now_iso

        await append_csv_row(CSV_LOG_PATH, SIGNAL_HEADER, {
            'datetime_utc': now_iso,
            'symbol': sym,
            'direction': data['direction'],
            'exchange': data['exchange'],
            'confidence': data['confidence'],
            'price': data['price'],
            'rsi': data.get('rsi', ''),
            'adx': data.get('adx', ''),
            'count': data['count'],
            'sl': data.get('sl', ''),
            'tp1': data.get('tp1', ''),
            'tp2': data.get('tp2', ''),
            'tp3': data.get('tp3', ''),
            'leverage': data.get('leverage', ''),
            'htf_ok': data.get('htf_ok', False),
        })

    # cleanup
    for s in list(prev_signals.keys()):
        if s not in current_map:
            prev_signals.pop(s, None)
        elif _is_signal_stale(prev_signals[s].get('last_seen', ''), SIGNAL_STALE_MINUTES):
            prev_signals.pop(s, None)

    await save_prev_signals()

    if confirmed_list:
        await format_and_send_signals(confirmed_list)
    else:
        logger.info("No confirmed signals")

    return confirmed_list


async def main():
    try:
        start_time = time.time()
        logger.info("=" * 50)
        logger.info("SCANNER v3.0 — HTF/ADX/Volume/Calibrated kNN/Feedback/Cooldown")
        await load_tuning()
        await load_prev_signals()
        await init_exchanges()
        if not exchanges:
            raise Exception("No exchanges available")

        # Feedback loop: resolve old outcomes + auto-tune
        await resolve_pending_outcomes()
        await auto_tune_thresholds()

        confirmed = await scan_market_single_run()
        await close_exchanges()
        logger.info(f"Completed in {time.time()-start_time:.1f}s | Signals: {len(confirmed)}")
    except Exception as e:
        logger.error(str(e))
        try:
            await send_error_notification(str(e))
        except:
            pass
        raise


if __name__ == '__main__':
    try:
        asyncio.run(main())
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal: {e}")
        sys.exit(1)
