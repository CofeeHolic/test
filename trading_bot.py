import os
import logging
import time
from datetime import datetime, time as dt_time
import pandas as pd
import pandas_ta as ta
import pytz
from dhanhq import dhanhq
import requests
import schedule

# --- CONFIGURATION (Loaded from Environment Variables for EC2) ---
# API Credentials (as environment variables on the server)
DHAN_DATA_CLIENT_ID = os.environ.get('DHAN_DATA_CLIENT_ID')
DHAN_DATA_ACCESS_TOKEN = os.environ.get('DHAN_DATA_ACCESS_TOKEN')
DHAN_SANDBOX_CLIENT_ID = os.environ.get('DHAN_SANDBOX_CLIENT_ID')
DHAN_SANDBOX_ACCESS_TOKEN = os.environ.get('DHAN_SANDBOX_ACCESS_TOKEN')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

# Bot & Strategy Parameters
TICKERS = ["CANBK.NS", "IRFC.NS", "PNB.NS", "SJVN.NS", "GMRINFRA.NS"]
TIME_FRAME = '15'
RISK_PER_TRADE_PERCENT = 0.02

# --- STRATEGY RULES ---
RSI_PERIOD = 14
RSI_BUY_LEVEL = 40  # Signal only valid if RSI is BELOW this level
RSI_SELL_LEVEL = 60 # Signal only valid if RSI is ABOVE this level
ADX_PERIOD = 14
ADX_MIN = 18
ADX_MAX = 35
VOLUME_SMA_PERIOD = 20
VOLUME_MULTIPLIER = 1.2
RISK_REWARD_RATIO = 2.0

# --- LOGGING SETUP ---
log_file = 'trading_bot.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler() # Also print logs to console
    ]
)
logger = logging.getLogger(__name__)


# --- TELEGRAM BOT ---
def send_telegram_message(message):
    """Sends a message to the configured Telegram chat."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials are not set. Skipping message.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            logger.error(f"Failed to send Telegram message: {response.text}")
    except Exception as e:
        logger.error(f"Exception while sending Telegram message: {e}")

# --- DHAN API CLIENTS ---
dhan_data = None
dhan_trade = None
try:
    logger.info("Initializing Dhan API clients...")
    if not all([DHAN_DATA_CLIENT_ID, DHAN_DATA_ACCESS_TOKEN, DHAN_SANDBOX_CLIENT_ID, DHAN_SANDBOX_ACCESS_TOKEN]):
        raise ValueError("One or more Dhan API credentials are not set in environment variables.")

    # Live Data Client (Main API) - Fetches market data
    dhan_data = dhanhq(DHAN_DATA_CLIENT_ID, DHAN_DATA_ACCESS_TOKEN)

    # Sandbox Trading Client - Executes trades in the sandbox
    dhan_trade = dhanhq(DHAN_SANDBOX_CLIENT_ID, DHAN_SANDBOX_ACCESS_TOKEN, api_type='trading', is_sandbox=True)

    logger.info("Dhan API clients initialized successfully.")
    # Let's not send a telegram message here, startup message is enough.

except ValueError as e:
    logger.critical(f"API Initialization Error: {e}")
    send_telegram_message(f"CRITICAL: {e}")
    exit() # Exit if clients can't be initialized
except Exception as e:
    logger.critical(f"An unexpected error occurred during API initialization: {e}")
    send_telegram_message(f"CRITICAL: An unexpected error occurred during API initialization: {e}")
    exit()

# --- MARKET HOURS ---
def is_market_open():
    """Checks if the Indian stock market is open (9:15 AM to 3:30 PM IST)."""
    tz = pytz.timezone('Asia/Kolkata')
    now = datetime.now(tz)

    if now.weekday() > 4: # Monday=0, Sunday=6
        return False

    market_open_time = dt_time(9, 15)
    market_close_time = dt_time(15, 30)

    return market_open_time <= now.time() <= market_close_time

# --- DYNAMIC DATA ---
SECURITY_ID_MAPPING = {}

def build_security_id_mapping():
    """Fetches all scrips from Dhan and builds a mapping from symbol to security_id."""
    global SECURITY_ID_MAPPING
    try:
        logger.info("Building security ID mapping...")
        all_scrips = dhan_data.get_all_scrip_list()
        if isinstance(all_scrips, list):
             df = pd.DataFrame(all_scrips)
        else:
             # Assuming it might be a dict with a 'data' key
             df = pd.DataFrame(all_scrips.get('data', []))

        if df.empty:
            logger.error("Failed to fetch scrip list or list is empty.")
            return

        # Filter for equity segment on NSE
        df_nse_eq = df[(df['SEM_EXM_EXCH_ID'] == 'NSE') & (df['SEM_INSTRUMENT'] == 'EQUITY')]
        SECURITY_ID_MAPPING = pd.Series(df_nse_eq.SEM_SMST_SECURITY_ID.values, index=df_nse_eq.SEM_TRADING_SYMBOL).to_dict()
        logger.info(f"Security ID mapping built. Found {len(SECURITY_ID_MAPPING)} NSE Equity symbols.")
        if not all(ticker.split('.')[0] in SECURITY_ID_MAPPING for ticker in TICKERS):
             logger.warning("Some tickers not found in the security map. Data fetching might fail for them.")

    except Exception as e:
        logger.critical(f"CRITICAL: Failed to build security ID mapping: {e}")
        send_telegram_message(f"CRITICAL: Failed to build security ID mapping: {e}")
        exit()

# --- STRATEGY IMPLEMENTATION ---
def fetch_and_resample_data(ticker):
    """Fetches 1-minute intraday data and resamples it."""
    symbol = ticker.split('.')[0]
    security_id = SECURITY_ID_MAPPING.get(symbol)

    if not security_id:
        logger.error(f"Security ID not found for {symbol}. Skipping.")
        return None

    try:
        from datetime import date, timedelta
        to_date = date.today().strftime("%Y-%m-%d")
        from_date = (date.today() - timedelta(days=20)).strftime("%Y-%m-%d") # More data for indicator stability

        logger.info(f"Fetching data for {ticker} (ID: {security_id}) from {from_date} to {to_date}")
        hist_data = dhan_data.historical_intraday_data(
            security_id=str(security_id),
            exchange_segment='NSE_EQ',
            instrument_type='EQUITY',
            from_date=from_date,
            to_date=to_date
        )

        if hist_data.get('status') != 'success':
            logger.error(f"API Error for {ticker}: {hist_data.get('remarks')}")
            return None

        df = pd.DataFrame(hist_data['data'])
        df['datetime'] = pd.to_datetime(df['start_Time'], unit='s')
        df.set_index('datetime', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])

        resampled_df = df.resample(f'{TIME_FRAME}T').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()

        return resampled_df

    except Exception as e:
        logger.error(f"Exception in fetch_and_resample_data for {ticker}: {e}")
        return None


def calculate_indicators(df):
    """Calculates technical indicators."""
    if df is None or df.empty:
        return None
    try:
        df.ta.rsi(length=RSI_PERIOD, append=True, col_names=(f'RSI_{RSI_PERIOD}',))
        df.ta.adx(length=ADX_PERIOD, append=True, col_names=(f'ADX_{ADX_PERIOD}', f'DMP_{ADX_PERIOD}', f'DMN_{ADX_PERIOD}'))
        df.ta.sma(close='volume', length=VOLUME_SMA_PERIOD, append=True, col_names=(f'VOLUME_SMA_{VOLUME_SMA_PERIOD}',))
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logger.error(f"Exception during indicator calculation: {e}")
        return None

def detect_rsi_divergence(df, lookback=30):
    """Detects Bullish/Bearish RSI divergence on the last candle."""
    if len(df) < lookback:
        return None

    recent_df = df.iloc[-lookback:]
    last_candle = df.iloc[-1]

    # Bullish Divergence: Price makes a lower low, RSI makes a higher low.
    price_min_val = recent_df['low'].min()
    if last_candle['low'] < price_min_val:
        # Find the point of the previous lowest low
        prev_low_df = recent_df[recent_df['low'] == price_min_val]
        if not prev_low_df.empty:
            prev_rsi = prev_low_df[f'RSI_{RSI_PERIOD}'].iloc[0]
            if last_candle[f'RSI_{RSI_PERIOD}'] > prev_rsi:
                return 'BULLISH'

    # Bearish Divergence: Price makes a higher high, RSI makes a lower high.
    price_max_val = recent_df['high'].max()
    if last_candle['high'] > price_max_val:
        # Find the point of the previous highest high
        prev_high_df = recent_df[recent_df['high'] == price_max_val]
        if not prev_high_df.empty:
            prev_rsi = prev_high_df[f'RSI_{RSI_PERIOD}'].iloc[0]
            if last_candle[f'RSI_{RSI_PERIOD}'] < prev_rsi:
                return 'BEARISH'

    return None

def check_signal(ticker, df):
    """Checks if a valid trade signal exists on the last candle."""
    if df is None or len(df) < 2:
        return None, None

    last_candle = df.iloc[-1]
    divergence = detect_rsi_divergence(df)

    signal = None
    rsi_col = f'RSI_{RSI_PERIOD}'
    adx_col = f'ADX_{ADX_PERIOD}'
    vol_sma_col = f'VOLUME_SMA_{VOLUME_SMA_PERIOD}'

    if divergence == 'BULLISH':
        logger.info(f"[{ticker}] Bullish divergence detected. RSI: {last_candle[rsi_col]:.2f}, ADX: {last_candle[adx_col]:.2f}")
        if (last_candle[rsi_col] < RSI_BUY_LEVEL and
            last_candle['volume'] > last_candle[vol_sma_col] * VOLUME_MULTIPLIER and
            ADX_MIN <= last_candle[adx_col] <= ADX_MAX):
            signal = 'BUY'

    elif divergence == 'BEARISH':
        logger.info(f"[{ticker}] Bearish divergence detected. RSI: {last_candle[rsi_col]:.2f}, ADX: {last_candle[adx_col]:.2f}")
        if (last_candle[rsi_col] > RSI_SELL_LEVEL and
            last_candle['volume'] > last_candle[vol_sma_col] * VOLUME_MULTIPLIER and
            ADX_MIN <= last_candle[adx_col] <= ADX_MAX):
            signal = 'SELL'

    if signal:
        return signal, last_candle

    return None, None

# --- STATE MANAGEMENT ---
ACTIVE_TRADES = {} # { 'ticker': 'order_id' }

# --- TRADE EXECUTION ---
def calculate_position_details(ticker, signal, candle):
    """Calculates position size, stop loss, and target price."""
    try:
        fund_limits = dhan_trade.get_fund_limits()
        if fund_limits['status'] != 'success':
            logger.error("Failed to fetch fund limits.")
            return None

        account_balance = fund_limits['data']['availabelBalance']
        risk_amount = account_balance * RISK_PER_TRADE_PERCENT

        entry_price = candle['close']

        if signal == 'BUY':
            stop_loss_price = candle['low']
            risk_per_share = entry_price - stop_loss_price
        else: # SELL
            stop_loss_price = candle['high']
            risk_per_share = stop_loss_price - entry_price

        if risk_per_share <= 0:
            logger.warning(f"[{ticker}] Risk per share is zero or negative. Cannot calculate quantity.")
            return None

        quantity = int(risk_amount / risk_per_share)
        if quantity == 0:
            logger.warning(f"[{ticker}] Calculated quantity is 0. Risk amount might be too low.")
            return None

        # Dhan bracket orders use price offsets (delta)
        stop_loss_delta = abs(entry_price - stop_loss_price)
        target_delta = stop_loss_delta * RISK_REWARD_RATIO

        return {
            "quantity": quantity,
            "stop_loss_delta": round(stop_loss_delta, 2),
            "target_delta": round(target_delta, 2)
        }

    except Exception as e:
        logger.error(f"[{ticker}] Exception in calculate_position_details: {e}")
        return None

def execute_bracket_order(ticker, signal, candle):
    """Places a bracket order in the sandbox."""
    if ticker in ACTIVE_TRADES:
        logger.info(f"[{ticker}] Active trade already exists. Skipping new signal.")
        return

    logger.info(f"[{ticker}] Preparing to execute {signal} order.")
    details = calculate_position_details(ticker, signal, candle)
    if not details:
        logger.error(f"[{ticker}] Could not calculate position details. Aborting trade.")
        return

    symbol = ticker.split('.')[0]
    security_id = SECURITY_ID_MAPPING.get(symbol)

    try:
        order_response = dhan_trade.place_order(
            security_id=str(security_id),
            exchange_segment='NSE_EQ',
            transaction_type=dhan_trade.BUY if signal == 'BUY' else dhan_trade.SELL,
            quantity=details['quantity'],
            order_type=dhan_trade.BRACKET,
            product_type=dhan_trade.INTRA,
            price=0, # Market Order
            bo_profit_value=details['target_delta'],
            bo_stop_loss_value=details['stop_loss_delta']
        )

        if order_response and order_response.get('status') == 'success':
            order_id = order_response['data']['orderId']
            ACTIVE_TRADES[ticker] = order_id
            logger.info(f"[{ticker}] Bracket order placed successfully. Order ID: {order_id}")

            message = (
                f"✅ *New Trade Executed ({signal})*\n\n"
                f"*Ticker:* `{ticker}`\n"
                f"*Quantity:* `{details['quantity']}`\n"
                f"*Entry Price:* ~`{candle['close']:.2f}` (Market)\n"
                f"*Stop Loss Δ:* `{details['stop_loss_delta']}`\n"
                f"*Target Δ:* `{details['target_delta']}`\n"
                f"*Order ID:* `{order_id}`"
            )
            send_telegram_message(message)
        else:
            error_msg = order_response.get('remarks', 'Unknown error')
            logger.error(f"[{ticker}] Failed to place order: {error_msg}")
            send_telegram_message(f"❌ *Trade Failed for {ticker}* ❌\nReason: {error_msg}")

    except Exception as e:
        logger.error(f"[{ticker}] Exception during order placement: {e}")
        send_telegram_message(f"❌ *Trade Exception for {ticker}* ❌\n`{e}`")


# --- POSITION MONITORING ---
def monitor_and_log_closed_trades():
    """Checks the order book for closed trades and logs them."""
    if not ACTIVE_TRADES:
        return

    # Always monitor active trades, but log warnings if it's outside market hours
    if not is_market_open():
        logger.warning(f"Monitoring trades outside market hours. Active trades: {list(ACTIVE_TRADES.keys())}")

    logger.info("--- Monitoring active positions ---")
    try:
        order_book = dhan_trade.get_order_book()
        if order_book['status'] != 'success' or not order_book.get('data'):
            logger.warning("Could not fetch order book or it is empty.")
            return

        closed_tickers = []
        for ticker, order_id in list(ACTIVE_TRADES.items()):
            found_order = False
            for order in order_book['data']:
                if order['orderId'] == order_id:
                    found_order = True
                    if order['orderStatus'] == 'EXECUTED':
                        logger.info(f"[{ticker}] Position closed (Order ID: {order_id}).")
                        message = (
                            f"🎉 *Position Closed for {ticker}*\n\n"
                            f"The Bracket Order `{order_id}` has been executed (SL or TP hit)."
                        )
                        send_telegram_message(message)
                        closed_tickers.append(ticker)
                    elif order['orderStatus'] in ['CANCELED', 'REJECTED']:
                        logger.info(f"[{ticker}] Order {order_id} is {order['orderStatus']}.")
                        message = f"ℹ️ *Order Update for {ticker}*\n\nOrder `{order_id}` is now `{order['orderStatus']}`."
                        send_telegram_message(message)
                        closed_tickers.append(ticker)
                    break

            if not found_order:
                logger.warning(f"[{ticker}] Active order {order_id} not found in order book. Assuming closed/stale.")
                closed_tickers.append(ticker)

        for ticker in closed_tickers:
            if ticker in ACTIVE_TRADES:
                del ACTIVE_TRADES[ticker]

    except Exception as e:
        logger.error(f"Exception in monitor_and_log_closed_trades: {e}")


# --- MAIN LOOP ---
def main_job():
    """The main job to be scheduled."""
    if not is_market_open():
        logger.info("Market is closed. Skipping strategy check.")
        return

    logger.info("===== Running Strategy Check =====")
    for ticker in TICKERS:
        if ticker in ACTIVE_TRADES:
            logger.info(f"[{ticker}] Skipping check, active trade present.")
            continue

        logger.info(f"--- Checking {ticker} ---")
        data = fetch_and_resample_data(ticker)
        if data is None or data.empty:
            logger.warning(f"[{ticker}] No data received, skipping.")
            continue

        data_with_indicators = calculate_indicators(data)
        if data_with_indicators is None or data_with_indicators.empty:
            logger.warning(f"[{ticker}] Not enough data for indicators, skipping.")
            continue

        signal, candle = check_signal(ticker, data_with_indicators)

        if signal:
            logger.info(f"💥💥💥 NEW SIGNAL: {signal} for {ticker} 💥💥💥")
            execute_bracket_order(ticker, signal, candle)
        else:
            logger.info(f"[{ticker}] No signal found.")
    logger.info("===== Strategy Check Complete =====")


def main():
    """Main function to run the trading bot."""
    logger.info("Starting trading bot...")
    send_telegram_message("🤖 **Trading Bot Started** 🤖\nInitializing...")

    build_security_id_mapping()

    send_telegram_message("✅ Bot is now running and monitoring tickers.")

    # Schedule jobs
    schedule.every(int(TIME_FRAME)).minutes.do(main_job)
    schedule.every(60).seconds.do(monitor_and_log_closed_trades)

    # Run the main job once at the start to avoid waiting for the first interval
    main_job()

    # Main loop
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
