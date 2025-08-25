# ==============================================================================
# Dhan Intraday EMA Crossover Trading Bot
# ==============================================================================

# This script implements a scalping strategy based on a 9/21 EMA crossover
# with volume confirmation on 5-minute candlestick data for a list of NSE stocks.
# It uses the Dhan sandbox for paper trading, sends real-time alerts to a
# Telegram bot, and logs all trade activities.

# ==============================================================================

import os
import csv
import time
import datetime
from datetime import timedelta
import logging
import pandas as pd
import talib
import pytz
from dhanhq import dhanhq
from telegram import Bot

# ==============================================================================

# Configuration: Update these with your specific details
DHAN_CLIENT_ID = "2508058857"
DHAN_ACCESS_TOKEN = "eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJwYXJ0bmVySWQiOiIiLCJkaGFuQ2xpZW50SWQiOiIyNTA4MDU4ODU3Iiwid2ViaG9va1VybCI6IiIsImlzcyI6ImRoYW4iLCJleHAiOjE3NTY3MDY3NTd9.WkQkfxbIc1KJaRRLART-AYjjAfDjfiUzzoucthWwWrW9pRBWYctdu07wM3gqsBGG3SP_eKO84WKG7MuSKa0V7Q"

# Telegram Bot credentials
# NOTE: Replace 'YOUR_BOT_TOKEN' with your BotFather token.
# NOTE: Replace 'YOUR_CHAT_ID' with your personal chat ID or group chat ID.
TELEGRAM_BOT_TOKEN = "8488367827:AAFc4q3GUiIjBbD6vZEbrAz72ieIxsKRBys"
TELEGRAM_CHAT_ID = "1691456248"

# Stocks to monitor - map stock symbol to security ID
STOCK_IDS_TO_MONITOR = {
    "DCBBANK": "1333",
    "GMRP&UI": "17387",
    "EMIL": "27038",
    "GAEL": "11986",
    "PNB": "21867",
    "JAMNAAUTO": "17822",
    "NFL": "17109",
    "EMBDL": "15372"
}

# Strategy parameters
EMA_SHORT_PERIOD = 9
EMA_LONG_PERIOD = 21
VOLUME_PERIOD = 20
CANDLE_INTERVAL = "5"  # 5-minute candles
DATA_FETCH_INTERVAL = 30  # seconds

# Risk management
TOTAL_CAPITAL = 100000.0  # Total capital
RISK_PER_TRADE_PERCENT = 1.0  # Risk 1% per trade
SLIPPAGE = 0.001  # 0.1% slippage

# Logging setup
LOG_FILE = "trading_bot_log.csv"
LOG_HEADERS = ["timestamp", "stock", "action", "entry_price", "exit_price", "pnl", "quantity", "reason"]

# ==============================================================================

# Initialize clients
try:
    dhan = dhanhq(DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN)
    telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN)
    print("Dhan and Telegram clients initialized successfully.")
except Exception as e:
    print(f"Error initializing API clients: {e}")
    exit()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot_activity.log"),
        logging.StreamHandler()
    ]
)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(LOG_HEADERS)

open_positions = {}
IST = pytz.timezone('Asia/Kolkata')

# ==============================================================================

def send_telegram_message(message):
    try:
        telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        logging.info(f"Telegram message sent: {message}")
    except Exception as e:
        logging.error(f"Failed to send Telegram message: {e}")

def log_trade(stock, action, entry_price=None, exit_price=None, pnl=None, quantity=None, reason=""):
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, stock, action, entry_price, exit_price, pnl, quantity, reason])
    logging.info(f"Trade logged: {action} on {stock}, Reason: {reason}")

def get_historical_data(security_id, interval, count):
    try:
        to_date = datetime.datetime.now(IST)
        # Fetch up to 90 days, but we only need a few to get enough candles.
        from_date = to_date - timedelta(days=4)

        # The intraday endpoint requires the 'YYYY-MM-DD HH:MM:SS' format.
        to_date_str = to_date.strftime('%Y-%m-%d %H:%M:%S')
        from_date_str = from_date.strftime('%Y-%m-%d %H:%M:%S')

        # Final attempt with an educated guess based on docs and error messages.
        response_data = dhan.intraday_historical_data(
            security_id=security_id,
            exchange_segment="NSE_EQ",
            instrument_type="EQUITY",
            interval=interval,
            from_date=from_date_str,
            to_date=to_date_str
        )

        # The intraday API returns a dictionary of lists directly, not nested under 'data' or 'candles'.
        if response_data and isinstance(response_data, dict) and 'open' in response_data:
            df = pd.DataFrame({
                # Timestamp is in epoch, convert to datetime and then to IST.
                'date': pd.to_datetime(response_data['timestamp'], unit='s', utc=True).tz_convert(IST),
                'open': response_data['open'],
                'high': response_data['high'],
                'low': response_data['low'],
                'close': response_data['close'],
                'volume': response_data['volume']
            })
            df.set_index('date', inplace=True)
            df = df.astype(float)
            df.sort_index(inplace=True)
            return df
        else:
            logging.warning(f"No data or unexpected format for {security_id}: {response_data}")
            return None
    except Exception as e:
        logging.error(f"Error fetching historical data for {security_id}: {e}", exc_info=True)
        return None

def calculate_indicators(df):
    if df is None or len(df) < max(EMA_LONG_PERIOD, VOLUME_PERIOD):
        return None, None, None

    close_prices = df['close'].values
    volumes = df['volume'].values

    ema_short = talib.EMA(close_prices, timeperiod=EMA_SHORT_PERIOD)
    ema_long = talib.EMA(close_prices, timeperiod=EMA_LONG_PERIOD)
    avg_volume = talib.SMA(volumes, timeperiod=VOLUME_PERIOD)

    return ema_short, ema_long, avg_volume

def execute_trade(security_id, action, quantity, price):
    order_type = "MARKET"
    exchange_segment = "NSE_EQ"
    product_type = "INTRA"

    try:
        response = dhan.place_order(
            security_id=security_id,
            exchange_segment=exchange_segment,
            transaction_type=action,
            quantity=quantity,
            order_type=order_type,
            product_type=product_type,
            price=price
        )

        if response and response['status'] == 'success':
            logging.info(f"Order placed: {action} {quantity} units of {security_id}.")
            return response
        else:
            logging.error(f"Order failed for {security_id}: {response.get('remarks', 'Unknown error')}")
            return None
    except Exception as e:
        logging.error(f"Error placing order for {security_id}: {e}")
        return None

def main_strategy_loop():
    while True:
        now = datetime.datetime.now(IST)
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

        # Square off all positions after market close
        if now >= market_close:
            logging.info("Market is closing. Squaring off all open positions.")
            for stock, position in list(open_positions.items()):
                security_id = STOCK_IDS_TO_MONITOR[stock]
                df = get_historical_data(security_id, CANDLE_INTERVAL, 2)
                current_close = df['close'].iloc[-1] if df is not None and not df.empty else position['entry_price']

                action = "SELL" if position['direction'] == 'long' else 'BUY'
                pnl = 0
                if position['direction'] == 'long':
                    pnl = (current_close - position['entry_price']) * position['quantity']
                else:  # short
                    pnl = (position['entry_price'] - current_close) * position['quantity']

                execute_trade(security_id, action, position['quantity'], current_close)
                log_action = "EXIT_LONG" if action == "SELL" else "EXIT_SHORT"
                send_telegram_message(f"🤖 EOD Close ({position['direction']})\nStock: {stock}\nPrice: {current_close:.2f}\nP&L: {pnl:.2f}")
                log_trade(stock, log_action, exit_price=current_close, pnl=pnl, quantity=position['quantity'], reason="End of day square off")
                del open_positions[stock]

            # Sleep until next day's market open
            tomorrow = now.date() + datetime.timedelta(days=1)
            next_market_open = now.replace(year=tomorrow.year, month=tomorrow.month, day=tomorrow.day, hour=9, minute=15, second=0, microsecond=0)
            sleep_seconds = (next_market_open - now).total_seconds()
            if sleep_seconds > 0:
                logging.info(f"All positions closed. Sleeping until next market open in {sleep_seconds/3600:.2f} hours.")
                time.sleep(sleep_seconds)
            continue

        if now < market_open:
            logging.info("Market is not open. Sleeping 5 mins.")
            time.sleep(300)
            continue

        logging.info("Polling new candlestick data...")

        for stock, security_id in STOCK_IDS_TO_MONITOR.items():
            try:
                df = get_historical_data(security_id, CANDLE_INTERVAL, 100)
                if df is None or len(df) < max(EMA_LONG_PERIOD, VOLUME_PERIOD) + 2:
                    logging.warning(f"Not enough data for {stock} to calculate indicators.")
                    continue

                ema_short_series, ema_long_series, avg_volume_series = calculate_indicators(df)

                if ema_short_series is None:
                    continue

                current_volume = df['volume'].iloc[-1]
                current_close = df['close'].iloc[-1]

                ema9 = ema_short_series[-1]
                ema9_prev = ema_short_series[-2]
                ema21 = ema_long_series[-1]
                ema21_prev = ema_long_series[-2]
                avg_vol = avg_volume_series[-1]

                if any(pd.isna(x) for x in [ema9, ema9_prev, ema21, ema21_prev, avg_vol]):
                    continue

                # Check open positions for exit signals
                if stock in open_positions:
                    position = open_positions[stock]
                    is_exit = False
                    reason = ""
                    pnl = 0

                    sl_points = position['sl']

                    if position['direction'] == 'long':
                        target = position['entry_price'] + sl_points
                        stop_loss = position['entry_price'] - sl_points

                        if current_close >= target:
                            is_exit, reason = True, "Target profit hit"
                        elif current_close <= stop_loss:
                            is_exit, reason = True, "Stop loss hit"
                        elif ema9 < ema21 and ema9_prev > ema21_prev:
                            is_exit, reason = True, "EMA crossover reversal"

                        if is_exit:
                            pnl = (current_close - position['entry_price']) * position['quantity']
                            execute_trade(security_id, "SELL", position['quantity'], current_close)
                            send_telegram_message(f"🤖 EXIT (Long) 📈\nStock: {stock}\nPrice: {current_close:.2f}\nP&L: {pnl:.2f}\nReason: {reason}")
                            log_trade(stock, "EXIT_LONG", exit_price=current_close, pnl=pnl, quantity=position['quantity'], reason=reason)
                            del open_positions[stock]

                    elif position['direction'] == 'short':
                        target = position['entry_price'] - sl_points
                        stop_loss = position['entry_price'] + sl_points

                        if current_close <= target:
                            is_exit, reason = True, "Target profit hit"
                        elif current_close >= stop_loss:
                            is_exit, reason = True, "Stop loss hit"
                        elif ema9 > ema21 and ema9_prev < ema21_prev:
                            is_exit, reason = True, "EMA crossover reversal"

                        if is_exit:
                            pnl = (position['entry_price'] - current_close) * position['quantity']
                            execute_trade(security_id, "BUY", position['quantity'], current_close)
                            send_telegram_message(f"🤖 EXIT (Short) 📉\nStock: {stock}\nPrice: {current_close:.2f}\nP&L: {pnl:.2f}\nReason: {reason}")
                            log_trade(stock, "EXIT_SHORT", exit_price=current_close, pnl=pnl, quantity=position['quantity'], reason=reason)
                            del open_positions[stock]

                # Check for entry signals if no position is open
                else:
                    risk_amount = TOTAL_CAPITAL * (RISK_PER_TRADE_PERCENT / 100)
                    sl_points = current_close * 0.01  # 1% of current price as SL
                    quantity = int(risk_amount / sl_points) if sl_points > 0 else 1
                    if quantity == 0: quantity = 1

                    # Long entry: bullish crossover + volume confirmation
                    if ema9 > ema21 and ema9_prev < ema21_prev and current_volume > avg_vol:
                        order_response = execute_trade(security_id, "BUY", quantity, current_close)
                        if order_response:
                            open_positions[stock] = {'direction': 'long', 'entry_price': current_close, 'quantity': quantity, 'sl': sl_points}
                            send_telegram_message(f"🤖 NEW (Long) 📈\nStock: {stock}\nPrice: {current_close:.2f}\nQuantity: {quantity}")
                            log_trade(stock, "ENTER_LONG", entry_price=current_close, quantity=quantity, reason="EMA crossover and volume")

                    # Short entry: bearish crossover + volume confirmation
                    elif ema9 < ema21 and ema9_prev > ema21_prev and current_volume > avg_vol:
                        order_response = execute_trade(security_id, "SELL", quantity, current_close)
                        if order_response:
                            open_positions[stock] = {'direction': 'short', 'entry_price': current_close, 'quantity': quantity, 'sl': sl_points}
                            send_telegram_message(f"🤖 NEW (Short) 📉\nStock: {stock}\nPrice: {current_close:.2f}\nQuantity: {quantity}")
                            log_trade(stock, "ENTER_SHORT", entry_price=current_close, quantity=quantity, reason="EMA crossover and volume")

            except Exception as e:
                logging.error(f"Error processing {stock}: {e}", exc_info=True)

        logging.info(f"Sleeping for {DATA_FETCH_INTERVAL} seconds...")
        time.sleep(DATA_FETCH_INTERVAL)

if __name__ == "__main__":
    try:
        send_telegram_message("🤖 Trading Bot Started! Monitoring stocks...")
        main_strategy_loop()
    except KeyboardInterrupt:
        logging.info("Bot stopped by user.")
    except Exception as e:
        logging.critical("A critical error occurred in the main loop.", exc_info=True)
        send_telegram_message(f"🚨 CRITICAL ERROR: Trading bot has stopped unexpectedly. Error: {e}")
