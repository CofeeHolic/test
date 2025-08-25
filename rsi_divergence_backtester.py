# In a Google Colab notebook, create a new cell and run this command first:
# !pip install dhanhq==1.1.2 pandas-ta==0.3.14b

import pandas as pd
import pandas_ta as ta
from dhanhq import dhanhq
from datetime import datetime, timedelta
import time
from typing import Dict, List, Tuple, Optional

# ==============================================================================
# --- 1. CONFIGURATION ---
# ==============================================================================

# --- Dhan API Credentials ---
# IMPORTANT: Replace with your actual Dhan credentials
CLIENT_ID = "YOUR_CLIENT_ID"
ACCESS_TOKEN = "YOUR_ACCESS_TOKEN"

# --- Backtesting Parameters ---
START_DATE = "2024-03-01"  # YYYY-MM-DD
END_DATE = "2024-03-22"    # YYYY-MM-DD
TIME_FRAME = "15"          # Time interval in minutes

# --- Ticker Configuration ---
# IMPORTANT: Replace with the correct Dhan security_id for each stock.
# You can find the security_id from the Dhan API or by using their symbol master list.
# The format is { "TICKER_NAME_FOR_LOGGING": "SECURITY_ID" }
TICKERS = {
    "CANBK": "3518",
    "IRFC": "10022",
    "PNB": "25",
    "SJVN": "30146"
}

# --- Strategy & Capital Parameters ---
INITIAL_CAPITAL = 20000.0
RISK_PER_TRADE_PERCENT = 1.0
RSI_PERIOD = 14
VOLUME_SMA_PERIOD = 20
ADX_PERIOD = 14
DIVERGENCE_LOOKBACK = 30  # Number of candles to look back for divergence
TAKE_PROFIT_RR = 1.5      # Take Profit as a multiple of Risk (Risk/Reward Ratio)


# ==============================================================================
# --- 2. HELPER FUNCTIONS ---
# ==============================================================================

def fetch_historical_data(dhan: dhanhq, security_id: str, from_date: str, to_date: str, interval: str) -> Optional[pd.DataFrame]:
    """Fetches historical OHLCV data for a given security."""
    try:
        # The dhanhq library expects dates in DD-MM-YYYY format for this call
        formatted_from = datetime.strptime(from_date, '%Y-%m-%d').strftime('%d-%m-%Y')
        formatted_to = datetime.strptime(to_date, '%Y-%m-%d').strftime('%d-%m-%Y')

        data = dhan.historical_intraday_data(
            security_id=security_id,
            exchange_segment='NSE_EQ',
            instrument_type='EQUITY',
            from_date=formatted_from,
            to_date=formatted_to
        )
        if data['status'] == 'success' and data.get('data'):
            df = pd.DataFrame(data['data'])
            df['datetime'] = pd.to_datetime(df['start_date'])
            df.set_index('datetime', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            # The API returns data for the whole day, so we might need to filter by time if needed
            # For this backtest, we assume the full day's 15min data is what we want.
            return df.sort_index()
        else:
            print(f"Warning: No data fetched for security_id {security_id}. Response: {data.get('remarks')}")
            return None
    except Exception as e:
        print(f"Error fetching data for security_id {security_id}: {e}")
        return None

def calculate_indicators(df: pd.DataFrame):
    """Calculates and appends required technical indicators to the DataFrame."""
    df.ta.rsi(length=RSI_PERIOD, append=True)
    df.ta.sma(close=df['volume'], length=VOLUME_SMA_PERIOD, append=True, col_names=(f'VOL_SMA_{VOLUME_SMA_PERIOD}',))
    df.ta.adx(length=ADX_PERIOD, append=True)
    df.dropna(inplace=True)

def find_pivots(series: pd.Series, window: int) -> List[int]:
    """Finds pivot points (highs/lows) in a series."""
    pivots = []
    # Simplified pivot detection: a point is a pivot if it's the max/min in a window
    for i in range(window, len(series) - window):
        is_pivot = True
        # Check if it's a local min/max in the window
        if series.name.endswith("_low"): # Detecting pivot lows
            for j in range(i - window, i + window + 1):
                if series.iloc[j] < series.iloc[i]:
                    is_pivot = False
                    break
        else: # Detecting pivot highs
            for j in range(i - window, i + window + 1):
                if series.iloc[j] > series.iloc[i]:
                    is_pivot = False
                    break
        if is_pivot:
            pivots.append(series.index[i])
    return pivots

def detect_divergence(price_history: pd.DataFrame, rsi_series: pd.Series) -> Tuple[Optional[str], Optional[float]]:
    """
    Detects bullish or bearish RSI divergence.
    Returns (signal_type, stop_loss_price).
    """
    if len(price_history) < DIVERGENCE_LOOKBACK:
        return None, None

    recent_price = price_history.tail(DIVERGENCE_LOOKBACK)
    recent_rsi = rsi_series.tail(DIVERGENCE_LOOKBACK)

    # Find last two pivot lows for bullish divergence
    low_pivots = recent_price[recent_price['low'] == recent_price['low'].rolling(5, center=True).min()].index
    if len(low_pivots) >= 2:
        last_low_t, second_last_low_t = low_pivots[-1], low_pivots[-2]
        price_last_low = recent_price.loc[last_low_t, 'low']
        price_second_last_low = recent_price.loc[second_last_low_t, 'low']
        rsi_last_low = recent_rsi.loc[last_low_t]
        rsi_second_last_low = recent_rsi.loc[second_last_low_t]

        if price_last_low < price_second_last_low and rsi_last_low > rsi_second_last_low:
            stop_loss = price_last_low * 0.998 # Place SL slightly below the low
            return "BULLISH", stop_loss

    # Find last two pivot highs for bearish divergence
    high_pivots = recent_price[recent_price['high'] == recent_price['high'].rolling(5, center=True).max()].index
    if len(high_pivots) >= 2:
        last_high_t, second_last_high_t = high_pivots[-1], high_pivots[-2]
        price_last_high = recent_price.loc[last_high_t, 'high']
        price_second_last_high = recent_price.loc[second_last_high_t, 'high']
        rsi_last_high = recent_rsi.loc[last_high_t]
        rsi_second_last_high = recent_rsi.loc[second_last_high_t]

        if price_last_high > second_last_high_t and rsi_last_high < rsi_second_last_high:
            stop_loss = price_last_high * 1.002 # Place SL slightly above the high
            return "BEARISH", stop_loss

    return None, None

# ==============================================================================
# --- 3. BACKTESTING ENGINE ---
# ==============================================================================

def run_backtest():
    """Main backtesting function."""
    if CLIENT_ID == "YOUR_CLIENT_ID" or ACCESS_TOKEN == "YOUR_ACCESS_TOKEN":
        print("!!! IMPORTANT !!!")
        print("Please update CLIENT_ID and ACCESS_TOKEN in the configuration section.")
        return

    print("--- Starting Backtest ---")
    dhan = dhanhq(CLIENT_ID, ACCESS_TOKEN)

    # 1. Fetch and prepare data for all tickers
    print("Fetching and preparing data...")
    data_dict: Dict[str, pd.DataFrame] = {}
    all_timestamps = set()

    for ticker, security_id in TICKERS.items():
        print(f"Fetching data for {ticker} ({security_id})...")
        df = fetch_historical_data(dhan, security_id, START_DATE, END_DATE, TIME_FRAME)
        if df is not None and not df.empty:
            calculate_indicators(df)
            data_dict[ticker] = df
            all_timestamps.update(df.index)
        time.sleep(0.5) # Avoid hitting API rate limits

    if not data_dict:
        print("No data fetched for any ticker. Exiting.")
        return

    sorted_timestamps = sorted(list(all_timestamps))

    # 2. Initialize portfolio
    capital = INITIAL_CAPITAL
    portfolio_value = [INITIAL_CAPITAL]
    open_trades = {}
    trade_log = []

    print("Starting chronological simulation...")
    # 3. Main backtesting loop
    for i in range(DIVERGENCE_LOOKBACK, len(sorted_timestamps)):
        timestamp = sorted_timestamps[i]

        for ticker in data_dict.keys():
            # Skip if ticker doesn't have data for this timestamp
            if timestamp not in data_dict[ticker].index:
                continue

            current_candle = data_dict[ticker].loc[timestamp]

            # --- a. Manage Open Trades ---
            if ticker in open_trades:
                trade = open_trades[ticker]
                pnl = 0
                exit_price = None

                if trade['Direction'] == 'LONG':
                    if current_candle['low'] <= trade['Stop Loss']:
                        exit_price = trade['Stop Loss']
                    elif current_candle['high'] >= trade['Take Profit']:
                        exit_price = trade['Take Profit']
                elif trade['Direction'] == 'SHORT':
                    if current_candle['high'] >= trade['Stop Loss']:
                        exit_price = trade['Stop Loss']
                    elif current_candle['low'] <= trade['Take Profit']:
                        exit_price = trade['Take Profit']

                if exit_price:
                    if trade['Direction'] == 'LONG':
                        pnl = (exit_price - trade['Entry Price']) * trade['Size']
                    else:
                        pnl = (trade['Entry Price'] - exit_price) * trade['Size']

                    capital += pnl
                    portfolio_value.append(capital)

                    trade['Exit Time'] = timestamp
                    trade['Exit Price'] = exit_price
                    trade['PnL'] = pnl
                    trade_log.append(trade)
                    del open_trades[ticker]

            # --- b. Check for New Signals ---
            if ticker not in open_trades:
                # Signal is based on the candle that just closed
                signal_candle_idx = data_dict[ticker].index.get_loc(timestamp) - 1
                if signal_candle_idx < 0:
                    continue

                history_df = data_dict[ticker].iloc[:signal_candle_idx + 1]

                signal, stop_loss = detect_divergence(history_df[['low', 'high']], history_df[f'RSI_{RSI_PERIOD}'])

                if signal:
                    entry_price = current_candle['open']
                    risk_per_share = abs(entry_price - stop_loss)

                    if risk_per_share == 0: continue

                    risk_amount = capital * (RISK_PER_TRADE_PERCENT / 100)
                    position_size = risk_amount / risk_per_share

                    if position_size > 0:
                        direction = 'LONG' if signal == 'BULLISH' else 'SHORT'

                        if direction == 'LONG':
                            take_profit = entry_price + (risk_per_share * TAKE_PROFIT_RR)
                        else: # SHORT
                            take_profit = entry_price - (risk_per_share * TAKE_PROFIT_RR)

                        open_trades[ticker] = {
                            'Ticker': ticker,
                            'Entry Time': timestamp,
                            'Entry Price': entry_price,
                            'Direction': direction,
                            'Stop Loss': stop_loss,
                            'Take Profit': take_profit,
                            'Size': position_size,
                        }

    # 4. Finalize and print results
    print("--- Backtest Complete ---")
    print_performance_summary(trade_log, portfolio_value)


# ==============================================================================
# --- 4. REPORTING & ANALYTICS ---
# ==============================================================================
def calculate_max_drawdown(portfolio_values: List[float]) -> float:
    """Calculates the maximum drawdown from a list of portfolio values."""
    if not portfolio_values:
        return 0.0

    peak = portfolio_values[0]
    max_dd = 0
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_dd:
            max_dd = drawdown
    return max_dd * 100

def print_performance_summary(trade_log: List[Dict], portfolio_history: List[float]):
    """Calculates and prints the performance summary of the backtest."""

    trade_df = pd.DataFrame(trade_log)

    # --- Performance Metrics ---
    ending_capital = portfolio_history[-1] if portfolio_history else INITIAL_CAPITAL
    net_profit = ending_capital - INITIAL_CAPITAL
    net_profit_percent = (net_profit / INITIAL_CAPITAL) * 100

    total_trades = len(trade_df)
    if total_trades > 0:
        winning_trades = trade_df[trade_df['PnL'] > 0]
        losing_trades = trade_df[trade_df['PnL'] <= 0]

        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0

        gross_profit = winning_trades['PnL'].sum()
        gross_loss = abs(losing_trades['PnL'].sum())

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    else:
        win_rate = 0
        profit_factor = 0

    max_drawdown = calculate_max_drawdown(portfolio_history)

    # --- Print Summary ---
    print("\n--- Performance Summary ---")
    print(f"Period:                          {START_DATE} to {END_DATE}")
    print("-" * 30)
    print(f"Starting Capital:                ${INITIAL_CAPITAL:,.2f}")
    print(f"Ending Capital:                  ${ending_capital:,.2f}")
    print(f"Net Profit/Loss:                 ${net_profit:,.2f} ({net_profit_percent:.2f}%)")
    print("-" * 30)
    print(f"Total Trades:                    {total_trades}")
    print(f"Win Rate:                        {win_rate:.2f}%")
    print(f"Profit Factor:                   {profit_factor:.2f}")
    print(f"Maximum Drawdown:                {max_drawdown:.2f}%")
    print("-" * 30)

    # --- Print Trade Log ---
    if not trade_df.empty:
        print("\n--- Trade Log ---")
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.width', 1000)
        # Reorder and format columns for display
        display_cols = [
            'Ticker', 'Direction', 'Entry Time', 'Entry Price',
            'Exit Time', 'Exit Price', 'Stop Loss', 'Take Profit', 'PnL'
        ]
        display_df = trade_df[display_cols].copy()
        for col in ['Entry Price', 'Exit Price', 'Stop Loss', 'Take Profit', 'PnL']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}")

        print(display_df)
    else:
        print("\n--- No trades were executed. ---")


# ==============================================================================
# --- 5. SCRIPT EXECUTION ---
# ==============================================================================

if __name__ == '__main__':
    run_backtest()
