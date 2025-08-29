import os
import pandas as pd
import pandas_ta as ta
from dhanhq import dhanhq
from datetime import datetime, date, timedelta
from io import StringIO
import base64
import json
import requests


def fetch_securities_in_memory():
    """
    Fetches the Dhan security list CSV and loads it into a pandas DataFrame
    without saving it to disk to avoid filesystem errors.
    """
    url = 'https://images.dhan.co/api-data/api-scrip-master.csv'
    try:
        response = requests.get(url)
        response.raise_for_status()
        csv_data = StringIO(response.text)
        # Suppress DtypeWarning as we don't use the mixed-type columns
        df = pd.read_csv(csv_data, low_memory=False)
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching security list from URL: {e}")
        return None


def get_historical_data(dhan, security_id, exchange_segment, from_date, to_date):
    """
    Fetches 15-minute historical data for a given security ID.
    Note: The dhanhq library's intraday_minute_data has a limitation of fetching
    data for the last 5-10 trading days only.
    """
    try:
        response = dhan.intraday_minute_data(
            security_id=str(security_id),
            exchange_segment=exchange_segment,
            instrument_type='EQUITY',
            from_date=from_date,
            to_date=to_date,
            interval=15
        )

        if response.get('status') == 'success' and 'data' in response and response['data']:
            data = pd.DataFrame(response['data'])
            data['datetime'] = pd.to_datetime(data['start_Time'], unit='s')
            # Adjust to IST timezone
            data['datetime'] = data['datetime'] + pd.Timedelta(hours=5, minutes=30)
            data.set_index('datetime', inplace=True)
            data.rename(columns={'close': 'price'}, inplace=True)
            for col in ['open', 'high', 'low', 'price', 'volume']:
                data[col] = pd.to_numeric(data[col])
            return data
        else:
            remarks = response.get('remarks', 'No data')
            if remarks and 'error_message' in remarks:
                print(f"Warning: Could not fetch data for security ID {security_id}. Reason: {remarks['error_message']}")
            else:
                print(f"Warning: Could not fetch data for security ID {security_id}. Response: {remarks}")
            return pd.DataFrame()

    except Exception as e:
        print(f"An error occurred while fetching data for security ID {security_id}: {e}")
        return pd.DataFrame()


def calculate_indicators(df):
    """Calculates all required technical indicators for the strategy."""
    df.ta.rsi(length=14, append=True, col_names=('RSI_14',))
    df.ta.sma(close='volume', length=20, append=True, col_names=('VOLUME_SMA_20',))
    # ADX calculation also provides DMP (+DI) and DMN (-DI)
    df.ta.adx(length=14, append=True, col_names=('ADX_14', 'DMP_14', 'DMN_14'))
    df.dropna(inplace=True)
    return df


def find_swing_points(series, window=5):
    """
    Finds swing high and low points in a time series using a rolling window.
    A point is a swing high if it's the maximum in a window of (2*window + 1) periods.
    A point is a swing low if it's the minimum in that same window.
    """
    rolling_max = series.rolling(window * 2 + 1, center=True).max()
    rolling_min = series.rolling(window * 2 + 1, center=True).min()

    highs = series[series == rolling_max].index
    lows = series[series == rolling_min].index

    return highs, lows


def run_backtest(df, ticker, capital):
    """
    Runs the backtesting simulation for a single ticker's data.
    Iterates through each candle, checks for entry/exit conditions,
    and simulates trades.
    """
    trades = []
    position = None # Can be 'LONG', 'SHORT', or None
    entry_price = 0
    shares = 0
    entry_idx = 0
    stop_loss = 0
    take_profit = 0

    # Pre-calculate all swing points for efficiency
    price_highs_idx, price_lows_idx = find_swing_points(df['high'])

    for i in range(1, len(df)):
        current_price = df['price'].iloc[i]

        # --- EXIT LOGIC ---
        # Check if a stop-loss or take-profit has been hit for the open position.
        if position == 'LONG':
            if current_price <= stop_loss:
                pnl = (stop_loss - entry_price) * shares
                capital += entry_price * shares + pnl
                trades.append((ticker, df.index[entry_idx], df.index[i], pnl))
                position = None
            elif current_price >= take_profit:
                pnl = (take_profit - entry_price) * shares
                capital += entry_price * shares + pnl
                trades.append((ticker, df.index[entry_idx], df.index[i], pnl))
                position = None
            continue

        if position == 'SHORT':
            if current_price >= stop_loss:
                pnl = (entry_price - stop_loss) * shares
                capital += entry_price * shares + pnl
                trades.append((ticker, df.index[entry_idx], df.index[i], pnl))
                position = None
            elif current_price <= take_profit:
                pnl = (entry_price - take_profit) * shares
                capital += entry_price * shares + pnl
                trades.append((ticker, df.index[entry_idx], df.index[i], pnl))
                position = None
            continue

        # --- ENTRY LOGIC ---
        # If no position is open, check for new trade signals.
        if position is None:
            # Common conditions for both long and short entry
            is_volume_high = df['volume'].iloc[i] > df['VOLUME_SMA_20'].iloc[i]
            is_adx_strong = df['ADX_14'].iloc[i] > 20

            # 1. Bullish RSI Divergence (Long Entry)
            if is_volume_high and is_adx_strong and df['DMP_14'].iloc[i] > df['DMN_14'].iloc[i]:
                recent_price_lows = price_lows_idx[price_lows_idx < df.index[i]]
                if len(recent_price_lows) >= 2:
                    last_low_idx, prev_low_idx = recent_price_lows[-1], recent_price_lows[-2]

                    # Condition 1: Price makes a lower low
                    if df.loc[last_low_idx, 'low'] < df.loc[prev_low_idx, 'low']:
                        # Condition 2: RSI makes a higher low
                        if df.loc[last_low_idx, 'RSI_14'] > df.loc[prev_low_idx, 'RSI_14']:
                            # All conditions met, open a LONG position
                            position = 'LONG'
                            entry_price = current_price
                            entry_idx = i
                            shares = capital / entry_price
                            stop_loss = df.loc[last_low_idx, 'low'] # SL at the swing low
                            risk = entry_price - stop_loss
                            take_profit = entry_price + (risk * 2) # 2:1 Risk-Reward Ratio
                            capital = 0 # Allocate full capital

            # 2. Bearish RSI Divergence (Short Entry)
            if is_volume_high and is_adx_strong and df['DMN_14'].iloc[i] > df['DMP_14'].iloc[i]:
                recent_price_highs = price_highs_idx[price_highs_idx < df.index[i]]
                if len(recent_price_highs) >= 2:
                    last_high_idx, prev_high_idx = recent_price_highs[-1], recent_price_highs[-2]

                    # Condition 1: Price makes a higher high
                    if df.loc[last_high_idx, 'high'] > df.loc[prev_high_idx, 'high']:
                        # Condition 2: RSI makes a lower high
                        if df.loc[last_high_idx, 'RSI_14'] < df.loc[prev_high_idx, 'RSI_14']:
                            # All conditions met, open a SHORT position
                            position = 'SHORT'
                            entry_price = current_price
                            entry_idx = i
                            shares = capital / entry_price
                            stop_loss = df.loc[last_high_idx, 'high'] # SL at the swing high
                            risk = stop_loss - entry_price
                            take_profit = entry_price - (risk * 2) # 2:1 Risk-Reward Ratio
                            capital = 0 # Allocate full capital

    # At the end of the data, if a position is still open, close it at the last known price.
    if position == 'LONG':
        pnl = (df['price'].iloc[-1] - entry_price) * shares
        capital += entry_price * shares + pnl
        trades.append((ticker, df.index[entry_idx], df.index[-1], pnl))
    elif position == 'SHORT':
        pnl = (entry_price - df['price'].iloc[-1]) * shares
        capital += entry_price * shares + pnl
        trades.append((ticker, df.index[entry_idx], df.index[-1], pnl))

    return capital, trades


def main():
    """
    Main function to orchestrate the entire backtesting process.
    """
    # --- 1. Parameters & Setup ---
    access_token = os.getenv("DHAN_ACCESS_TOKEN")
    if not access_token:
        print("Error: DHAN_ACCESS_TOKEN environment variable not set.")
        print("Please set it using: export DHAN_ACCESS_TOKEN='your_token_here'")
        return

    try:
        # Extract client ID from the JWT access token payload
        payload = access_token.split('.')[1]
        payload += '=' * (-len(payload) % 4)
        decoded_payload = base64.urlsafe_b64decode(payload)
        client_id = json.loads(decoded_payload)['dhanClientId']
        dhan = dhanhq(client_id, access_token)
    except Exception as e:
        print(f"Error initializing DhanHQ client. Please check your access token. Error: {e}")
        return

    tickers = ["CANBK", "IRFC", "PNB", "SJVN", "GMRAPORTS", "ASHOKLEY", "ABFRL", "NTPC"]

    # NOTE: The DhanHQ API only provides intraday data for the last 5-10 trading days.
    # We are using a recent 5-day period to ensure the script can run for demonstration.
    end_date = date.today().strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=5)).strftime("%Y-%m-%d")

    initial_capital = 10000.0
    total_capital = initial_capital * len(tickers)
    final_capital = 0
    all_trades = []

    # --- 2. Print Header ---
    print("--- RSI Divergence Reversal Strategy Backtest ---")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Date Range: {start_date} to {end_date} (Demonstration Run)")
    print("NOTE: Backtest is run on a recent 5-day period due to API limitations.")
    print(f"Timeframe: 15-minute Candles")
    print(f"Initial Capital per Ticker: ₹{initial_capital:,.2f}")
    print(f"Total Starting Capital: ₹{total_capital:,.2f}")
    print("-" * 50)

    # --- 3. Fetch Security IDs ---
    print("Fetching security list to map tickers to security IDs...")
    securities_df = fetch_securities_in_memory()
    if securities_df is None:
        print("Fatal: Could not fetch security list. Exiting.")
        return

    # Filter for the correct Equity segment ('E')
    nse_equities_df = securities_df[securities_df['SEM_SEGMENT'] == 'E']
    ticker_to_id = {}
    for ticker in tickers:
        security_row = nse_equities_df[nse_equities_df['SEM_TRADING_SYMBOL'] == ticker]
        if not security_row.empty:
            ticker_to_id[ticker] = security_row.iloc[0]['SEM_SMST_SECURITY_ID']
        else:
            print(f"Warning: Security ID not found for {ticker} in the Equity segment. It will be skipped.")

    # --- 4. Run Backtest for each Ticker ---
    for ticker in tickers:
        if ticker not in ticker_to_id:
            final_capital += initial_capital # Add back capital for skipped tickers
            continue

        print(f"Processing {ticker}...")
        security_id = ticker_to_id[ticker]
        data = get_historical_data(dhan, security_id, dhan.NSE, start_date, end_date)

        if data.empty or len(data) < 30: # Need enough data for indicators
            print(f"Not enough data for {ticker} to run backtest.")
            final_capital += initial_capital
            continue

        data_with_indicators = calculate_indicators(data)

        capital_for_ticker, trades_for_ticker = run_backtest(data_with_indicators, ticker, initial_capital)
        final_capital += capital_for_ticker
        all_trades.extend(trades_for_ticker)

    # --- 5. Print Results ---
    print("\n--- Backtest Results ---")
    total_pnl = final_capital - total_capital
    num_trades = len(all_trades)
    wins = sum(1 for trade in all_trades if trade[3] > 0)
    win_rate = (wins / num_trades * 100) if num_trades > 0 else 0

    print(f"Total Starting Capital: ₹{total_capital:,.2f}")
    print(f"Final Capital:          ₹{final_capital:,.2f}")
    print(f"Total P&L:              ₹{total_pnl:,.2f}")
    print(f"Total Trades Executed:  {num_trades}")
    print(f"Win Rate:               {win_rate:.2f}%")

    print("\n--- Trade Log ---")
    if not all_trades:
        print("No trades were executed.")
    else:
        trade_log_df = pd.DataFrame(all_trades, columns=['Ticker', 'Entry Time', 'Exit Time', 'P&L'])
        trade_log_df['P&L'] = trade_log_df['P&L'].map('₹{:,.2f}'.format)
        print(trade_log_df.to_string())

    print("\n--- End of Report ---")


if __name__ == "__main__":
    main()
