# RSI Divergence Backtesting Script (Corrected for older dhanhq library)
#
# How to Run:
# 1. In a separate cell, run: !pip install dhanhq pandas numpy requests talib-binary
# 2. Paste all three parts of this script into a single, new cell.
# 3. Enter your credentials where indicated below.
# 4. Run the cell with the script to see the results.

# ==============================================================================
# PART 1: Configuration & Data
# ==============================================================================
import pandas as pd
from dhanhq import dhanhq
import requests
import json
import numpy as np
import talib

# --- Backtest Parameters (EDIT THESE) ---
START_DATE = "2025-08-01"
END_DATE = "2025-08-29"
TIME_FRAME = '15' # Timeframe in minutes: '1', '5', '15', '30', '60'
TICKERS = [
    "CANBK.NS", "IRFC.NS", "PNB.NS", "SJVN.NS",
    "GMRINFRA.NS", "ASHOKLEY.NS", "ABFRL.NS"
]
INITIAL_CAPITAL = 10000.0
RISK_PER_TRADE_PERCENT = 0.02
RSI_PERIOD = 14
VOLUME_MULTIPLIER = 1.2
ADX_THRESHOLD = 18
RISK_REWARD_RATIO = 2.0
USE_RELAXED_LOGIC = True
# --- End of Configuration ---

# --- !! IMPORTANT !! ---
# Enter your Dhan credentials below.
client_id = "YOUR_CLIENT_ID_HERE"
access_token = "YOUR_ACCESS_TOKEN_HERE"
# --------------------

try:
    if client_id == "YOUR_CLIENT_ID_HERE" or access_token == "YOUR_ACCESS_TOKEN_HERE":
        raise ValueError("Please enter your actual Dhan Client ID and Access Token.")

    # CORRECTED INITIALIZATION: Initialize with client_id only for older library version
    dhan = dhanhq(client_id)
    print("Attempting to connect to Dhan API...")

    # CORRECTED API CALL: Pass access_token as a keyword argument
    funds = dhan.get_fund_limits(access_token=access_token)

    if funds.get('status') == 'success':
        print("Successfully connected to Dhan API.")
    else:
        raise ConnectionError(f"Failed to connect: {funds.get('remarks', 'No remarks')}")
except Exception as e:
    print(f"Error during connection: {e}")
    raise

def fetch_intraday_with_timeframe(symbol, from_date, to_date, timeframe, access_token):
    url = "https://api.dhan.co/historical-intraday-data"
    headers = {"access-token": access_token, "Content-Type": "application/json", "Accept": "application/json"}
    payload = {"symbol": symbol, "exchange": "NSE", "instrument": "EQUITY", "from": from_date, "to": to_date, "interval": timeframe}
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        if data.get('status') == 'success' and 'data' in data:
            return pd.DataFrame(data['data'])
        else:
            print(f"API Error for {symbol}: {data.get('remarks', 'Unknown error')}")
            return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        print(f"HTTP Error for {symbol}: {e}")
        return pd.DataFrame()

# CORRECTED API CALL: Pass access_token as a keyword argument
all_scrips_response = dhan.get_all_scrips(access_token=access_token)
if isinstance(all_scrips_response, dict) and 'data' in all_scrips_response:
    all_scrips = all_scrips_response['data']
    symbol_to_id_map = {scrip['SEM_TRADING_SYMBOL']: scrip['SEM_SECURITY_ID'] for scrip in all_scrips.values() if scrip.get('SEM_EXCH_ID') == 'NSE_EQ'}
else:
    symbol_to_id_map = {}

all_historical_data = []
tickers_to_fetch = [t.replace('.NS', '') for t in TICKERS]
print(f"\nStarting data fetch for {len(tickers_to_fetch)} tickers...")
for ticker in tickers_to_fetch:
    if ticker not in symbol_to_id_map:
        print(f"Warning: Ticker {ticker} not found. Skipping.")
        continue
    df = fetch_intraday_with_timeframe(symbol=ticker, from_date=START_DATE, to_date=END_DATE, timeframe=TIME_FRAME, access_token=access_token)
    if not df.empty:
        df['ticker'] = ticker
        all_historical_data.append(df)

if all_historical_data:
    combined_data = pd.concat(all_historical_data, ignore_index=True)
    combined_data['datetime'] = pd.to_datetime(combined_data['start_Time'], unit='s')
    combined_data.set_index('datetime', inplace=True)
    combined_data.sort_index(inplace=True)
    combined_data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    combined_data.drop(columns=['start_Time'], inplace=True)
    print("\n--- Data Fetching and Processing Complete ---")
else:
    print("\n--- No historical data fetched. ---")
    combined_data = pd.DataFrame()

# ==============================================================================
# PART 2: Backtesting Engine
# ==============================================================================
if 'combined_data' in locals() and not combined_data.empty:
    print("\nCalculating indicators for all tickers...")

    data_with_indicators = combined_data.copy()
    data_with_indicators.sort_index(inplace=True)

    indicator_groups = []
    for ticker, group in data_with_indicators.groupby('ticker'):
        group = group.copy()
        group['rsi'] = talib.RSI(group['Close'], timeperiod=RSI_PERIOD)
        group['adx'] = talib.ADX(group['High'], group['Low'], group['Close'], timeperiod=ADX_THRESHOLD)
        group['vol_avg'] = group['Volume'].rolling(RSI_PERIOD).mean()
        indicator_groups.append(group)

    if indicator_groups:
        indicator_data = pd.concat(indicator_groups).sort_index()
        indicator_data.dropna(inplace=True)
        print("Indicators calculated successfully.")
    else:
        indicator_data = pd.DataFrame()
        print("No data to calculate indicators on.")

    if not indicator_data.empty:
        print("\n--- Starting Backtest Simulation (Long & Short) ---")

        capital = INITIAL_CAPITAL
        trade_log = []
        open_positions = {}
        divergence_lookback = 30

        for i in range(divergence_lookback, len(indicator_data)):
            current_candle = indicator_data.iloc[i]
            ticker = current_candle['ticker']

            # --- A. Check for Exits ---
            if ticker in open_positions:
                trade = open_positions[ticker]
                exit_reason = None
                if trade['Direction'] == 'LONG':
                    if current_candle['Low'] <= trade['Stop_Loss']:
                        exit_price, exit_reason = trade['Stop_Loss'], 'Stop-Loss'
                    elif current_candle['High'] >= trade['Take_Profit']:
                        exit_price, exit_reason = trade['Take_Profit'], 'Take-Profit'
                elif trade['Direction'] == 'SHORT':
                    if current_candle['High'] >= trade['Stop_Loss']:
                        exit_price, exit_reason = trade['Stop_Loss'], 'Stop-Loss'
                    elif current_candle['Low'] <= trade['Take_Profit']:
                        exit_price, exit_reason = trade['Take_Profit'], 'Take-Profit'

                if exit_reason:
                    pnl = (exit_price - trade['Entry_Price']) * trade['Quantity'] if trade['Direction'] == 'LONG' else (trade['Entry_Price'] - exit_price) * trade['Quantity']
                    capital += (trade['Entry_Price'] * trade['Quantity']) + pnl
                    trade.update({'Exit_Time': current_candle.name, 'Exit_Price': exit_price, 'PnL': pnl, 'Exit_Reason': exit_reason})
                    trade_log.append(trade)
                    del open_positions[ticker]
                    continue

            # --- B. Check for Entries ---
            if ticker not in open_positions:
                history = indicator_data.iloc[i - divergence_lookback : i]
                history = history[history['ticker'] == ticker]
                if len(history) < divergence_lookback - 5: continue

                # Bullish Divergence
                price_ll_idx = history['Low'].idxmin()
                price_ll_val = history.loc[price_ll_idx, 'Low']
                rsi_at_ll = history.loc[price_ll_idx, 'rsi']
                prior_history_bull = history.loc[:price_ll_idx].iloc[:-1]
                if not prior_history_bull.empty:
                    prior_price_l_idx = prior_history_bull['Low'].idxmin()
                    prior_price_l_val = prior_history_bull.loc[prior_price_l_idx, 'Low']
                    is_bullish_divergence = price_ll_val < prior_price_l_val and rsi_at_ll > prior_history_bull.loc[prior_price_l_idx, 'rsi']

                    if (is_bullish_divergence and (current_candle['adx'] > ADX_THRESHOLD and current_candle['Volume'] > current_candle['vol_avg'] * VOLUME_MULTIPLIER)):
                        entry_price, stop_loss = current_candle['Open'], price_ll_val
                        if entry_price > stop_loss:
                            risk = entry_price - stop_loss
                            qty = int((capital * RISK_PER_TRADE_PERCENT) / risk)
                            if qty > 0:
                                open_positions[ticker] = {'Ticker': ticker, 'Direction': 'LONG', 'Entry_Time': current_candle.name, 'Entry_Price': entry_price, 'Quantity': qty, 'Stop_Loss': stop_loss, 'Take_Profit': entry_price + (risk * RISK_REWARD_RATIO)}
                                continue

                # Bearish Divergence
                price_hh_idx = history['High'].idxmax()
                price_hh_val = history.loc[price_hh_idx, 'High']
                rsi_at_hh = history.loc[price_hh_idx, 'rsi']
                prior_history_bear = history.loc[:price_hh_idx].iloc[:-1]
                if not prior_history_bear.empty:
                    prior_price_h_idx = prior_history_bear['High'].idxmax()
                    prior_price_h_val = prior_history_bear.loc[prior_price_h_idx, 'High']
                    is_bearish_divergence = price_hh_val > prior_price_h_val and rsi_at_hh < prior_history_bear.loc[prior_price_h_idx, 'rsi']

                    if (is_bearish_divergence and (current_candle['adx'] > ADX_THRESHOLD and current_candle['Volume'] > current_candle['vol_avg'] * VOLUME_MULTIPLIER)):
                        entry_price, stop_loss = current_candle['Open'], price_hh_val
                        if entry_price < stop_loss:
                            risk = stop_loss - entry_price
                            qty = int((capital * RISK_PER_TRADE_PERCENT) / risk)
                            if qty > 0:
                                open_positions[ticker] = {'Ticker': ticker, 'Direction': 'SHORT', 'Entry_Time': current_candle.name, 'Entry_Price': entry_price, 'Quantity': qty, 'Stop_Loss': stop_loss, 'Take_Profit': entry_price - (risk * RISK_REWARD_RATIO)}

        print("\n--- Backtest Simulation Complete ---")
        if open_positions:
            print("Closing open positions at the end of the backtest period...")
            for ticker, trade in list(open_positions.items()):
                exit_price = indicator_data[indicator_data['ticker'] == ticker].iloc[-1]['Close']
                pnl = (exit_price - trade['Entry_Price']) * trade['Quantity'] if trade['Direction'] == 'LONG' else (trade['Entry_Price'] - exit_price) * trade['Quantity']
                capital += (trade['Entry_Price'] * trade['Quantity']) + pnl
                trade.update({'Exit_Time': indicator_data.index[-1], 'Exit_Price': exit_price, 'PnL': pnl, 'Exit_Reason': 'End of Backtest'})
                trade_log.append(trade)
                del open_positions[ticker]

# ==============================================================================
# PART 3: Performance Reporting
# ==============================================================================
if 'trade_log' in locals() and trade_log:
    print("\n--- Backtest Performance Report ---")

    tradelog_df = pd.DataFrame(trade_log)
    tradelog_df = tradelog_df[[
        'Ticker', 'Direction', 'Entry_Time', 'Entry_Price',
        'Exit_Time', 'Exit_Price', 'Stop_Loss', 'Take_Profit',
        'Quantity', 'PnL', 'Exit_Reason'
    ]]
    tradelog_df['Entry_Price'] = tradelog_df['Entry_Price'].round(2)
    tradelog_df['Exit_Price'] = tradelog_df['Exit_Price'].round(2)
    tradelog_df['PnL'] = tradelog_df['PnL'].round(2)

    print("\n--- Trade Log ---")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(tradelog_df)

    net_profit = tradelog_df['PnL'].sum()
    net_profit_pct = (net_profit / INITIAL_CAPITAL) * 100
    total_trades = len(tradelog_df)

    wins = tradelog_df[tradelog_df['PnL'] > 0]
    losses = tradelog_df[tradelog_df['PnL'] <= 0]

    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0

    gross_profit = wins['PnL'].sum()
    gross_loss = abs(losses['PnL'].sum())

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    tradelog_df.sort_values(by='Exit_Time', inplace=True)
    tradelog_df['Cumulative_PnL'] = tradelog_df['PnL'].cumsum()
    tradelog_df['Equity'] = INITIAL_CAPITAL + tradelog_df['Cumulative_PnL']

    tradelog_df['Running_Max_Equity'] = tradelog_df['Equity'].cummax()
    tradelog_df['Drawdown'] = tradelog_df['Running_Max_Equity'] - tradelog_df['Equity']
    tradelog_df['Drawdown_Pct'] = (tradelog_df['Drawdown'] / tradelog_df['Running_Max_Equity']) * 100

    max_drawdown_pct = tradelog_df['Drawdown_Pct'].max() if not tradelog_df.empty else 0.0

    print("\n--- Summary Metrics ---")
    print(f"Initial Capital:         ₹{INITIAL_CAPITAL:,.2f}")
    print(f"Final Capital:           ₹{(INITIAL_CAPITAL + net_profit):,.2f}")
    print(f"Net Profit/Loss:         ₹{net_profit:,.2f} ({net_profit_pct:.2f}%)")
    print("-" * 30)
    print(f"Total Trades:            {total_trades}")
    print(f"Winning Trades:          {len(wins)}")
    print(f"Losing Trades:           {len(losses)}")
    print(f"Win Rate:                {win_rate:.2f}%")
    print("-" * 30)
    print(f"Profit Factor:           {profit_factor:.2f}")
    print(f"Max Drawdown:            {max_drawdown_pct:.2f}%")
    print("-" * 30)

else:
    print("\n--- No trades were executed. No performance report to generate. ---")
