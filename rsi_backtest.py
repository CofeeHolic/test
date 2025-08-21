import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install required packages if not already installed
try:
    import yfinance
except ImportError:
    install('yfinance')
    import yfinance

try:
    import pandas as pd
except ImportError:
    install('pandas')
    import pandas as pd

try:
    import ta
except ImportError:
    install('ta')
    import ta

try:
    import matplotlib.pyplot as plt
except ImportError:
    install('matplotlib')
    import matplotlib.pyplot as plt

# --- 1. Parameters ---
START_DATE = "2025-07-01"
END_DATE = "2025-08-20"
INTERVAL = "5m"
INITIAL_CAPITAL = 100000
RISK_PER_TRADE_PCT = 1.0

# RSI parameters
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
RSI_LONG_ENTRY = 40
RSI_SHORT_ENTRY = 60

# ADX parameters
ADX_PERIOD = 14
ADX_THRESHOLD = 25

# Trade parameters
TAKE_PROFIT_PCT = 1.0
TRAILING_STOP_PCT = 0.6

# --- 2. Data Fetching ---
def fetch_data(ticker, start, end, interval):
    """Fetches historical data from yfinance."""
    print(f"Fetching data for {ticker} from {start} to {end} with {interval} interval...")
    data = yfinance.download(ticker, start=start, end=end, interval=interval)
    if data.empty:
        print(f"No data found for {ticker}. Please check the ticker and date range.")
        return None
    data.dropna(inplace=True)
    print("Data fetched successfully.")
    return data

# --- 3. Indicator Calculation ---
def calculate_indicators(data):
    """Calculates RSI and other indicators."""
    # Force data into 1D pandas Series to fix issues with some yfinance/pandas versions
    close_series = pd.Series(data['Close'].values.flatten(), index=data.index)
    high_series = pd.Series(data['High'].values.flatten(), index=data.index)
    low_series = pd.Series(data['Low'].values.flatten(), index=data.index)

    # Calculate RSI
    data['rsi'] = ta.momentum.RSIIndicator(close_series, window=RSI_PERIOD).rsi()

    # Calculate ADX
    adx_indicator = ta.trend.ADXIndicator(high_series, low_series, close_series, window=ADX_PERIOD)
    data['adx'] = adx_indicator.adx()

    data.dropna(inplace=True)
    return data

# --- 4. Backtesting Engine ---
def run_backtest(data, initial_capital, risk_pct):
    """Runs the backtesting simulation."""
    capital = initial_capital
    trade_log = []
    position = None  # Can be 'LONG', 'SHORT', or None
    equity_curve = [initial_capital]

    # Track RSI state
    rsi_was_oversold = False
    rsi_was_overbought = False

    for i in range(1, len(data)):
        # yfinance can return a multi-index, so we use .item() to get the scalar
        # value from the 'Close' column to avoid FutureWarnings.
        # Calculated columns ('rsi', 'adx') are simple Series, so .iloc is fine.
        current_price = data['Close'].iloc[i].item()
        current_rsi = data['rsi'].iloc[i]
        prev_rsi = data['rsi'].iloc[i-1]
        adx = data['adx'].iloc[i]

        # Update RSI state flags
        if prev_rsi < RSI_OVERSOLD:
            rsi_was_oversold = True
        if prev_rsi > RSI_OVERBOUGHT:
            rsi_was_overbought = True

        # Reset flags if RSI crosses middle line to avoid stale signals
        if prev_rsi < 50 and current_rsi >= 50:
            rsi_was_oversold = False
        if prev_rsi > 50 and current_rsi <= 50:
            rsi_was_overbought = False

        # --- Position Management ---
        if position:
            pnl = 0
            exit_reason = None

            # Trailing Stop Loss Update
            if position == 'LONG':
                # Update trailing stop
                new_trailing_stop = current_price * (1 - TRAILING_STOP_PCT / 100)
                trailing_stop_loss = max(trailing_stop_loss, new_trailing_stop)

                # Check for exit
                if current_price <= trailing_stop_loss:
                    pnl = (trailing_stop_loss - entry_price) * quantity
                    exit_reason = "Trailing Stop"
                elif current_price >= take_profit:
                    pnl = (take_profit - entry_price) * quantity
                    exit_reason = "Take Profit"

            elif position == 'SHORT':
                # Update trailing stop
                new_trailing_stop = current_price * (1 + TRAILING_STOP_PCT / 100)
                trailing_stop_loss = min(trailing_stop_loss, new_trailing_stop)

                # Check for exit
                if current_price >= trailing_stop_loss:
                    pnl = (entry_price - trailing_stop_loss) * quantity
                    exit_reason = "Trailing Stop"
                elif current_price <= take_profit:
                    pnl = (entry_price - take_profit) * quantity
                    exit_reason = "Take Profit"

            if exit_reason:
                capital += pnl
                trade_log.append({
                    "EntryDate": entry_time,
                    "EntryPrice": entry_price,
                    "ExitDate": data.index[i],
                    "ExitPrice": current_price,
                    "Direction": position,
                    "Quantity": quantity,
                    "PnL": pnl,
                    "ExitReason": exit_reason
                })
                position = None
                rsi_was_oversold = False
                rsi_was_overbought = False

        # --- Entry Logic ---
        if not position:
            # Long Entry
            if rsi_was_oversold and prev_rsi < RSI_LONG_ENTRY and current_rsi >= RSI_LONG_ENTRY and adx > ADX_THRESHOLD:
                position = 'LONG'
                entry_price = current_price
                entry_time = data.index[i]

                take_profit = entry_price * (1 + TAKE_PROFIT_PCT / 100)
                trailing_stop_loss = entry_price * (1 - TRAILING_STOP_PCT / 100)

                risk_per_share = entry_price - trailing_stop_loss
                risk_amount = (capital * risk_pct) / 100
                quantity = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0

                if quantity > 0:
                    print(f"\nNew LONG trade initiated at {entry_time}")
                else:
                    position = None # Invalidate trade if quantity is 0

            # Short Entry
            elif rsi_was_overbought and prev_rsi > RSI_SHORT_ENTRY and current_rsi <= RSI_SHORT_ENTRY and adx > ADX_THRESHOLD:
                position = 'SHORT'
                entry_price = current_price
                entry_time = data.index[i]

                take_profit = entry_price * (1 - TAKE_PROFIT_PCT / 100)
                trailing_stop_loss = entry_price * (1 + TRAILING_STOP_PCT / 100)

                risk_per_share = trailing_stop_loss - entry_price
                risk_amount = (capital * risk_pct) / 100
                quantity = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0

                if quantity > 0:
                    print(f"\nNew SHORT trade initiated at {entry_time}")
                else:
                    position = None # Invalidate trade if quantity is 0

        equity_curve.append(capital)

    return pd.DataFrame(trade_log), pd.Series(equity_curve, index=data.index)


# --- 5. Performance Analysis ---
def analyze_performance(trade_log, equity_curve, initial_capital):
    """Calculates and returns a dictionary of performance metrics."""
    if trade_log.empty:
        return {
            "Total PnL": 0,
            "Total Trades": 0,
            "Win Rate": 0,
            "Max Drawdown": 0,
            "Final Portfolio Value": initial_capital
        }

    num_trades = len(trade_log)
    wins = trade_log[trade_log['PnL'] > 0]

    win_rate = (len(wins) / num_trades) * 100 if num_trades > 0 else 0

    final_capital = equity_curve.iloc[-1]
    total_pnl = final_capital - initial_capital

    # Max Drawdown
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = drawdown.min() * 100

    return {
        "Total PnL": round(total_pnl, 2),
        "Total Trades": num_trades,
        "Win Rate": round(win_rate, 2),
        "Max Drawdown": round(max_drawdown, 2),
        "Final Portfolio Value": round(final_capital, 2)
    }

# --- Main Execution ---
if __name__ == "__main__":
    TICKER_LIST = [
        "DCBBANK.NS", "GMRP&UI.NS", "EMIL.NS", "GAEL.NS", "PNB.NS", "JAMNAAUTO.NS",
        "NFL.NS", "EMBDL.NS", "SPARC.NS", "BAJAJHFL.NS", "BANKINDIA.NS",
        "LEMONTREE.NS", "STLTECH.NS", "JAIBALAJI.NS", "NTPCGREEN.NS", "NIVABUPA.NS",
        "INOXWIND.NS", "BEPL.NS", "ELECTCAST.NS", "SJVN.NS", "TVSSCS.NS",
        "CANBK.NS", "SBFC.NS", "IRFC.NS", "JAICORPLTD.NS", "SAMMAANCAP.NS"
    ]

    all_results = []

    for ticker in TICKER_LIST:
        print(f"--- Running backtest for {ticker} ---")
        try:
            # 1. Fetch Data
            stock_data = fetch_data(ticker, START_DATE, END_DATE, INTERVAL)

            if stock_data is not None and not stock_data.empty:
                # 2. Calculate Indicators
                stock_data = calculate_indicators(stock_data)

                # 3. Run Backtest
                trade_log, equity_curve = run_backtest(stock_data, INITIAL_CAPITAL, RISK_PER_TRADE_PCT)

                # 4. Analyze Performance
                metrics = analyze_performance(trade_log, equity_curve, INITIAL_CAPITAL)
                metrics['Ticker'] = ticker
                all_results.append(metrics)
                print(f"Backtest for {ticker} complete. PnL: {metrics['Total PnL']}")

        except Exception as e:
            print(f"An error occurred for ticker {ticker}: {e}")

    # Create and print summary DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.sort_values(by="Total PnL", ascending=False, inplace=True)
        print("\n\n--- Aggregated Backtest Results ---")
        print(results_df.to_string())
