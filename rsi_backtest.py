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
TRAILING_STOP_PCT = 0.6 # Used for initial stop and breakeven trigger

# Volume parameters
VOLUME_MA_PERIOD = 20
VOLUME_FACTOR = 1.5

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

    # Calculate Volume MA
    data['volume_ma'] = data['Volume'].rolling(window=VOLUME_MA_PERIOD).mean()

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
        # Get all required values for the current 5-minute candle, ensuring they are scalar
        current_price = data['Close'].iloc[i].item()
        current_rsi = data['rsi'].iloc[i].item()
        prev_rsi = data['rsi'].iloc[i-1].item()
        adx = data['adx'].iloc[i].item()
        volume = data['Volume'].iloc[i].item()
        volume_ma = data['volume_ma'].iloc[i].item()
        close_1h = data['close_1h'].iloc[i].item()
        ema_50_1h = data['ema_50_1h'].iloc[i].item()

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

            # Move to Breakeven Logic
            if not stop_moved_to_be:
                if position == 'LONG' and current_price >= entry_price * (1 + TRAILING_STOP_PCT / 100):
                    stop_loss = entry_price
                    stop_moved_to_be = True
                    # print(f"Stop moved to breakeven for LONG trade at {data.index[i]}")
                elif position == 'SHORT' and current_price <= entry_price * (1 - TRAILING_STOP_PCT / 100):
                    stop_loss = entry_price
                    stop_moved_to_be = True
                    # print(f"Stop moved to breakeven for SHORT trade at {data.index[i]}")

            # Check for Stop Loss or Take Profit
            if position == 'LONG':
                if current_price <= stop_loss:
                    pnl = (stop_loss - entry_price) * quantity
                    exit_reason = "Stop Loss"
                elif current_price >= take_profit:
                    pnl = (take_profit - entry_price) * quantity
                    exit_reason = "Take Profit"

            elif position == 'SHORT':
                if current_price >= stop_loss:
                    pnl = (entry_price - stop_loss) * quantity
                    exit_reason = "Stop Loss"
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
            # --- CONFLUENCE ENTRY LOGIC ---

            # 1. Trend Filter (HTF)
            is_uptrend = close_1h > ema_50_1h
            is_downtrend = close_1h < ema_50_1h

            # 2. Momentum Filter (LTF)
            has_momentum = adx > ADX_THRESHOLD

            # 3. RSI Signal (LTF)
            long_rsi_signal = rsi_was_oversold and prev_rsi < RSI_LONG_ENTRY and current_rsi >= RSI_LONG_ENTRY
            short_rsi_signal = rsi_was_overbought and prev_rsi > RSI_SHORT_ENTRY and current_rsi <= RSI_SHORT_ENTRY

            # 4. Volume Confirmation (LTF)
            has_volume = volume > volume_ma * VOLUME_FACTOR

            # Long Entry
            if is_uptrend and has_momentum and long_rsi_signal and has_volume:
                position = 'LONG'
                entry_price = current_price
                entry_time = data.index[i]

                take_profit = entry_price * (1 + TAKE_PROFIT_PCT / 100)
                stop_loss = entry_price * (1 - TRAILING_STOP_PCT / 100) # This is now a fixed stop

                risk_per_share = entry_price - stop_loss
                risk_amount = (capital * risk_pct) / 100
                quantity = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0

                if quantity > 0:
                    print(f"\nNew LONG trade initiated at {entry_time}")
                    stop_moved_to_be = False # Reset breakeven flag
                else:
                    position = None

            # Short Entry
            elif is_downtrend and has_momentum and short_rsi_signal and has_volume:
                position = 'SHORT'
                entry_price = current_price
                entry_time = data.index[i]

                take_profit = entry_price * (1 - TAKE_PROFIT_PCT / 100)
                stop_loss = entry_price * (1 + TRAILING_STOP_PCT / 100) # This is now a fixed stop

                risk_per_share = stop_loss - entry_price
                risk_amount = (capital * risk_pct) / 100
                quantity = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0

                if quantity > 0:
                    print(f"\nNew SHORT trade initiated at {entry_time}")
                    stop_moved_to_be = False # Reset breakeven flag
                else:
                    position = None

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
            # 1. Fetch Data for multiple timeframes
            data_5m = fetch_data(ticker, START_DATE, END_DATE, interval="5m")
            data_1h = fetch_data(ticker, START_DATE, END_DATE, interval="1h")

            if data_5m is None or data_5m.empty or data_1h is None or data_1h.empty:
                print(f"Could not fetch sufficient data for {ticker}. Skipping.")
                continue

            # 2. Prepare HTF (Higher Timeframe) indicators and merge
            ema_close_1h = pd.Series(data_1h['Close'].values.flatten(), index=data_1h.index)
            data_1h['ema_50_1h'] = ta.trend.EMAIndicator(ema_close_1h, window=50).ema_indicator()
            htf_data = data_1h[['ema_50_1h', 'Close']].rename(columns={'Close': 'close_1h'})

            # Forward-fill the 1-hour data onto the 5-minute timeline
            merged_data = pd.merge_asof(data_5m.sort_index(), htf_data.sort_index(), left_index=True, right_index=True, direction='backward')
            merged_data.dropna(inplace=True)

            if merged_data.empty:
                print(f"Data for {ticker} could not be merged. Skipping.")
                continue

            # 3. Calculate LTF (Lower Timeframe) indicators
            stock_data = calculate_indicators(merged_data)

            # 4. Run Backtest
            trade_log, equity_curve = run_backtest(stock_data, INITIAL_CAPITAL, RISK_PER_TRADE_PCT)

            # 5. Analyze Performance
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
