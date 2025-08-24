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

try:
    from scipy.signal import find_peaks
except ImportError:
    install('scipy')
    from scipy.signal import find_peaks

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
TRAILING_STOP_PCT = 0.6 # Used for initial stop and breakeven trigger

# Volume parameters
VOLUME_MA_PERIOD = 20
VOLUME_FACTOR = 1.5

# Bollinger Band parameters
BB_PERIOD = 20
BB_STD_DEV = 2
BBW_SQUEEZE_THRESHOLD = 0.015 # Bollinger Band Width squeeze threshold

# Divergence Parameters
DIVERGENCE_LOOKBACK = 30 # Lookback period for finding divergence
DIVERGENCE_PEAK_DISTANCE = 5 # Min distance between peaks/troughs for divergence

# --- 2. Data Fetching ---
def fetch_data(ticker, start, end, interval):
    """Fetches historical data from yfinance and standardizes column names."""
    print(f"Fetching data for {ticker} from {start} to {end} with {interval} interval...")
    data = yfinance.download(ticker, start=start, end=end, interval=interval)
    if data.empty:
        print(f"No data found for {ticker}. Please check the ticker and date range.")
        return None

    # Flatten MultiIndex columns if they exist (e.g., ('Close', 'RELIANCE.NS'))
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Standardize column names to lowercase
    data.columns = [str(col).lower() for col in data.columns]

    data.dropna(inplace=True)
    print("Data fetched successfully.")
    return data

# --- 3. Indicator Calculation ---
def calculate_indicators(data):
    """Calculates all required indicators and joins them to the main dataframe."""

    # Create a new DataFrame for the indicators to avoid index issues
    indicators = pd.DataFrame(index=data.index)

    # Ensure source columns are 1D Series before passing to indicators
    close = data['close'].squeeze()
    high = data['high'].squeeze()
    low = data['low'].squeeze()
    volume = data['volume'].squeeze()

    # Calculate indicators
    indicators['rsi'] = ta.momentum.RSIIndicator(close, window=RSI_PERIOD).rsi()
    indicators['adx'] = ta.trend.ADXIndicator(high, low, close, window=ADX_PERIOD).adx()
    indicators['volume_ma'] = volume.rolling(window=VOLUME_MA_PERIOD).mean()
    indicators['bbw'] = ta.volatility.BollingerBands(close, window=BB_PERIOD, window_dev=BB_STD_DEV).bollinger_wband()

    # Join the new indicators back to the original data
    data = data.join(indicators)

    data.dropna(inplace=True)
    return data

def find_divergence(prices, indicator, lookback, peak_distance):
    """
    Finds bullish or bearish divergence between price and an indicator.
    Returns: 'BULLISH', 'BEARISH', or None
    """
    # Find peaks (highs) and troughs (lows)
    # For troughs, we find peaks in the negative series
    price_highs, _ = find_peaks(prices, distance=peak_distance)
    price_lows, _ = find_peaks(-prices, distance=peak_distance)
    indicator_highs, _ = find_peaks(indicator, distance=peak_distance)
    indicator_lows, _ = find_peaks(-indicator, distance=peak_distance)

    # Check for Bearish Divergence (Higher High in Price, Lower High in Indicator)
    if len(price_highs) >= 2 and len(indicator_highs) >= 2:
        last_price_high = prices[price_highs[-1]]
        prev_price_high = prices[price_highs[-2]]
        last_indicator_high = indicator[indicator_highs[-1]]
        prev_indicator_high = indicator[indicator_highs[-2]]

        if last_price_high > prev_price_high and last_indicator_high < prev_indicator_high:
            return 'BEARISH'

    # Check for Bullish Divergence (Lower Low in Price, Higher Low in Indicator)
    if len(price_lows) >= 2 and len(indicator_lows) >= 2:
        last_price_low = prices[price_lows[-1]]
        prev_price_low = prices[price_lows[-2]]
        last_indicator_low = indicator[indicator_lows[-1]]
        prev_indicator_low = indicator[indicator_lows[-2]]

        if last_price_low < prev_price_low and last_indicator_low > prev_indicator_low:
            return 'BULLISH'

    return None

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

    # State management for divergence
    divergence_signal = None

    for i in range(DIVERGENCE_LOOKBACK, len(data)):
        # Get all required values for the current 5-minute candle, ensuring they are scalar
        current_price = data['close'].iloc[i].item()
        current_rsi = data['rsi'].iloc[i].item()
        prev_rsi = data['rsi'].iloc[i-1].item()
        adx = data['adx'].iloc[i].item()
        volume = data['volume'].iloc[i].item()
        volume_ma = data['volume_ma'].iloc[i].item()
        close_1h = data['close_1h'].iloc[i].item()
        ema_50_1h = data['ema_50_1h'].iloc[i].item()
        bbw = data['bbw'].iloc[i].item()

        # --- Divergence Detection ---
        # We only check for divergence if we are not in a position
        if not position:
            price_slice = data['close'].iloc[i-DIVERGENCE_LOOKBACK:i+1].values.flatten()
            rsi_slice = data['rsi'].iloc[i-DIVERGENCE_LOOKBACK:i+1].values.flatten()
            divergence_signal = find_divergence(price_slice, rsi_slice, DIVERGENCE_LOOKBACK, DIVERGENCE_PEAK_DISTANCE)

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

        # --- Position Management (Trend Rider Exit Logic) ---
        if position:
            pnl = 0
            exit_reason = None

            if position == 'LONG':
                # Stage 3: Trailing stop after breakeven
                if stop_moved_to_be:
                    peak_price = max(peak_price, current_price)
                    stop_loss = peak_price * (1 - 1.5 / 100) # 1.5% trail from peak
                # Stage 2: Move to breakeven
                elif not stop_moved_to_be and current_price >= entry_price * (1 + TRAILING_STOP_PCT / 100):
                    stop_loss = entry_price
                    stop_moved_to_be = True

                # Check for exit (Stage 1 initial stop is the default)
                if current_price <= stop_loss:
                    pnl = (stop_loss - entry_price) * quantity
                    exit_reason = "Stop Loss"

            elif position == 'SHORT':
                # Stage 3: Trailing stop after breakeven
                if stop_moved_to_be:
                    peak_price = min(peak_price, current_price)
                    stop_loss = peak_price * (1 + 1.5 / 100) # 1.5% trail from peak
                # Stage 2: Move to breakeven
                elif not stop_moved_to_be and current_price <= entry_price * (1 - TRAILING_STOP_PCT / 100):
                    stop_loss = entry_price
                    stop_moved_to_be = True

                # Check for exit (Stage 1 initial stop is the default)
                if current_price >= stop_loss:
                    pnl = (entry_price - stop_loss) * quantity
                    exit_reason = "Stop Loss"

            # NOTE: Fixed Take Profit has been removed for the Trend Rider logic

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
            # --- DIVERGENCE CONFIRMATION ENTRY LOGIC (6 STEPS) ---

            # 1. Volatility Filter
            is_volatile = bbw > BBW_SQUEEZE_THRESHOLD

            # 2. Trend Filter
            is_uptrend = close_1h > ema_50_1h
            is_downtrend = close_1h < ema_50_1h

            # 3. Momentum Filter
            has_momentum = adx > ADX_THRESHOLD

            # 4. RSI Divergence Signal (already detected and stored in divergence_signal)

            # 5. RSI Entry Trigger
            long_rsi_trigger = rsi_was_oversold and prev_rsi < RSI_LONG_ENTRY and current_rsi >= RSI_LONG_ENTRY
            short_rsi_trigger = rsi_was_overbought and prev_rsi > RSI_SHORT_ENTRY and current_rsi <= RSI_SHORT_ENTRY

            # 6. Volume Confirmation
            has_volume = volume > volume_ma * VOLUME_FACTOR

            # Long Entry
            if (is_volatile and is_uptrend and has_momentum and
                divergence_signal == 'BULLISH' and long_rsi_trigger and has_volume):
                position = 'LONG'
                entry_price = current_price
                entry_time = data.index[i]

                stop_loss = entry_price * (1 - TRAILING_STOP_PCT / 100)

                risk_per_share = entry_price - stop_loss
                risk_amount = (capital * risk_pct) / 100
                quantity = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0

                if quantity > 0:
                    print(f"\nNew LONG trade initiated at {entry_time}")
                    stop_moved_to_be = False
                    peak_price = entry_price
                else:
                    position = None

            # Short Entry
            elif (is_volatile and is_downtrend and has_momentum and
                  divergence_signal == 'BEARISH' and short_rsi_trigger and has_volume):
                position = 'SHORT'
                entry_price = current_price
                entry_time = data.index[i]

                stop_loss = entry_price * (1 + TRAILING_STOP_PCT / 100)

                risk_per_share = stop_loss - entry_price
                risk_amount = (capital * risk_pct) / 100
                quantity = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0

                if quantity > 0:
                    print(f"\nNew SHORT trade initiated at {entry_time}")
                    stop_moved_to_be = False
                    peak_price = entry_price
                else:
                    position = None

        equity_curve.append(capital)

    return pd.DataFrame(trade_log), pd.Series(equity_curve, index=data.index[DIVERGENCE_LOOKBACK-1:])


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
        "CANBK.NS", "SBFC.NS", "IRFC.NS", "JAICORPLTD.NS", "SAMMAANCAP.NS",
        "NHPC.NS", "IOC.NS", "ASHOKLEY.NS", "MRPL.NS", "REDTAPE.NS",
        "WELSPUNLIV.NS", "IREDA.NS", "NBCC.NS", "UNIONBANK.NS", "IEX.NS",
        "PRSMJOHNSN.NS", "RBA.NS", "VMM.NS", "LLOYDSENT.NS", "SAIL.NS", "J&KBANK.NS",
        "IDBI.NS", "TEXRAIL.NS", "MOTHERSON.NS", "EDELWEISS.NS", "ZEEL.NS",
        "GMRAIRPORT.NS", "HEMIPROP.NS"
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
            data_1h['ema_50_1h'] = ta.trend.EMAIndicator(data_1h['close'].squeeze(), window=50).ema_indicator()
            htf_data = data_1h[['ema_50_1h', 'close']].rename(columns={'close': 'close_1h'})

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
