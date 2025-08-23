# -*- coding: utf-8 -*-
"""
This script fetches historical price data from the Shoonya API by Finvasia.
"""

# Import necessary libraries
import pandas as pd
import pyotp
from api_helper import ShoonyaApiPy
from datetime import datetime, timedelta

# --- Step 1: Add your credentials here ---
# Replace the placeholder values with your actual Shoonya API credentials.
user = "YOUR_USER_ID"          # Your user ID
pwd = "YOUR_PASSWORD"        # Your password
totp_secret = "YOUR_TOTP_SECRET"  # Your TOTP secret key for 2FA
vc = "YOUR_VENDOR_CODE"      # Your vendor code
api_key = "YOUR_API_KEY"       # Your API key
imei = "YOUR_IMEI"           # Your IMEI or a unique identifier

# --- Step 2: Initialize the API and Generate TOTP ---
api = ShoonyaApiPy()
factor2 = pyotp.TOTP(totp_secret).now() # Generate the current TOTP

# --- Step 3: Login to the API ---
try:
    ret = api.login(userid=user, password=pwd, twoFA=factor2, vendor_code=vc, api_secret=api_key, imei=imei)

    if ret and ret['stat'] == 'Ok':
        print("Login successful.")

        # --- Step 4: Define parameters for historical data ---
        exchange = 'NSE'  # Exchange
        token = '22'      # Token for NIFTY 50

        # Calculate the start time for the last 30 days
        endtime = datetime.now()
        starttime = endtime - timedelta(days=30)

        # Format dates for the API
        starttime_str = starttime.strftime('%d-%m-%Y %H:%M:%S')

        # --- Step 5: Fetch historical data ---
        print(f"Fetching data for NIFTY 50 from {starttime_str} to now...")
        result = api.get_time_price_series(exchange=exchange, token=token, starttime=starttime.timestamp(), interval='1D')

        if result:
            # --- Step 6: Convert to pandas DataFrame and process data ---
            df = pd.DataFrame(result)

            # Rename columns for clarity
            df.rename(columns={
                'into': 'open',
                'inth': 'high',
                'intl': 'low',
                'intc': 'close',
                'intv': 'volume',
                'time': 'date'
            }, inplace=True)

            # Convert the 'date' column to a more readable format
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M:%S').dt.date

            # --- Step 7: Display the first 10 rows of the DataFrame ---
            print("\nFirst 10 rows of historical data:")
            print(df.head(10))

        else:
            print("Failed to fetch historical data.")

    else:
        print(f"Login failed: {ret}")

except Exception as e:
    print(f"An error occurred: {e}")
