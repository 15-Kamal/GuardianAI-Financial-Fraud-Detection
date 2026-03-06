import pandas as pd
import os
import glob
import time

DATA_PATH = 'transactions.pkl' 
OUTPUT_FILE = 'engineered_transactions.pkl'

print("1. Loading all 1.75 million transactions... (Please wait)")
start_time = time.time()

# Load and combine all .pkl files
all_files = glob.glob(os.path.join(DATA_PATH, "*.pkl"))
df = pd.concat([pd.read_pickle(f) for f in all_files], ignore_index=True)

print(f"Data loaded in {time.time() - start_time:.1f} seconds.")

# --- SORT CHRONOLOGICALLY ---
# It is critical to sort by time when building historical features
print("2. Sorting data chronologically...")
df = df.sort_values('TX_DATETIME').reset_index(drop=True)

# --- ENGINEER NEW FEATURES ---
print("3. Engineering Time, Customer, and Terminal Features...")

# Feature 1: Time of Day & Day of Week
# Fraud often happens at unusual hours.
df['TX_HOUR'] = df['TX_DATETIME'].dt.hour
df['TX_DAY_OF_WEEK'] = df['TX_DATETIME'].dt.dayofweek

# Feature 2: Customer Spending Habits (Scenario 3)
# Calculate the customer's historical average transaction amount
df['CUST_AVG_AMOUNT'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform('mean')

# Calculate the Ratio: How much larger is this transaction compared to their normal average?
# A sudden spike will result in a high ratio, flagging the AI to look closer.
df['CUST_SPENDING_RATIO'] = df['TX_AMOUNT'] / df['CUST_AVG_AMOUNT']

# Feature 3: Terminal Usage Patterns (Scenario 2)
# Calculate how many times this specific terminal was used on this exact day
df['TX_DATE'] = df['TX_DATETIME'].dt.date
df['TERM_DAILY_TX_COUNT'] = df.groupby(['TERMINAL_ID', 'TX_DATE'])['TRANSACTION_ID'].transform('count')

# Drop the temporary date column
df.drop('TX_DATE', axis=1, inplace=True)

# --- SAVE THE NEW DATASET ---
print(f"4. Saving the engineered dataset to {OUTPUT_FILE}...")
df.to_pickle(OUTPUT_FILE)

print("Feature Engineering Complete! New file saved.")
print(f"Total processing time: {time.time() - start_time:.1f} seconds.")

# Show the new columns
print("\n--- NEW FEATURE COLUMNS ---")
print(df[['TX_AMOUNT', 'CUST_AVG_AMOUNT', 'CUST_SPENDING_RATIO', 'TX_HOUR', 'TERM_DAILY_TX_COUNT']].head())