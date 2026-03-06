import pandas as pd
import os
import glob

# --- 1. Load the Data ---
DATA_PATH = 'transactions.pkl' 

if not os.path.exists(DATA_PATH):
    print(f"Error: Could not find '{DATA_PATH}'. Check the name and location.")
    exit()

if os.path.isdir(DATA_PATH):
    print(f"'{DATA_PATH}' is a folder! Looking for .pkl files inside...")
    # Find all .pkl files inside the folder
    all_files = glob.glob(os.path.join(DATA_PATH, "*.pkl"))
    
    if len(all_files) == 0:
        print("Error: No .pkl files found inside the folder.")
        exit()
        
    print(f"Found {len(all_files)} files. Combining them now... (This might take a moment)")
    
    # Read each file and combine them into one massive DataFrame
    df_list = [pd.read_pickle(file) for file in all_files]
    df = pd.concat(df_list, ignore_index=True)
else:
    print(f"'{DATA_PATH}' is a single file. Loading...")
    df = pd.read_pickle(DATA_PATH)

print("Data successfully loaded!")

# --- Handle Column Names (Spaces vs Underscores) ---
fraud_col = 'TX_FRAUD' if 'TX_FRAUD' in df.columns else 'TX FRAUD'
amount_col = 'TX_AMOUNT' if 'TX_AMOUNT' in df.columns else 'TX AMOUNT'

# --- 2. Dataset Overview ---
print("\n--- DATASET OVERVIEW ---")
print(f"Total Transactions: {len(df)}")
print(df.head()) 

# --- 3. Check Fraud Imbalance ---
fraud_counts = df[fraud_col].value_counts()
legit = fraud_counts.get(0, 0)
fraud = fraud_counts.get(1, 0)
fraud_percentage = (fraud / len(df)) * 100

print("\n--- FRAUD STATISTICS ---")
print(f"Legitimate Transactions (0): {legit}")
print(f"Fraudulent Transactions (1): {fraud}")
print(f"Percentage of Fraud: {fraud_percentage:.2f}%")

# --- 4. Verify Scenario 1 (Amount > 220) ---
high_amount_tx = df[df[amount_col] > 220]
high_amount_fraud = high_amount_tx[high_amount_tx[fraud_col] == 1]

print("\n--- VERIFYING SCENARIO 1 ---")
print(f"Total transactions over 220: {len(high_amount_tx)}")
print(f"How many of those are marked as fraud? {len(high_amount_fraud)}")

if len(high_amount_tx) > 0 and len(high_amount_tx) == len(high_amount_fraud):
    print("Scenario 1 confirmed: All transactions > 220 are fraudulent!")
else:
    print("The rule isn't 100% perfect in the data.")