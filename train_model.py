import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# --- 1. Load the Engineered Data ---
DATA_PATH = 'engineered_transactions.pkl'
print("1. Loading engineered dataset...")
df = pd.read_pickle(DATA_PATH)

# --- 2. Select Features (X) and Target (y) ---
# We must drop identifying columns (like IDs) and the target columns.
# We only want the AI to look at the patterns, not memorize transaction IDs.
features_to_drop = [
    'TRANSACTION_ID', 'TX_DATETIME', 'CUSTOMER_ID', 
    'TERMINAL_ID', 'TX_FRAUD', 'TX_FRAUD_SCENARIO'
]

# Ensure we only drop columns that actually exist in the dataframe
features_to_drop = [col for col in features_to_drop if col in df.columns]

X = df.drop(columns=features_to_drop)
y = df['TX_FRAUD']

print(f"\nFeatures being used by the AI: {list(X.columns)}")

# --- 3. Split the Data ---
# 80% for training the AI, 20% for testing it on unseen data
print("\n2. Splitting data into Train and Test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 4. Handle Imbalance with SMOTE ---
# WARNING: We ONLY apply SMOTE to the training data. The test data must remain untouched 
# to simulate the real world where fraud is rare.
print("3. Applying SMOTE to balance the training data... (This takes about 1-2 minutes)")
start_time = time.time()
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"SMOTE finished in {time.time() - start_time:.1f} seconds.")
print(f"Old training size: {len(y_train)} | New training size: {len(y_train_smote)}")

# --- 5. Train the Random Forest Model ---
# We limit the trees (n_estimators=50) and depth (max_depth=15) so your computer doesn't freeze.
# n_jobs=-1 tells Python to use all available CPU cores.
print("\n4. Training the Random Forest AI... (This will take 2-4 minutes)")
start_time = time.time()
model = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train_smote, y_train_smote)
print(f"Training finished in {time.time() - start_time:.1f} seconds.")

# --- 6. Evaluate the Model ---
print("\n5. Testing the AI on unseen data...")
y_pred = model.predict(X_test)

print("\n--- CONFUSION MATRIX ---")
print("Top-Left: True Legitimate  | Top-Right: False Alarms (Said Fraud, but was Legit)")
print("Bottom-Left: Missed Frauds | Bottom-Right: Caught Frauds")
print(confusion_matrix(y_test, y_pred))

print("\n--- CLASSIFICATION REPORT ---")
# Precision: When it flags fraud, how often is it right?
# Recall: Out of all the actual frauds, how many did it catch?
print(classification_report(y_test, y_pred))

print("\nModel Trained!")