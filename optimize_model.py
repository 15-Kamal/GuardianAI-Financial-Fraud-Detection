import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# --- 1. Load Data & Split ---
print("1. Loading data & splitting...")
df = pd.read_pickle('engineered_transactions.pkl')

features_to_drop = ['TRANSACTION_ID', 'TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_FRAUD', 'TX_FRAUD_SCENARIO']
features_to_drop = [col for col in features_to_drop if col in df.columns]

X = df.drop(columns=features_to_drop)
y = df['TX_FRAUD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 2. Train with SMOTE ---
print("2. Applying SMOTE & Training Model... (Please wait ~3 minutes)")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

model = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train_smote, y_train_smote)

# --- 3. The Magic: Custom Decision Threshold ---
print("\n3. Testing Custom Decision Threshold...")

# Get the raw probability scores (0.0 to 1.0) instead of just 0 or 1
probabilities = model.predict_proba(X_test)[:, 1]

# We are raising the bar: The AI must be 85% sure to flag a fraud
CUSTOM_THRESHOLD = 0.85
y_pred_custom = (probabilities >= CUSTOM_THRESHOLD).astype(int)

print(f"\n--- CONFUSION MATRIX (Threshold: {CUSTOM_THRESHOLD}) ---")
print("Top-Left: True Legitimate  | Top-Right: False Alarms")
print("Bottom-Left: Missed Frauds | Bottom-Right: Caught Frauds")
print(confusion_matrix(y_test, y_pred_custom))

print(f"\n--- CLASSIFICATION REPORT (Threshold: {CUSTOM_THRESHOLD}) ---")
print(classification_report(y_test, y_pred_custom))

print("\nOptimization Complete!") 