import streamlit as st
import pandas as pd
import numpy as np
import joblib # Used for loading tabular models

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="GuardianAI Fraud Engine", page_icon="🛡️", layout="wide")
CONFIDENCE_THRESHOLD = 0.60 # Lowered to account for Random Forest voting caps

# --- 2. LOAD THE AI MODEL ---
@st.cache_resource
def load_ai_model():
    # Update this filename to whatever you saved your Random Forest model as!
    # If you used pickle, it would be pickle.load(open('model.pkl', 'rb'))
    try:
        return joblib.load('fraud_model.pkl') 
    except FileNotFoundError:
        st.error("Model file not found! Please ensure your .pkl file is in the folder.")
        return None

model = load_ai_model()

# --- 3. THE WEB INTERFACE ---
st.title(" GuardianAI: Live Transaction Monitor")
st.write("Adjust the parameters below to simulate a real-time banking transaction security check.")
st.divider()

# Create a clean layout with two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Transaction Details")
    # Interactive sliders for the user to play with
    tx_amount = st.slider("Transaction Amount ($)", min_value=1.0, max_value=10000.0, value=250.0, step=10.0)
    tx_hour = st.slider("Time of Day (Hour: 0-23)", min_value=0, max_value=23, value=14, step=1)
    
    # Simulating your engineered features
    spending_ratio = st.slider("Spending Ratio vs Normal", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    distance_from_home = st.number_input("Distance from Home Address (Miles)", min_value=0, value=5)

with col2:
    st.subheader("Security Assessment")
    
    # Only run if the model successfully loaded
    if model is not None:
        if st.button(" Run Security Scan", type="primary", use_container_width=True):
            with st.spinner("Analyzing transaction patterns..."):
                
                # --- 4. FORMAT THE DATA FOR THE AI ---
                # We calculate a logical average amount based on their spending ratio
                calculated_avg_amount = tx_amount / spending_ratio if spending_ratio > 0 else tx_amount
                
                # The AI requires these exact 8 columns in this exact order to run.
                # We use your slider inputs for the main ones, and baseline averages for the rest.
                input_data = pd.DataFrame({
                    'TX_AMOUNT': [tx_amount],
                    'TX_TIME_SECONDS': [86400],        # Baseline dummy value
                    'TX_TIME_DAYS': [15],              # Baseline dummy value
                    'TX_HOUR': [tx_hour],
                    'TX_DAY_OF_WEEK': [3],             # Baseline: Wednesday
                    'CUST_AVG_AMOUNT': [calculated_avg_amount],
                    'CUST_SPENDING_RATIO': [spending_ratio],
                    'TERM_DAILY_TX_COUNT': [10]        # Baseline dummy value
                })
                
                # --- 5. PREDICTION LOGIC ---
                # We use predict_proba() to get the exact percentage of fraud likelihood
                probabilities = model.predict_proba(input_data)[0]
                fraud_probability = probabilities[1] # The probability of class '1' (Fraud)
                
                st.divider()
                
                # --- 6. DISPLAY RESULTS BASED ON YOUR THRESHOLD ---
                st.metric(label="Fraud Risk Score", value=f"{fraud_probability * 100:.2f}%")
                
                if fraud_probability >= CONFIDENCE_THRESHOLD:
                    st.error(" **FRAUD DETECTED: TRANSACTION BLOCKED**")
                    st.write(f"This transaction exceeded our {CONFIDENCE_THRESHOLD*100}% risk threshold.")
                    st.snow() # Fun visual effect for catching a criminal!
                else:
                    st.success(" **TRANSACTION APPROVED**")
                    st.write("Activity appears consistent with legitimate customer behavior.")