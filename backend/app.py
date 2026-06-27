# backend/app.py — GuardianAI (Single URL: Flask serves React + API)

from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import joblib
import os

# ── Point Flask at the React production build ─────────────────
BASE_DIR    = os.path.dirname(__file__)
BUILD_DIR   = os.path.join(BASE_DIR, '..', 'frontend', 'build')
MODEL_PATH  = os.path.join(BASE_DIR, '..', 'fraud_model.pkl')

app = Flask(__name__, static_folder=BUILD_DIR, static_url_path='')

# ── Load model ────────────────────────────────────────────────
print("Loading GuardianAI model...")
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: fraud_model.pkl not found at {MODEL_PATH}")
    model = None

THRESHOLD = 0.60


# ── Serve React frontend ───────────────────────────────────────
@app.route('/')
def serve_react():
    return send_from_directory(app.static_folder, 'index.html')

@app.errorhandler(404)
def not_found(e):
    # Let React Router handle unknown paths
    return send_from_directory(app.static_folder, 'index.html')


# ── Health check ──────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "threshold": THRESHOLD
    })


# ── Prediction endpoint ───────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    try:
        tx_amount       = float(data.get('tx_amount', 0))
        tx_hour         = int(data.get('tx_hour', 12))
        tx_day          = int(data.get('tx_day', 3))
        cust_avg_amount = float(data.get('cust_avg_amount', max(tx_amount, 1)))
        term_count      = int(data.get('term_count', 10))
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400

    spending_ratio = tx_amount / cust_avg_amount if cust_avg_amount > 0 else 1.0

    input_df = pd.DataFrame({
        'TX_AMOUNT':           [tx_amount],
        'TX_TIME_SECONDS':     [86400],
        'TX_TIME_DAYS':        [15],
        'TX_HOUR':             [tx_hour],
        'TX_DAY_OF_WEEK':      [tx_day],
        'CUST_AVG_AMOUNT':     [cust_avg_amount],
        'CUST_SPENDING_RATIO': [spending_ratio],
        'TERM_DAILY_TX_COUNT': [term_count],
    })

    probabilities = model.predict_proba(input_df)[0]
    fraud_prob    = float(probabilities[1])
    legit_prob    = float(probabilities[0])

    distance = abs(fraud_prob - THRESHOLD)
    confidence = "HIGH" if distance >= 0.3 else "MEDIUM" if distance >= 0.15 else "LOW"

    risk_factors = []

    if spending_ratio > 3.0:
        risk_factors.append({"factor": "Extreme spending ratio", "severity": "HIGH",
            "detail": f"Transaction is {spending_ratio:.1f}x the customer's normal spend — strong fraud signal."})
    elif spending_ratio > 1.5:
        risk_factors.append({"factor": "Elevated spending ratio", "severity": "MEDIUM",
            "detail": f"Transaction is {spending_ratio:.1f}x above the customer's average."})
    else:
        risk_factors.append({"factor": "Normal spending ratio", "severity": "LOW",
            "detail": f"Amount is close to the customer's historical average ({spending_ratio:.2f}x)."})

    if tx_hour in [1, 2, 3, 4, 5]:
        risk_factors.append({"factor": "Unusual transaction hour", "severity": "MEDIUM",
            "detail": f"Transaction at {tx_hour:02d}:00 — late-night activity correlates with fraud."})
    elif 9 <= tx_hour <= 18:
        risk_factors.append({"factor": "Business hours transaction", "severity": "LOW",
            "detail": f"Transaction at {tx_hour:02d}:00 falls within normal business hours."})

    if term_count > 100:
        risk_factors.append({"factor": "Critical terminal activity", "severity": "HIGH",
            "detail": f"Terminal used {term_count} times today — likely compromised (skimming)."})
    elif term_count > 50:
        risk_factors.append({"factor": "High terminal usage", "severity": "MEDIUM",
            "detail": f"Terminal used {term_count} times today — above expected volume."})

    if tx_amount > 10000:
        risk_factors.append({"factor": "Very large transaction", "severity": "HIGH",
            "detail": f"${tx_amount:,.0f} is an unusually large single transaction."})
    elif tx_amount > 5000:
        risk_factors.append({"factor": "Large transaction amount", "severity": "MEDIUM",
            "detail": f"${tx_amount:,.0f} exceeds typical transaction thresholds."})

    return jsonify({
        "fraud_probability":      round(fraud_prob, 4),
        "legitimate_probability": round(legit_prob, 4),
        "verdict":                "BLOCKED" if fraud_prob >= THRESHOLD else "APPROVED",
        "confidence":             confidence,
        "threshold":              THRESHOLD,
        "spending_ratio":         round(spending_ratio, 4),
        "risk_factors":           risk_factors[:4],
    })


if __name__ == '__main__':
    print("GuardianAI running on http://localhost:5000")
    app.run(debug=True, port=5000, host='0.0.0.0')