import joblib
import numpy as np
from flask import Flask, request, jsonify

# Load model once
model = joblib.load('model.pkl')

app = Flask(__name__)

# ✅ FINAL feature schema (MUST match training EXACTLY)
FEATURE_NAMES = [
    'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
    'V11','V12','V13','V14','V15','V16','V17','V18',
    'V19','V20','V21','V22','V23','V24','V25','V26',
    'V27','V28','Hour','Amount_scaled','Time_scaled'
]

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'RandomForest'})

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    data = request.get_json()

    # ✅ Validate fields
    missing = [f for f in FEATURE_NAMES if f not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    # ✅ Convert to float
    try:
        features = np.array([[float(data[f]) for f in FEATURE_NAMES]])
    except (ValueError, TypeError) as e:
        return jsonify({'error': f'All fields must be numeric. Detail: {str(e)}'}), 400

    # ✅ NO SCALING HERE (already scaled in input)

    # ✅ Prediction
    fraud_probability = model.predict_proba(features)[0][1]
    is_fraud = bool(fraud_probability >= 0.75)

    return jsonify({
        'fraud_probability': round(float(fraud_probability), 4),
        'is_fraud': is_fraud,
        'threshold_used': 0.75,
        'model': 'RandomForest'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)