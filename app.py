"Churn prediction Api"
import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
MODEL_DIR = Path("model")
model = joblib.load(MODEL_DIR / "churn_model.joblib")          # loaded ONCE at startup
info = json.loads((MODEL_DIR / "model_info.json").read_text())
FEATURES, NUM_COLS, THRESHOLD = info["features"], info["num_cols"], info["threshold"]

# The allowed values for each category come from the trained model itself
encoder = model.named_steps["prep"].named_transformers_["cat"]
CAT_COLS = [c for c in FEATURES if c not in NUM_COLS]
OPTIONS = {col: [str(v) for v in cats] for col, cats in zip(CAT_COLS, encoder.categories_)}

app = Flask(__name__)


@app.get("/")
def form():
    return render_template("index.html", options=OPTIONS, num_cols=NUM_COLS)
def risk_level(p):
    if p>=0.6:
        return "high"
    if p >= THRESHOLD:
        return "medium"
    return "low"

@app.get("/health")
def health():
    return jsonify(status="ok", model=info["model"], threshold=THRESHOLD)

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify(error="Send a JSON body"), 400

    customers = payload if isinstance(payload, list) else [payload]   # one customer or a list

    for i, c in enumerate(customers):
        missing = [f for f in FEATURES if f not in c]
        if missing:
            return jsonify(error=f"Customer {i} is missing fields", missing=missing), 400

    X = pd.DataFrame(customers)[FEATURES]
    try:
        X[NUM_COLS] = X[NUM_COLS].astype(float)
    except ValueError:
        return jsonify(error=f"{NUM_COLS} must be numbers"), 400

    probs = model.predict_proba(X)[:, 1]
    results = [{"churn_probability": round(float(p), 3),
                "will_churn": bool(p >= THRESHOLD),
                "risk": risk_level(p)} for p in probs]
    return jsonify(results if isinstance(payload, list) else results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    