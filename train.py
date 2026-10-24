"""Train the churn model and save it to model/ (run once before starting the API)."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/Telco_customer_churn.xlsx")
MODEL_DIR = Path("model")
THRESHOLD = 0.4          # your Day 2 business decision (catch more churners)

NUM_COLS = ["Tenure Months", "Monthly Charges"]
CAT_COLS = ["Gender", "Senior Citizen", "Partner", "Dependents", "Phone Service",
            "Multiple Lines", "Internet Service", "Online Security", "Online Backup",
            "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies",
            "Contract", "Paperless Billing", "Payment Method"]
FEATURES = NUM_COLS + CAT_COLS

df = pd.read_excel(DATA_PATH)
X = df[FEATURES]                 # leakage columns + Total Charges are never selected
y = df["Churn Value"]

# same split as Days 2-3 → same locked test set
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
X_dev, y_dev = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])

pipeline = Pipeline([
    ("prep", ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])),
    ("model", LogisticRegression(max_iter=1000)),
])
pipeline.fit(X_dev, y_dev)

p = pipeline.predict_proba(X_test)[:, 1]
pred = (p >= THRESHOLD).astype(int)
metrics = {"accuracy": round(accuracy_score(y_test, pred), 3),
           "precision": round(precision_score(y_test, pred), 3),
           "recall": round(recall_score(y_test, pred), 3),
           "f1": round(f1_score(y_test, pred), 3)}
print("Test metrics @ threshold", THRESHOLD, metrics)

MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(pipeline, MODEL_DIR / "churn_model.joblib")
with open(MODEL_DIR / "model_info.json", "w") as f:
    json.dump({"threshold": THRESHOLD, "features": FEATURES, "num_cols": NUM_COLS,
               "test_metrics": metrics, "model": "LogisticRegression"}, f, indent=2)
print("Saved to", MODEL_DIR.resolve())