"""Send test requests to the running API."""
import requests

URL = "http://localhost:5000"

risky = {
    "Tenure Months": 2, "Monthly Charges": 95.0,
    "Gender": "Female", "Senior Citizen": "No", "Partner": "No", "Dependents": "No",
    "Phone Service": "Yes", "Multiple Lines": "No", "Internet Service": "Fiber optic",
    "Online Security": "No", "Online Backup": "No", "Device Protection": "No",
    "Tech Support": "No", "Streaming TV": "Yes", "Streaming Movies": "Yes",
    "Contract": "Month-to-month", "Paperless Billing": "Yes", "Payment Method": "Electronic check",
}
loyal = {**risky, "Tenure Months": 60, "Monthly Charges": 55.0, "Internet Service": "DSL",
         "Online Security": "Yes", "Tech Support": "Yes", "Contract": "Two year",
         "Payment Method": "Credit card (automatic)", "Partner": "Yes", "Dependents": "Yes"}

print("health :", requests.get(f"{URL}/health").json())
print("risky  :", requests.post(f"{URL}/predict", json=risky).json())
print("loyal  :", requests.post(f"{URL}/predict", json=loyal).json())
print("batch  :", requests.post(f"{URL}/predict", json=[risky, loyal]).json())
bad = requests.post(f"{URL}/predict", json={"Tenure Months": 5})
print("bad    :", bad.status_code, bad.json())
