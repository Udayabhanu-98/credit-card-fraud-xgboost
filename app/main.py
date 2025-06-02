from fastapi import FastAPI
import pandas as pd
import joblib
import traceback

# Initialize FastAPI app
app = FastAPI()

# Load the model
try:
    model = joblib.load("xgb_model.pkl")
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Model loading failed:")
    print(traceback.format_exc())
    model = None

# Define expected input order
expected_order = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                  'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                  'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

# Home route
@app.get("/")
def home():
    return {"message": "Credit Card Fraud Detection API is Live!"}

# Predict route
@app.post("/predict")
def predict(data: dict):
    try:
        if model is None:
            return {"error": "Model not loaded"}

        input_df = pd.DataFrame([data])
        input_df = input_df[expected_order]  # Enforce column order

        # Get fraud probability
        probability = model.predict_proba(input_df)[0][1]
        prediction = 1 if probability >= 0.6 else 0

        return {
            "prediction": prediction,
            "fraud_probability": round(float(probability), 4),
            "result": "Fraud" if prediction == 1 else "Not Fraud"
        }

    except Exception as e:
        print("🔥 Error during prediction:")
        print(traceback.format_exc())
        return {"error": str(e)}
