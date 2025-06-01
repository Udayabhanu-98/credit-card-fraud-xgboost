from fastapi import FastAPI
import pandas as pd
import joblib
import traceback

# Initialize FastAPI app
app = FastAPI()

# Load your trained model
try:
    model = joblib.load("xgb_model.pkl")
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Failed to load model:")
    print(traceback.format_exc())
    model = None  # Prevent crashes if model not found

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

        # Convert input to DataFrame
        input_df = pd.DataFrame([data])

        # Prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        label = "Fraud" if prediction == 1 else "Not Fraud"

        return {
            "prediction": int(prediction),
            "fraud_probability": round(probability, 4),
            "result": label
        }

    except Exception as e:
        print("🔥 Error during prediction:")
        print(traceback.format_exc())
        return {"error": str(e)}
