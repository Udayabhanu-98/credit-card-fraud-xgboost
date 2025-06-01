from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

# Load the trained XGBoost model
model = joblib.load("xgb_model.pkl")

@app.get("/")
def root():
    return {"message": "Credit Card Fraud Detection API is Live!"}

@app.post("/predict")
def predict(data: dict):
    try:
        # Convert input dict to DataFrame
        input_df = pd.DataFrame([data])

        # Get prediction and probability
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # Convert numeric prediction to label
        label = "Fraud" if prediction == 1 else "Not Fraud"

        return {
            "prediction": int(prediction),
            "fraud_probability": round(probability, 4),
            "result": label
        }

    except Exception as e:
        return {"error": str(e)}
