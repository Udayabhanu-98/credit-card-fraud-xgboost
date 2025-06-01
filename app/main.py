from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()
model = joblib.load("xgb_model.pkl")

@app.get("/")
def home():
    return {"message": "Fraud Detection API running!"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    result = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    return {"prediction": int(result), "probability": round(prob, 4)}
