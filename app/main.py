import traceback  # Add this at the top

@app.post("/predict")
def predict(data: dict):
    try:
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        label = "Fraud" if prediction == 1 else "Not Fraud"

        return {
            "prediction": int(prediction),
            "fraud_probability": round(probability, 4),
            "result": label
        }

    except Exception as e:
        print("🔥 Prediction error:", traceback.format_exc())  # <-- real debug print
        return {"error": str(e)}
