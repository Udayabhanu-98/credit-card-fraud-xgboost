# 🛡️ Credit Card Fraud Detection – End-to-End ML Pipeline (XGBoost + AWS + FastAPI)

This project implements an end-to-end pipeline for detecting credit card fraud using real-world data from Kaggle. The solution leverages AWS services for data processing and training, and exposes the trained model through a FastAPI-based REST API deployed on Render.

---

## 📂 Dataset

- Source: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Real-world European cardholder transactions
- Highly imbalanced (only ~0.172% fraud)

---

## 🛠️ Technologies Used

| Component        | Technology                         |
|------------------|-------------------------------------|
| Data Storage     | AWS S3                              |
| Schema Detection | AWS Glue Crawler                    |
| Model Training   | Amazon SageMaker (Jupyter Notebook) |
| Class Balancing  | SMOTE                               |
| Model Deployment | FastAPI + Render                    |
| Model Registry   | Joblib (`xgb_model.pkl`)            |
| Code Hosting     | GitHub                              |

---

## 🧪 Models Trained & Evaluated

| Model               | Recall | Precision | F1-Score | AUC-PR Score |
|---------------------|--------|-----------|----------|--------------|
| Logistic Regression | 0.87   | 0.05      | 0.10     | 0.71         |
| Decision Tree       | 0.74   | 0.46      | 0.57     | 0.60         |
| Random Forest       | 0.82   | 0.59      | 0.69     | 0.76         |
| **XGBoost**         | 0.80   | **0.75**  | **0.77** | **0.8140**     |
| MLP (Keras)         | 0.82   | 0.60      | 0.70     | 0.8159       |
| TabNet              | 0.81   | 0.33      | 0.47     | 0.7477       |

✅ **Final Model Chosen**: **XGBoost** (best overall precision, F1, AUC-PR)

---

## 🔍 Evaluation Focus

- ⚠️ Accuracy **not used** due to class imbalance.
- 📈 Metrics used:
  - **Recall** – catch as many frauds as possible
  - **Precision** – minimize false alarms
  - **F1-Score** – balance between recall & precision
  - **AUC-PR** – best metric for imbalanced problems

---

## 🧩 Model Training Pipeline (SageMaker)

1. Ingested CSV from S3
2. Applied Glue Crawler to detect schema
3. Explored and cleaned data in SageMaker Notebook
4. Split data and applied SMOTE for balancing
5. Trained multiple models with hyperparameter tuning
6. Chose XGBoost based on production-relevant metrics
7. Saved final model using `joblib`

---

## 🚀 FastAPI Deployment (Render.com)

- **API Framework**: FastAPI
- **Model**: `xgb_model.pkl`
- **Deployment**: Render (free tier)
- **Swagger UI**: `/docs` for testing requests

### ✅ Sample POST Request

```json
{
  "Time": 4462.0,
  "V1": -2.303349568,
  "V2": 1.75924746,
  "V3": -0.359744743,
  "V4": 2.330243051,
  "V5": -0.821628328,
  "V6": -0.075787571,
  "V7": 0.562319782,
  "V8": -0.399146578,
  "V9": -0.238253368,
  "V10": -1.525411627,
  "V11": 2.032912158,
  "V12": -6.560124295,
  "V13": 0.022937323,
  "V14": -1.470101536,
  "V15": -0.698826069,
  "V16": -2.282193829,
  "V17": -4.781830856,
  "V18": -2.615664945,
  "V19": -1.334441067,
  "V20": -0.430021867,
  "V21": -0.294166318,
  "V22": -0.932391057,
  "V23": 0.172726296,
  "V24": -0.087329538,
  "V25": -0.156114265,
  "V26": -0.542627889,
  "V27": 0.039565989,
  "V28": -0.153028797,
  "Amount": 239.93
}
