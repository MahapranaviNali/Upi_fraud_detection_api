# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = FastAPI()

# Load your models and scaler
model = load_model("autoencoder_model.h5", compile=False)
scaler = joblib.load("scaler.pkl")

class Transaction(BaseModel):
    sender_upi: str
    receiver_upi: str
    amount: float

@app.post("/predict")
def predict(txn: Transaction):
    sender_encoded = abs(hash(txn.sender_upi)) % (10**6)
    receiver_encoded = abs(hash(txn.receiver_upi)) % (10**6)

    features = np.array([[sender_encoded, receiver_encoded, txn.amount]])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)
    mse = np.mean(np.power(scaled - prediction, 2), axis=1)
    threshold = 0.95 * np.max(mse)
    result = "FRAUD" if mse[0] > threshold else "LEGIT"

    return {
        "prediction": result,
        "reconstruction_error": float(mse[0]),
        "threshold": float(threshold)
    }
