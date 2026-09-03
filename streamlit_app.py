import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import joblib
import xgboost as xgb
import pandas as pd
import os

# ---------------- CONFIG ----------------
FEATURES = 8
SEQ_LEN = 24

FEATURE_NAMES = [
    "demand",
    "temperature",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos"
]

# ---------------- LOAD MODELS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(BASE_DIR, "backend", "assets")

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(FEATURES, 64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

@st.cache_resource
def load_models():
    lstm = LSTMModel()
    lstm.load_state_dict(torch.load(os.path.join(ASSETS, "lstm.pth"), map_location="cpu"))
    lstm.eval()

    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(os.path.join(ASSETS, "xgb.json"))

    scaler = joblib.load(os.path.join(ASSETS, "scaler.pkl"))

    return lstm, xgb_model, scaler

lstm, xgb_model, scaler = load_models()

# ---------------- UI ----------------
st.set_page_config(page_title="GridSage AI", layout="centered")
st.title("⚡ GridSage AI")
st.write("Hybrid LSTM + XGBoost Electricity Demand Forecast")

st.info("Enter the last 24-hour demand sequence")

# Simple input: repeat one row
base_row = []
cols = st.columns(2)
with cols[0]:
    base_row.append(st.number_input("Demand (MW)", 0.0, 10000.0, 4000.0))
    base_row.append(st.number_input("Temperature (°C)", -10.0, 50.0, 25.0))
    base_row.append(st.number_input("Hour sin", -1.0, 1.0, 0.0))
    base_row.append(st.number_input("Hour cos", -1.0, 1.0, 1.0))
with cols[1]:
    base_row.append(st.number_input("Day sin", -1.0, 1.0, 0.43))
    base_row.append(st.number_input("Day cos", -1.0, 1.0, -0.90))
    base_row.append(st.number_input("Month sin", -1.0, 1.0, 0.5))
    base_row.append(st.number_input("Month cos", -1.0, 1.0, 0.86))

sequence = np.array([base_row] * 24)

if st.button("Predict Demand"):
    df = pd.DataFrame(sequence, columns=FEATURE_NAMES)
    scaled = scaler.transform(df)
    X = torch.tensor(scaled).unsqueeze(0).float()

    with torch.no_grad():
        lstm_pred = lstm(X).numpy()

    flat = scaled.reshape(1, -1)
    xgb_input = np.concatenate([flat, lstm_pred], axis=1)
    correction = xgb_model.predict(xgb_input).reshape(-1, 1)

    hybrid = np.clip(lstm_pred + correction, -1, 1)

    dummy = np.zeros((1, FEATURES))
    dummy[0, 0] = hybrid[0, 0]

    final = scaler.inverse_transform(dummy)[0, 0]
    st.success(f"🔮 Predicted Demand: {final:.2f} MW")
