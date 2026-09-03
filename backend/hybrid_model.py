import torch
import torch.nn as nn
import numpy as np
import joblib
import xgboost as xgb
import pandas as pd


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

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(FEATURES, 64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

lstm = LSTMModel()
lstm.load_state_dict(torch.load("assets/lstm.pth", map_location="cpu"))
lstm.eval()

xgb_model = xgb.XGBRegressor()
xgb_model.load_model("assets/xgb.json")

scaler = joblib.load("assets/scaler.pkl")

def predict(sequence):
    scaled = scaler.transform(
        pd.DataFrame(sequence, columns=FEATURE_NAMES)
    )

    X = torch.tensor(scaled).unsqueeze(0).float()

    with torch.no_grad():
        lstm_pred = lstm(X).numpy()

    flat = scaled.reshape(1, -1)
    xgb_input = np.concatenate([flat, lstm_pred], axis=1)
    correction = xgb_model.predict(xgb_input).reshape(-1, 1)

    hybrid = np.clip(lstm_pred + correction, -1, 1)

    dummy = np.zeros((1, FEATURES))
    dummy[0, 0] = hybrid[0, 0]

    return float(scaler.inverse_transform(dummy)[0, 0])
