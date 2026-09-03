from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
import numpy as np
from hybrid_model import predict

app = FastAPI(title="GridSage AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InputData(BaseModel):
    sequence: list

@app.post("/predict")
def forecast(data: InputData):
    seq = np.array(data.sequence, dtype=np.float32)
    value = predict(seq)
    return {"predicted_demand_MW": round(value, 2)}

# For Render deployment
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
