import torch
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

checkpoint = torch.load("ml/kavach_lstm_v2.pth", map_location="cpu")
model = checkpoint["model"]
label_encoder = checkpoint["label_encoder"]

class LogSequence(BaseModel):
    sequence: list  # shape: [timesteps, features]

@router.post("/infer")
def infer_attack(data: LogSequence):
    seq = np.array(data.sequence)
    with torch.no_grad():
        out = model(torch.tensor(seq, dtype=torch.float32).unsqueeze(0))
        prob = torch.softmax(out, dim=1)
        idx = torch.argmax(prob).item()

    return {
        "prediction": label_encoder.inverse_transform([idx])[0],
        "confidence": round(float(prob[0][idx]), 3)
    }
