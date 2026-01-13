# Ensure the correct import path or install the necessary package
from fastapi import FastAPI
from .infer import router as infer_router

app = FastAPI()
app.include_router(infer_router)

# Define 'final_decision' before using it
final_decision = "Some decision or value" 

from backend.logging.loki_logger import send_to_loki
send_to_loki(final_decision)
