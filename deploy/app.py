"""
FastAPI server for the BUSI multi-task U-Net.

Endpoints:
    GET  /            -> HTML frontend (static/index.html)
    GET  /health      -> {"status": "ok"}
    POST /predict     -> multipart image upload -> JSON prediction

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 7860
"""

import os

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io

from inference import predict, load_model

app = FastAPI(title="Breast Ultrasound U-Net", version="1.0")

HERE = os.path.dirname(__file__)
STATIC_DIR = os.path.join(HERE, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def _warmup():
    # Load weights at boot so the first request isn't slow. If the weights file
    # is missing we don't crash the server — /predict will report it instead.
    try:
        load_model()
        print("Model loaded.")
    except Exception as e:
        print(f"WARNING: could not load model at startup: {e}")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw))
        result = predict(img)
        return JSONResponse(result)
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="best_model.pth not found. Add the trained weights.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
