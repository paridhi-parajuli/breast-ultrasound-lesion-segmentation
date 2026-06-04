---
title: Breast Ultrasound Lesion Segmentation U-Net
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Breast Ultrasound U-Net — Lesion Segmentation

ResNet34 U-Net that, given a breast-ultrasound image, returns a **lesion
segmentation mask** with lesion area and confidence stats.

Build this repo as a Docker Space and serve on port 7860.

---

## Files

| File | Purpose |
|------|---------|
| `app.py` | FastAPI server (`/`, `/health`, `/predict`) |
| `inference.py` | preprocessing + prediction (mirrors training preprocessing) |
| `model_def.py` | rebuilds the ResNet34 U-Net (mirrors the model in model.py) |
| `static/index.html` | the web frontend |
| `pyproject.toml` | dependencies, managed by **uv** |
| `Dockerfile` | container build (uses uv) |
| `best_model.pth` | **trained weights — added after training using model.py in Google Colab T4 GPU** |

---

## Run locally with uv

```bash
uv sync                 # creates .venv and installs everything from pyproject.toml
uv run uvicorn app:app --host 0.0.0.0 --port 7860
# open http://localhost:7860
```


## API

```bash
curl -X POST http://localhost:7860/predict \
  -F "file=@some_ultrasound.png"
```

Returns:
```json
{
  "lesion_detected": true,
  "lesion_area_percent": 4.7,
  "mask_confidence": 0.86,
  "mask_overlay_png": "data:image/png;base64,..."
}
```

Interactive API docs (Swagger UI): http://localhost:7860/docs
