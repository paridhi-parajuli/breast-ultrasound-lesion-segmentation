# BUSI Lesion Segmentation

Breast-ultrasound lesion segmentation on the [BUSI dataset](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)
using a ResNet34 U-Net (PyTorch + segmentation-models-pytorch), served as a
FastAPI app with a web frontend.

> ⚠️ Research/education demo only. **Not a medical device, not for diagnosis.**

## Layout

| Path | Purpose |
|------|---------|
| `model.py` | training script (run in Colab; saves `best_model.pth`) |
| `best_model.pth` | trained weights (Git LFS) |
| `deploy/` | FastAPI server + frontend + Dockerfile — see [deploy/README.md](deploy/README.md) |

## Model

- ResNet34 encoder (ImageNet-pretrained) + U-Net decoder, 1-channel input, 256×256
- Loss: 0.3·BCE + 0.7·Dice, AdamW, ReduceLROnPlateau on val Dice
- Train-only augmentation: h/v flips + ±15° rotation (applied jointly to image & mask)
- Early stopping on val Dice; best checkpoint saved as `best_model.pth`

## Quick start (serving)

```bash
cd deploy
MODEL_PATH=../best_model.pth uv run uvicorn app:app --port 7860
# open http://localhost:7860  (frontend)  ·  /docs (Swagger)
```
