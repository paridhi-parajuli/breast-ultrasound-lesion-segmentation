"""
Inference utilities: load the trained model once, run a prediction on an image.

The preprocessing here MUST match training exactly:
    grayscale -> resize 256 -> ToTensor [0,1] -> normalize(mean=0.5, std=0.5)
"""

import io
import os
import base64

import numpy as np
import torch
from PIL import Image

from model_def import build_model, IMG_SIZE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL = None


def load_model(weights_path=None):
    """Load weights once and cache. Returns the model in eval mode.

    Resolution order: explicit arg -> $MODEL_PATH -> ./best_model.pth.
    """
    global _MODEL
    if _MODEL is None:
        if weights_path is None:
            weights_path = os.environ.get("MODEL_PATH", "best_model.pth")
        model = build_model()
        state = torch.load(weights_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.to(DEVICE).eval()
        _MODEL = model
    return _MODEL


def preprocess(pil_img):
    """PIL image -> normalized [1,1,256,256] tensor (matches training)."""
    img = pil_img.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0      # [0,1]
    arr = (arr - 0.5) / 0.5                               # normalize -> ~[-1,1]
    tensor = torch.from_numpy(arr)[None, None]            # [1,1,H,W]
    return tensor.to(DEVICE)


def _mask_to_png_base64(mask_bool, size):
    """Binary mask -> red transparent overlay PNG, base64 data URI."""
    h, w = mask_bool.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = 255                                    # red channel
    rgba[..., 3] = (mask_bool * 140).astype(np.uint8)     # alpha where lesion
    overlay = Image.fromarray(rgba, mode="RGBA").resize(size, Image.NEAREST)
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@torch.no_grad()
def predict(pil_img, thresh=0.5):
    """Returns dict with segmentation overlay + lesion stats.

    The deployed checkpoint is segmentation-only (the classification head was
    disabled during training), so there is no class prediction here.
    """
    model = load_model()
    x = preprocess(pil_img)

    seg_logits = model(x)

    seg_prob = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()
    mask = (seg_prob > thresh)
    lesion_pct = round(float(mask.mean()) * 100.0, 2)
    lesion_detected = bool(mask.any())

    # mean probability inside the predicted lesion = rough confidence signal
    mask_confidence = round(float(seg_prob[mask].mean()), 4) if lesion_detected else 0.0

    overlay = _mask_to_png_base64(mask, size=pil_img.size)

    return {
        "lesion_detected": lesion_detected,
        "lesion_area_percent": lesion_pct,
        "mask_confidence": mask_confidence,
        "mask_overlay_png": overlay,
    }


if __name__ == "__main__":
    # Quick smoke test:  python inference.py path/to/image.png
    import sys
    import json

    img = Image.open(sys.argv[1])
    result = predict(img)
    result["mask_overlay_png"] = result["mask_overlay_png"][:60] + "..."
    print(json.dumps(result, indent=2))
