"""
api.py — Flask server for Cloud Removal
Receives a cloudy optical image + SAR image via POST /restore
Returns the restored (cloud-free) image as PNG bytes

Run with:
    python api.py
"""

import io
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import torchvision.transforms as transforms

from src.model import MultiModalUNet

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_PATH  = "best_model.pth"
TARGET_SIZE = (256, 256)
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# LOAD MODEL (once at startup)
# --------------------------------------------------
print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
model = MultiModalUNet(n_channels=5, n_classes=3).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()
print("Model ready.")

# --------------------------------------------------
# APP
# --------------------------------------------------
app = Flask(__name__)
CORS(app)  # Allow frontend (any origin) to call this API

to_tensor = transforms.ToTensor()


@app.route("/restore", methods=["POST"])
def restore():
    """
    Accepts:
      - cloudy: the cloudy optical satellite image (form-data)
      - sar:    the corresponding SAR (radar) image (form-data)
    Returns the cloud-removed optical image as PNG.
    """

    # --- Validate inputs ---
    if "cloudy" not in request.files or "sar" not in request.files:
        return jsonify({"error": "Both 'cloudy' and 'sar' image files are required."}), 400

    cloudy_file = request.files["cloudy"]
    sar_file    = request.files["sar"]

    # --- Read & resize ---
    cloudy_pil = Image.open(cloudy_file.stream).convert("RGB").resize(TARGET_SIZE, Image.BILINEAR)
    sar_pil    = Image.open(sar_file.stream).convert("RGB").resize(TARGET_SIZE, Image.BILINEAR)

    optical_tensor  = to_tensor(cloudy_pil)           # [3, 256, 256]
    sar_tensor_full = to_tensor(sar_pil)              # [3, 256, 256]
    sar_tensor      = sar_tensor_full[:2, :, :]       # [2, 256, 256] — VV/VH

    # --- Inference ---
    cloudy_t = optical_tensor.unsqueeze(0).to(DEVICE) # [1, 3, 256, 256]
    sar_t    = sar_tensor.unsqueeze(0).to(DEVICE)     # [1, 2, 256, 256]

    with torch.no_grad():
        restored = model(sar_t, cloudy_t)
        restored = torch.clamp(restored, 0.0, 1.0)

    # --- Postprocess ---
    out_np  = restored[0].cpu().permute(1, 2, 0).numpy()
    out_np  = (out_np * 255).clip(0, 255).astype(np.uint8)
    out_pil = Image.fromarray(out_np, "RGB")

    buf = io.BytesIO()
    out_pil.save(buf, format="PNG")
    buf.seek(0)

    return send_file(buf, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
