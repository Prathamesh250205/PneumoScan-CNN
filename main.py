import base64
import io
import logging
import os

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="PneumoScan", description="Pediatric chest X-ray pneumonia screening demo (ResNet18 vs MobileNetV2).")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
PUBLIC_DIR = os.path.join(BASE_DIR, "public")
MODELS = ["resnet18", "mobilenet_v2"]  # backend/{name}.onnx, exported by scripts/export_onnx.py (outputs: logits, cam)

sessions = {}
model_error = None


def load_models():
    global model_error
    try:
        for name in MODELS:
            if name not in sessions:
                sessions[name] = ort.InferenceSession(os.path.join(BACKEND_DIR, f"{name}.onnx"), providers=["CPUExecutionProvider"])
        model_error = None
    except Exception as e:  # surfaced by /health
        model_error = str(e)
        logger.error(f"Model load failed: {e}")
    return len(sessions) == len(MODELS)


@app.get("/health")
def health_check():
    loaded = load_models()
    return {"status": "healthy" if loaded else "degraded", "model_loaded": loaded, "models": list(sessions), "model_error": model_error}


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def preprocess(image):
    """Same as the training eval transform: grayscale -> 3ch -> 224x224 bilinear -> [0,1] -> ImageNet norm -> NCHW."""
    img = image.convert("L").convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    return ((x - mean) / std)[None].astype(np.float32)


def jet(v):
    """v in [0,1] -> RGB uint8, classic Grad-CAM colormap."""
    rgb = np.stack([np.clip(1.5 - np.abs(4 * v - k), 0, 1) for k in (3, 2, 1)], axis=-1)
    return (rgb * 255).astype(np.uint8)


def overlay(gray, cam):
    """gray: PIL 'L' image (display size); cam: (7,7) non-negative. Returns a JPEG data URL."""
    cam = cam / (cam.max() + 1e-8)
    heat = Image.fromarray((cam * 255).astype(np.uint8)).resize(gray.size, Image.Resampling.BICUBIC)
    heat = jet(np.asarray(heat, dtype=np.float32) / 255)
    base = np.asarray(gray.convert("RGB"), dtype=np.float32)
    out = Image.fromarray((0.55 * base + 0.45 * heat).astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Pneumonia probability from both served models, plus each model's Grad-CAM overlay for its predicted class."""
    if not load_models():
        raise HTTPException(status_code=503, detail=f"Models not loaded: {model_error}")
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an image file.")
    try:
        image = Image.open(file.file)
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Could not decode the image.")

    x = preprocess(image)
    display = image.convert("L")
    display.thumbnail((448, 448))
    results = {}
    for name, sess in sessions.items():
        logits, cam = sess.run(["output", "cam"], {"input": x})
        p = softmax(logits)[0]
        pred = int(p.argmax())
        results[name] = {"pneumonia_probability": float(p[1]), "heatmap": overlay(display, np.maximum(cam[0, pred], 0))}

    primary = results["resnet18"]["pneumonia_probability"]
    return {"normal_probability": 1 - primary, "pneumonia_probability": primary, "models": results}


# Local dev: serve the frontend. On Vercel, public/ is served by the CDN and this dir is not in the function bundle.
if os.path.isdir(PUBLIC_DIR):
    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")
