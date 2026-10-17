"""Render the landing page's exploded-view layers for one test X-ray: the image, its lung outline and each served
model's Grad-CAM as transparent WebP plates (public/images/hero/).

    python scripts/make_hero_layers.py chest_xray/test/NORMAL/NORMAL2-IM-0325-0001.jpeg --mask-cache masks.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.ndimage import binary_erosion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import preprocess, softmax  # noqa: E402

W = 640  # plate width in px; height follows the X-ray's aspect ratio


def attention_rgba(cam, size):
    """Single warm ramp, alpha follows intensity, so low attention is see-through."""
    cam = np.maximum(cam, 0)
    cam = np.asarray(Image.fromarray((cam / (cam.max() + 1e-8) * 255).astype(np.uint8)).resize(size, Image.Resampling.BICUBIC),
                     dtype=np.float32) / 255
    lo, hi = np.array([242, 177, 52]), np.array([194, 54, 43])  # amber -> brick
    rgb = lo + (hi - lo) * cam[..., None]
    alpha = np.clip(cam ** 1.4 * 235, 0, 235)
    return Image.fromarray(np.dstack([rgb, alpha]).astype(np.uint8), "RGBA")


def lung_rgba(mask224, size):
    m = np.asarray(Image.fromarray(mask224.astype(np.uint8) * 255).resize(size, Image.Resampling.BILINEAR)) > 127
    edge = m & ~binary_erosion(m, iterations=3)
    out = np.zeros((*m.shape, 4), np.uint8)
    out[m] = (92, 194, 173, 40)       # faint fill
    out[edge] = (92, 194, 173, 255)   # outline
    return Image.fromarray(out, "RGBA")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--mask-cache", required=True, help="npz of 224x224 lung masks keyed by file name (seed_analysis.py)")
    ap.add_argument("--out", default=str(ROOT / "public" / "images" / "hero"))
    args = ap.parse_args()

    f = Path(args.image)
    img = Image.open(f).convert("L")
    size = (W, round(W * img.height / img.width))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    img.resize(size, Image.Resampling.LANCZOS).save(out / "xray.webp", quality=82)
    lung_rgba(np.load(args.mask_cache)[f.name], size).save(out / "lungs.webp", quality=90)
    x = preprocess(img)
    for name in ("resnet18", "mobilenet_v2"):
        logits, cam = ort.InferenceSession(str(ROOT / "backend" / f"{name}.onnx")).run(["output", "cam"], {"input": x})
        p = softmax(logits)[0]
        attention_rgba(cam[0, int(p.argmax())], size).save(out / f"cam_{name}.webp", quality=88)
        print(f"{name}: P(pneumonia) = {p[1]:.3f}")
    print(f"wrote {out} at {size[0]}x{size[1]}")


if __name__ == "__main__":
    main()
