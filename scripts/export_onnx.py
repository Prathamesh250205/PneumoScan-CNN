"""Export a trained checkpoint from notebooks/chest-x-ray.ipynb to backend/model.onnx and evaluate it
through the exact preprocessing used by the API.

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    python scripts/export_onnx.py resnet18_best.pth --test-dir chest_xray/test
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torch import nn
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import preprocess, softmax  # noqa: E402  (serving pipeline is the source of truth)


def build(arch):
    m = getattr(models, arch)(weights=None)
    if arch == "resnet18":
        m.fc = nn.Linear(m.fc.in_features, 2)
    else:
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    return m


def auc(labels, scores):
    """Mann-Whitney AUC (ties averaged)."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    for s in np.unique(scores):
        ranks[scores == s] = ranks[scores == s].mean()
    pos = labels == 1
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "mobilenet_v2", "efficientnet_b0"])
    ap.add_argument("--out", default=str(ROOT / "backend" / "model.onnx"))
    ap.add_argument("--test-dir", help="ImageFolder-style dir with NORMAL/ and PNEUMONIA/")
    args = ap.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = state if isinstance(state, nn.Module) else build(args.arch)
    if not isinstance(state, nn.Module):
        model.load_state_dict(state)
    model.eval()

    x = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, x, args.out, input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                      opset_version=17, dynamo=False)
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    ref = model(x).detach().numpy()
    diff = np.abs(sess.run(None, {"input": x.numpy()})[0] - ref).max() / max(1.0, np.abs(ref).max())
    assert diff < 1e-3, f"ONNX/PyTorch relative mismatch: {diff}"
    print(f"exported {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB), max rel diff = {diff:.2e}")

    if not args.test_dir:
        return
    labels, probs = [], []
    for label, cls in enumerate(["NORMAL", "PNEUMONIA"]):
        for f in sorted((Path(args.test_dir) / cls).iterdir()):
            logits = sess.run(None, {"input": preprocess(Image.open(f))})[0]
            probs.append(float(softmax(logits)[0][1]))
            labels.append(label)
    y, p = np.array(labels), np.array(probs)
    pred = (p >= 0.5).astype(int)
    tp, tn = ((pred == 1) & (y == 1)).sum(), ((pred == 0) & (y == 0)).sum()
    fp, fn = ((pred == 1) & (y == 0)).sum(), ((pred == 0) & (y == 1)).sum()
    print(f"n={len(y)}  acc={(tp + tn) / len(y):.3f}  sensitivity={tp / (tp + fn):.3f}  "
          f"specificity={tn / (tn + fp):.3f}  f1={2 * tp / (2 * tp + fp + fn):.3f}  auc={auc(y, p):.3f}")
    print(f"confusion [[TN FP] [FN TP]] = [[{tn} {fp}] [{fn} {tp}]]")


if __name__ == "__main__":
    main()
