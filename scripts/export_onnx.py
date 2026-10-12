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
from scipy.stats import rankdata
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


class WithCAM(nn.Module):
    """Returns (logits, cam). All three heads are global-avg-pool -> linear, so Grad-CAM on the last conv block
    reduces exactly to CAM: sum_k W[c,k] * A_k (the 1/HW gradient factor cancels on normalisation)."""

    def __init__(self, m, arch):
        super().__init__()
        self.m, self.arch = m, arch
        self.fc = m.fc if arch == "resnet18" else m.classifier[1]

    def features(self, x):
        m = self.m
        if self.arch != "resnet18":
            return m.features(x)
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        return m.layer4(m.layer3(m.layer2(m.layer1(x))))

    def forward(self, x):
        a = self.features(x)
        logits = self.fc(torch.flatten(a.mean(dim=(2, 3)), 1))
        cam = torch.einsum("ck,bkhw->bchw", self.fc.weight, a)
        return logits, cam


def confusion(y, p, t):
    pred = p >= t
    return (pred & (y == 1)).sum(), (~pred & (y == 0)).sum(), (pred & (y == 0)).sum(), (~pred & (y == 1)).sum()


def threshold_for(y, p, target):
    """Highest threshold whose sensitivity on (y, p) is still >= target."""
    pos = np.sort(p[y == 1])
    return float(pos[int(np.floor((1 - target) * len(pos)))])


def auc(labels, scores):
    """Mann-Whitney AUC (ties averaged)."""
    ranks = rankdata(scores)
    pos = labels == 1
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "mobilenet_v2", "efficientnet_b0"])
    ap.add_argument("--out", help="default: backend/{arch}.onnx")
    ap.add_argument("--test-dir", help="ImageFolder-style dir with NORMAL/ and PNEUMONIA/")
    ap.add_argument("--target-sensitivity", type=float, default=0.95)
    ap.add_argument("--probs-csv", help="write file,label,p_pneumonia per test image")
    args = ap.parse_args()
    args.out = args.out or str(ROOT / "backend" / f"{args.arch}.onnx")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = state if isinstance(state, nn.Module) else build(args.arch)
    if not isinstance(state, nn.Module):
        model.load_state_dict(state)
    model.eval()

    x = torch.randn(1, 3, 224, 224)
    wrapped = WithCAM(model, args.arch).eval()
    ref_logits = model(x).detach().numpy()
    assert np.allclose(wrapped(x)[0].detach().numpy(), ref_logits, rtol=1e-4, atol=1e-3), "CAM wrapper changed logits"
    torch.onnx.export(wrapped, x, args.out, input_names=["input"], output_names=["output", "cam"],
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}, "cam": {0: "batch_size"}},
                      opset_version=17, dynamo=False)
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    ref = ref_logits
    diff = np.abs(sess.run(["output"], {"input": x.numpy()})[0] - ref).max() / max(1.0, np.abs(ref).max())
    assert diff < 1e-3, f"ONNX/PyTorch relative mismatch: {diff}"
    print(f"exported {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB), max rel diff = {diff:.2e}")

    if not args.test_dir:
        return
    labels, probs, names = [], [], []
    for label, cls in enumerate(["NORMAL", "PNEUMONIA"]):
        for f in sorted((Path(args.test_dir) / cls).iterdir()):
            logits = sess.run(["output"], {"input": preprocess(Image.open(f))})[0]
            probs.append(float(softmax(logits)[0][1]))
            labels.append(label)
            names.append(f.name)
    y, p = np.array(labels), np.array(probs)
    if args.probs_csv:
        rows = "".join(f"{n},{lb},{q:.6f}\n" for n, lb, q in zip(names, labels, probs))
        Path(args.probs_csv).write_text("file,label,p_pneumonia\n" + rows)
    tp, tn, fp, fn = confusion(y, p, 0.5)
    print(f"n={len(y)}  acc={(tp + tn) / len(y):.3f}  sensitivity={tp / (tp + fn):.3f}  "
          f"specificity={tn / (tn + fp):.3f}  f1={2 * tp / (2 * tp + fp + fn):.3f}  auc={auc(y, p):.3f}")
    print(f"confusion [[TN FP] [FN TP]] = [[{tn} {fp}] [{fn} {tp}]]")

    # Screening operating point: the highest threshold that keeps sensitivity >= target. Chosen on one half,
    # scored on the other (stratified 2-fold, 50 random splits) so the estimate isn't tuned on its own data.
    target, rng, sens, spec = args.target_sensitivity, np.random.default_rng(0), [], []
    for _ in range(50):
        fold = np.zeros(len(y), bool)
        for k in (0, 1):
            idx = np.flatnonzero(y == k)
            fold[rng.choice(idx, len(idx) // 2, replace=False)] = True
        for a, b in ((fold, ~fold), (~fold, fold)):
            t = threshold_for(y[a], p[a], target)
            tp, tn, fp, fn = confusion(y[b], p[b], t)
            sens.append(tp / (tp + fn))
            spec.append(tn / (tn + fp))
    t_all = threshold_for(y, p, target)
    print(f"operating point (target sensitivity {target:.2f}): threshold on all data = {t_all:.4f}; "
          f"cross-validated sensitivity = {np.mean(sens):.3f} ± {np.std(sens):.3f}, "
          f"specificity = {np.mean(spec):.3f} ± {np.std(spec):.3f}")


if __name__ == "__main__":
    main()
