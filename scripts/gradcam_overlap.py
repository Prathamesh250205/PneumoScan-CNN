"""Quantify where each CNN's Grad-CAM attention falls, using anatomical masks from
torchxrayvision's ChestX-Det PSPNet (lungs, clavicles, scapulae).

    pip install torch torchvision torchxrayvision matplotlib --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
    python scripts/gradcam_overlap.py --weights-dir ~/Downloads --test-dir chest_xray/test

For each test image and model: Grad-CAM for the predicted class on the last conv block, then the share of
CAM energy inside the lung mask vs. the shoulder girdle (clavicles + scapulae), and whether the CAM peak lies
in the lungs. Appends to <out>/gradcam_overlap.csv (re-running resumes) and prints a summary with bootstrap 95% CIs.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from main import preprocess  # noqa: E402
from export_onnx import build  # noqa: E402

ARCHS = ["resnet18", "mobilenet_v2", "efficientnet_b0"]
CLASSES = ["NORMAL", "PNEUMONIA"]


def target_layer(model, arch):
    return model.layer4 if arch == "resnet18" else model.features[-1]


def gradcam(model, layer, x):
    store = {}

    def hook(_, __, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    h = layer.register_forward_hook(hook)
    logits = model(x)
    h.remove()
    pred = int(logits.argmax(1))
    model.zero_grad()
    logits[0, pred].backward()
    weights = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * store["act"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    return cam.detach().numpy(), float(torch.softmax(logits, 1)[0, 1])


def segment(seg, gray224):
    """gray224: float array in [0,1]. Returns {name: bool mask 224x224}."""
    x = torch.from_numpy(gray224 * 2048 - 1024).float()[None, None]
    x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
    with torch.no_grad():
        prob = torch.sigmoid(seg(x))
    prob = F.interpolate(prob, size=(224, 224), mode="area")[0].numpy()
    return {t: prob[i] > 0.5 for i, t in enumerate(seg.targets)}


def ci(v, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, (n, len(v))).mean(1)
    return v.mean(), *np.percentile(boots, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", required=True, help="dir with {arch}_best.zip / .pth checkpoints")
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--limit", type=int, default=0, help="images per class (0 = all)")
    args = ap.parse_args()

    import torchxrayvision as xrv
    seg = xrv.baseline_models.chestx_det.PSPNet().eval()
    torch.set_grad_enabled(True)

    models = {}
    for arch in ARCHS:
        ckpt = next(p for p in Path(args.weights_dir).glob(f"{arch}_best*") if p.suffix in (".zip", ".pth", ".pt"))
        m = build(arch)
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        for mod in m.modules():  # in-place activations break tensor grad hooks
            if hasattr(mod, "inplace"):
                mod.inplace = False
        models[arch] = m.eval()

    files = []
    for label, cls in enumerate(CLASSES):
        fs = sorted((Path(args.test_dir) / cls).iterdir())
        files += [(f, label) for f in (fs[: args.limit] if args.limit else fs)]

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    csv_path = out / "gradcam_overlap.csv"
    fields = ["file", "label", "model", "p_pneumonia", "correct", "lung_energy", "lung_energy_dilated",
              "girdle_energy", "lung_area", "girdle_area", "peak_in_lung"]
    done = {r["file"] for r in csv.DictReader(open(csv_path))} if csv_path.exists() else set()  # resume
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if not done:
            w.writeheader()
        for i, (f, label) in enumerate(files):
            if f.name in done:
                continue
            img = Image.open(f)
            x = torch.from_numpy(preprocess(img))
            gray = np.asarray(img.convert("L").resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32) / 255
            mk = segment(seg, gray)
            lung = mk["Left Lung"] | mk["Right Lung"]
            # The segmenter (trained on adults) tends to drop consolidated lung; a dilated mask is a sensitivity check
            lung_dil = binary_dilation(lung, iterations=8)
            girdle = mk["Left Clavicle"] | mk["Right Clavicle"] | mk["Left Scapula"] | mk["Right Scapula"]
            for arch, m in models.items():
                cam, p = gradcam(m, target_layer(m, arch), x)
                total = cam.sum() + 1e-8
                peak = np.unravel_index(cam.argmax(), cam.shape)
                w.writerow({
                    "file": f.name, "label": label, "model": arch, "p_pneumonia": round(p, 4),
                    "correct": int((p >= 0.5) == label),
                    "lung_energy": round(float((cam * lung).sum() / total), 4),
                    "lung_energy_dilated": round(float((cam * lung_dil).sum() / total), 4),
                    "girdle_energy": round(float((cam * girdle).sum() / total), 4),
                    "lung_area": round(float(lung.mean()), 4),
                    "girdle_area": round(float(girdle.mean()), 4),
                    "peak_in_lung": int(lung[peak]),
                })
            fh.flush()
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(files)}", flush=True)

    rows = [{k: (v if k in ("file", "model") else float(v)) for k, v in r.items()} for r in csv.DictReader(open(csv_path))]

    print(f"\nn = {len(files)} images; lung mask covers {np.mean([r['lung_area'] for r in rows]):.1%} "
          f"and shoulder girdle {np.mean([r['girdle_area'] for r in rows]):.1%} of the frame on average\n")
    for k, cls in enumerate(CLASSES):
        summarize([x for x in rows if x["label"] == k], f"{cls} images (n = {sum(lb == k for _, lb in files)})")
    summarize(rows, f"all images (n = {len(files)})")


def summarize(rows, title):
    print(f"\n== {title}")
    print(f"{'model':16} {'lung energy (95% CI)':26} {'lung, dilated':14} {'girdle energy (95% CI)':26} {'peak in lung':>12}")
    per = {}
    for arch in ARCHS:
        r = [x for x in rows if x["model"] == arch]
        lung_e = np.array([x["lung_energy"] for x in r])
        gird_e = np.array([x["girdle_energy"] for x in r])
        per[arch] = lung_e
        a, b = ci(lung_e), ci(gird_e)
        dil = np.mean([x["lung_energy_dilated"] for x in r])
        print(f"{arch:16} {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}]       {dil:.3f}          {b[0]:.3f} [{b[1]:.3f}, {b[2]:.3f}]       "
              f"{np.mean([x['peak_in_lung'] for x in r]):>10.1%}")
    for other in ARCHS[1:]:
        d = ci(per[other] - per["resnet18"])
        print(f"paired diff lung energy {other} - resnet18: {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")


if __name__ == "__main__":
    main()
