"""Retrain each architecture with several seeds using the exact protocol of notebooks/chest-x-ray.ipynb
(ImageNet weights, 2-class head, Adam 1e-4, batch 32, 10 epochs, flip + ±10° rotation, best-val-accuracy checkpoint).

    # Kaggle (GPU): attach the Kermany dataset, enable internet, then
    python train_seeds.py --data-dir /kaggle/input/<dataset>/chest_xray --out /kaggle/working/seeds
    # local
    python scripts/train_seeds.py --data-dir chest_xray --out results/seeds --ckpt-dir checkpoints

Per run it writes {out}/{arch}_s{seed}_kermany.csv (file,label,p_pneumonia on the test split) and
{ckpt-dir}/{arch}_s{seed}.pth. Finished runs are skipped, so re-running resumes.
Self-contained on purpose (no repo imports) so it can be uploaded to Kaggle as a single file.
"""
import argparse
import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models
from torchvision.transforms import v2

CLASSES = ["NORMAL", "PNEUMONIA"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_split(root):
    """Grayscale -> 224x224 bilinear, cached as uint8 (same as the notebook's Grayscale + Resize on PIL)."""
    xs, ys, names = [], [], []
    for label, cls in enumerate(CLASSES):
        for f in sorted((Path(root) / cls).iterdir()):
            img = Image.open(f).convert("L").resize((224, 224), Image.Resampling.BILINEAR)
            xs.append(torch.from_numpy(np.asarray(img).copy()))
            ys.append(label)
            names.append(f.name)
    return torch.stack(xs)[:, None], torch.tensor(ys), names


def to_input(x_uint8, device):
    x = x_uint8.to(device).float().div(255).repeat(1, 3, 1, 1)
    return (x - MEAN.to(device)) / STD.to(device)


def build(arch):
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif arch == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    else:
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    return m


@torch.no_grad()
def predict(model, x, device, bs=64):
    model.eval()
    return torch.cat([torch.softmax(model(to_input(x[i:i + bs], device)), 1)[:, 1].cpu() for i in range(0, len(x), bs)])


def train_run(arch, seed, train, val, args, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = build(arch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    augment = v2.Compose([v2.RandomHorizontalFlip(), v2.RandomRotation(10)])
    gen = torch.Generator().manual_seed(seed)
    (xt, yt), (xv, yv) = train, val
    best_acc, best = -1.0, None
    for epoch in range(args.epochs):
        t0, model_loss = time.time(), 0.0
        model.train()
        order = torch.randperm(len(xt), generator=gen)
        for i in range(0, len(order), 32):
            idx = order[i:i + 32]
            xb = torch.stack([augment(img) for img in xt[idx]])
            yb = yt[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(to_input(xb, device)), yb)
            loss.backward()
            opt.step()
            model_loss += loss.item() * len(idx)
        val_acc = ((predict(model, xv, device) >= 0.5).long() == yv).float().mean().item()
        if val_acc > best_acc:  # notebook keeps the best validation accuracy, first one on ties
            best_acc, best = val_acc, copy.deepcopy(model.state_dict())
        print(f"  {arch} seed {seed} epoch {epoch + 1}/{args.epochs}  loss {model_loss / len(xt):.4f}  "
              f"val_acc {val_acc:.3f}  {time.time() - t0:.0f}s", flush=True)
    model.load_state_dict(best)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="dir with train/, val/, test/ (each NORMAL/ and PNEUMONIA/)")
    ap.add_argument("--archs", nargs="+", default=["resnet18", "mobilenet_v2", "efficientnet_b0"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out", default="results/seeds")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: images per class per split")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out, ckpt = Path(args.out), Path(args.ckpt_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)

    splits = {}
    for s in ("train", "val", "test"):
        x, y, names = load_split(Path(args.data_dir) / s)
        if args.limit:
            keep = torch.cat([torch.nonzero(y == k).flatten()[:args.limit] for k in (0, 1)])
            x, y, names = x[keep], y[keep], [names[i] for i in keep]
        splits[s] = (x, y, names)
        print(f"{s}: {len(y)} images", flush=True)

    for arch in args.archs:
        for seed in args.seeds:
            csv_path = out / f"{arch}_s{seed}_kermany.csv"
            if csv_path.exists():
                print(f"skip {arch} seed {seed} (done)")
                continue
            model = train_run(arch, seed, splits["train"][:2], splits["val"][:2], args, device)
            torch.save(model.state_dict(), ckpt / f"{arch}_s{seed}.pth")
            xs, ys, names = splits["test"]
            p = predict(model, xs, device)
            csv_path.write_text("file,label,p_pneumonia\n" + "".join(
                f"{n},{int(lb)},{q:.6f}\n" for n, lb, q in zip(names, ys, p.tolist())))
            acc = ((p >= 0.5).long() == ys).float().mean().item()
            print(f"{arch} seed {seed}: test acc {acc:.3f}", flush=True)


if __name__ == "__main__":
    main()
