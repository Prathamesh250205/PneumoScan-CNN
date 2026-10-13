"""Score every seed checkpoint from train_seeds.py: test metrics, NIH external AUC and lung attention on normal X-rays,
then summarise mean ± SD across seeds.

    python scripts/seed_analysis.py --ckpt-dir checkpoints --seeds-dir results/seeds \
        --kermany-test chest_xray/test --nih-dir nih_test --mask-cache masks.npz

Writes {seeds-dir}/{arch}_s{seed}_nih.csv, {seeds-dir}/attention.csv and {seeds-dir}/summary.md.
Lung masks (torchxrayvision PSPNet) are model-independent, so they're computed once and cached.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from main import preprocess, softmax  # noqa: E402
from export_onnx import auc, build, confusion  # noqa: E402
from gradcam_overlap import gradcam, segment, target_layer  # noqa: E402

ARCHS = ["resnet18", "mobilenet_v2", "efficientnet_b0"]
NAMES = {"resnet18": "ResNet18", "mobilenet_v2": "MobileNetV2", "efficientnet_b0": "EfficientNet-B0"}
NIH_NORMAL_LIMIT = 150  # same subset as the single-seed attention analysis


def normals(args):
    ped = sorted((Path(args.kermany_test) / "NORMAL").iterdir())
    nih = sorted((Path(args.nih_dir) / "NORMAL").iterdir())[:NIH_NORMAL_LIMIT]
    return {"kermany": ped, "nih": nih}


def lung_masks(files, cache):
    if Path(cache).exists():
        return dict(np.load(cache))
    import torchxrayvision as xrv
    seg = xrv.baseline_models.chestx_det.PSPNet().eval()
    masks = {}
    for i, f in enumerate([f for fs in files.values() for f in fs]):
        gray = np.asarray(Image.open(f).convert("L").resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        mk = segment(seg, gray)
        masks[f.name] = mk["Left Lung"] | mk["Right Lung"]
        if (i + 1) % 50 == 0:
            print(f"masks {i + 1}", flush=True)
    np.savez_compressed(cache, **masks)
    return masks


def read_probs(path):
    rows = list(csv.DictReader(open(path)))
    return np.array([int(r["label"]) for r in rows]), np.array([float(r["p_pneumonia"]) for r in rows])


def metrics(y, p):
    tp, tn, fp, fn = confusion(y, p, 0.5)
    return {"acc": (tp + tn) / len(y), "f1": 2 * tp / (2 * tp + fp + fn), "auc": auc(y, p),
            "sens": tp / (tp + fn), "spec": tn / (tn + fp)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--seeds-dir", default=str(ROOT / "results" / "seeds"))
    ap.add_argument("--kermany-test", required=True)
    ap.add_argument("--nih-dir", required=True)
    ap.add_argument("--mask-cache", required=True)
    ap.add_argument("--masks-only", action="store_true")
    args = ap.parse_args()

    files = normals(args)
    masks = lung_masks(files, args.mask_cache)
    if args.masks_only:
        print(f"{len(masks)} masks cached")
        return

    seeds_dir = Path(args.seeds_dir)
    att_path = seeds_dir / "attention.csv"
    done = {(r["model"], r["seed"]) for r in csv.DictReader(open(att_path))} if att_path.exists() else set()
    nih_files = [(f, k) for k, c in enumerate(["NORMAL", "PNEUMONIA"]) for f in sorted((Path(args.nih_dir) / c).iterdir())]

    for ck in sorted(Path(args.ckpt_dir).glob("*_s*.pth")):
        arch, seed = ck.stem.rsplit("_s", 1)
        m = build(arch)
        m.load_state_dict(torch.load(ck, map_location="cpu"))
        for mod in m.modules():  # in-place activations break tensor grad hooks
            if hasattr(mod, "inplace"):
                mod.inplace = False
        m.eval()

        nih_csv = seeds_dir / f"{arch}_s{seed}_nih.csv"
        if not nih_csv.exists():
            with torch.no_grad():
                ps = [float(softmax(m(torch.from_numpy(preprocess(Image.open(f)))).numpy())[0][1]) for f, _ in nih_files]
            nih_csv.write_text("file,label,p_pneumonia\n" + "".join(f"{f.name},{k},{q:.6f}\n" for (f, k), q in zip(nih_files, ps)))

        if (arch, seed) not in done:
            row = {"model": arch, "seed": seed}
            for name, fs in files.items():
                shares = []
                for f in fs:
                    cam, _ = gradcam(m, target_layer(m, arch), torch.from_numpy(preprocess(Image.open(f))))
                    shares.append(float((cam * masks[f.name]).sum() / (cam.sum() + 1e-8)))
                row[f"lung_{name}"] = round(float(np.mean(shares)), 4)
            new = not att_path.exists()
            with open(att_path, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["model", "seed", "lung_kermany", "lung_nih"])
                if new:
                    w.writeheader()
                w.writerow(row)
        print(f"scored {ck.name}", flush=True)

    # Summary: mean ± SD across seeds
    att = list(csv.DictReader(open(att_path)))
    chance = {k: float(np.mean([masks[f.name].mean() for f in fs])) for k, fs in files.items()}
    lines = ["| Model | Seeds | Accuracy | F1 | AUC | Sensitivity | Specificity | NIH AUC | Lung attention, pediatric | Lung attention, NIH |",
             "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
    per_seed_auc = {}
    for arch in ARCHS:
        seeds = sorted(r["seed"] for r in att if r["model"] == arch)
        if not seeds:
            continue
        ker = [metrics(*read_probs(seeds_dir / f"{arch}_s{s}_kermany.csv")) for s in seeds]
        nih = [auc(*read_probs(seeds_dir / f"{arch}_s{s}_nih.csv")) for s in seeds]
        per_seed_auc[arch] = dict(zip(seeds, zip([k["auc"] for k in ker], nih)))
        lung = {k: [float(r[f"lung_{k}"]) for r in att if r["model"] == arch] for k in ("kermany", "nih")}

        def ms(v, pct=False):
            v = np.array(v) * (100 if pct else 1)
            return f"{v.mean():.1f} ± {v.std(ddof=1):.1f}%" if pct else f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"
        lines.append(f"| {NAMES[arch]} | {len(seeds)} | {ms([k['acc'] for k in ker], True)} | {ms([k['f1'] for k in ker])} | "
                     f"{ms([k['auc'] for k in ker])} | {ms([k['sens'] for k in ker])} | {ms([k['spec'] for k in ker])} | "
                     f"{ms(nih)} | {ms(lung['kermany'], True)} | {ms(lung['nih'], True)} |")
    lines.append(f"\nChance lung attention (lung share of image): pediatric {chance['kermany']:.1%}, NIH {chance['nih']:.1%}. "
                 "Mean ± SD across seeds; SD uses ddof=1.")
    if {"resnet18", "mobilenet_v2"} <= per_seed_auc.keys():
        common = sorted(set(per_seed_auc["resnet18"]) & set(per_seed_auc["mobilenet_v2"]))
        d = [per_seed_auc["mobilenet_v2"][s][0] - per_seed_auc["resnet18"][s][0] for s in common]
        e = [per_seed_auc["mobilenet_v2"][s][1] - per_seed_auc["resnet18"][s][1] for s in common]
        lines.append(f"\nMobileNetV2 − ResNet18 AUC, per seed: pediatric {', '.join(f'{x:+.3f}' for x in d)}; "
                     f"NIH {', '.join(f'{x:+.3f}' for x in e)}.")
    summary = "\n".join(lines)
    (seeds_dir / "summary.md").write_text(summary + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
