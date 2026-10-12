"""Paired bootstrap of AUC across models and test sets, from probability CSVs written by
`export_onnx.py --probs-csv results/{set}_{arch}.csv`.

    python scripts/compare_auc.py results
"""
import sys
from pathlib import Path

import numpy as np

from export_onnx import auc

ARCHS = ["resnet18", "mobilenet_v2", "efficientnet_b0"]
SETS = {"kermany": "pediatric (Kermany test)", "nih": "adult external (NIH, expert labels)"}


def load(path):
    d = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    return d["label"].astype(int), d["p_pneumonia"].astype(float)


def boot(y, ps, n=2000, seed=0):
    """Resample images (stratified) once per draw and score every model on the same draw."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    out = np.empty((n, len(ps)))
    for i in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        out[i] = [auc(y[idx], p[idx]) for p in ps]
    return out


def fmt(v):
    lo, hi = np.percentile(v, [2.5, 97.5])
    return f"{np.mean(v):+.3f} [{lo:+.3f}, {hi:+.3f}]"


root = Path(sys.argv[1])
draws = {}
for s, title in SETS.items():
    data = [load(root / f"{s}_{a}.csv") for a in ARCHS]
    y = data[0][0]
    assert all((d[0] == y).all() for d in data), "models must be scored on the same images"
    draws[s] = boot(y, [d[1] for d in data])
    print(f"\n== {title}, n = {len(y)}")
    for j, a in enumerate(ARCHS):
        print(f"  AUC {a:16} {auc(y, data[j][1]):.3f}  95% CI [{np.percentile(draws[s][:, j], 2.5):.3f}, "
              f"{np.percentile(draws[s][:, j], 97.5):.3f}]")
    for j in (1, 2):
        print(f"  {ARCHS[j]} - resnet18: {fmt(draws[s][:, j] - draws[s][:, 0])}")

# Generalisation drop: independent resamples of the two test sets
drop = draws["nih"] - draws["kermany"]
print("\n== AUC drop, pediatric -> adult external")
for j, a in enumerate(ARCHS):
    print(f"  {a:16} {fmt(drop[:, j])}")
for j in (1, 2):
    print(f"  drop({ARCHS[j]}) - drop(resnet18): {fmt(drop[:, j] - drop[:, 0])}")
