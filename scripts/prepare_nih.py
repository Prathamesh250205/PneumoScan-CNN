"""Build an external test set from NIH ChestX-ray14 with radiologist-adjudicated labels
(Majkowska et al., Radiology 2020; Google relabel, shipped with torchxrayvision).

PNEUMONIA = "Airspace opacity" YES; NORMAL = all four expert labels NO. Images with only other findings are skipped.

    # download one batch (e.g. images_007.tar.gz) from https://nihcc.app.box.com/v/ChestXray-NIHCC and extract
    python scripts/prepare_nih.py path/to/images out_dir
"""
import shutil
import sys
from pathlib import Path

import pandas as pd
import torchxrayvision as xrv

src, out = Path(sys.argv[1]), Path(sys.argv[2])
labels = pd.read_csv(Path(xrv.__file__).parent / "data" / "google2019_nih-chest-xray-labels.csv.gz")
expert = ["Fracture", "Pneumothorax", "Airspace opacity", "Nodule or mass"]
labels["cls"] = None
labels.loc[labels["Airspace opacity"] == "YES", "cls"] = "PNEUMONIA"
labels.loc[(labels[expert] == "NO").all(axis=1), "cls"] = "NORMAL"

counts = {}
for _, r in labels.dropna(subset=["cls"]).iterrows():
    f = src / r["Image Index"]
    if f.exists():
        (out / r["cls"]).mkdir(parents=True, exist_ok=True)
        shutil.copy(f, out / r["cls"] / f.name)
        counts[(r["cls"], r["View Position"])] = counts.get((r["cls"], r["View Position"]), 0) + 1
print(counts)
