# PneumoScan: accuracy vs. attention in pediatric chest X-ray CNNs

[![Live demo](https://img.shields.io/badge/Live_demo-pneumoscan--nine.vercel.app-0f5f53?style=for-the-badge&logo=vercel&logoColor=white)](https://pneumoscan-nine.vercel.app)
[![Summary PDF](https://img.shields.io/badge/Summary-PDF-16181d?style=for-the-badge)](https://pneumoscan-nine.vercel.app/project_summary.pdf)

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-training-EE4C2C?logo=pytorch&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-inference-005CED?logo=onnx&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![Playwright](https://img.shields.io/badge/Playwright-e2e_tested-2EAD33?logo=playwright&logoColor=white)

**Try it: [pneumoscan-nine.vercel.app](https://pneumoscan-nine.vercel.app)**. Upload a chest X-ray (or click a sample) and see where ResNet18 and MobileNetV2 look, side by side.

[![PneumoScan live demo: verdict and side-by-side Grad-CAM heatmaps](docs/screenshot_demo.webp)](https://pneumoscan-nine.vercel.app/#demo)

Three ImageNet-pretrained CNNs (ResNet18, MobileNetV2, EfficientNet-B0) fine-tuned to detect pneumonia in pediatric chest X-rays. Each model was trained with three seeds. The most accurate one, ResNet18 (88%), puts barely more of its Grad-CAM attention on the lungs than chance would (47% vs 39%). MobileNetV2 puts 67% there, in every seed. All three rank cases equally well (AUC 0.954–0.960). On an external adult dataset every model degrades, and MobileNetV2 degrades least.

> Research demonstration only. Not a medical device and not for clinical use.

![Grad-CAM on a normal X-ray](public/images/gradcam_normal_example.webp)

*A normal X-ray. ResNet18 calls it pneumonia (P = 0.83) with attention on the neck and upper mediastinum. MobileNetV2 correctly calls it normal (P = 0.09).*

## Results

Every architecture was trained three times (seeds 0, 1, 2) with the original notebook's protocol ([`scripts/train_seeds.py`](scripts/train_seeds.py), Kaggle T4). Numbers are mean ± SD across seeds unless marked *served*. The served models come from the original single run.

### 1. In-distribution (Kermany et al. test split, n = 624)

| Model | Accuracy | F1 | AUC | Sensitivity | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| ResNet18 | **88.2 ± 2.9%** | **0.913 ± 0.019** | 0.954 ± 0.013 | 0.989 ± 0.005 | **0.704 ± 0.083** |
| MobileNetV2 | 83.9 ± 2.6% | 0.886 ± 0.016 | 0.956 ± 0.025 | 0.996 ± 0.001 | 0.578 ± 0.072 |
| EfficientNet-B0 | 84.3 ± 0.6% | 0.888 ± 0.004 | **0.960 ± 0.002** | 0.996 ± 0.003 | 0.588 ± 0.017 |

- ResNet18's accuracy lead is real across seeds. At a 0.5 threshold it mostly reflects how many healthy children each model flags.
- **The AUC lead was not real.** The original single run gave ResNet18 0.978 AUC (served model), but retraining gives 0.940, 0.956 and 0.966. All three architectures tie at 0.954–0.960.

**Operating point (served models).** At 0.5 all models flag 30–42% of healthy children. Choosing the threshold for 95% sensitivity (stratified 2-fold CV, 50 splits):

| Served model | Threshold | Sensitivity | Specificity |
| :--- | :---: | :---: | :---: |
| ResNet18 | 0.946 | 0.949 ± 0.021 | **0.882 ± 0.024** |
| MobileNetV2 | 0.951 | 0.948 ± 0.025 | 0.863 ± 0.032 |
| EfficientNet-B0 | 0.992 | 0.950 ± 0.020 | 0.790 ± 0.030 |

The demo uses ResNet18 at 0.946.

### 2. Where the models look

For each image, torchxrayvision's ChestX-Det PSPNet segments the lungs, clavicles and scapulae. We measure the share of Grad-CAM energy (last conv block, predicted class) inside the lung mask. If attention were spread at random, it would equal the lung's share of the image ("chance").

Normal X-rays are the clean comparison. On pneumonia X-rays the adult-trained segmenter drops some consolidated lung.

| Normal X-rays | Pediatric (n = 234) | Per seed | Adult NIH (n = 150) | Per seed |
| :--- | :---: | :---: | :---: | :---: |
| *chance (lung area)* | *38.5%* | | *33.4%* | |
| ResNet18 | 46.8 ± 5.5% | 47.8 · 41.0 · 51.8 | 39.0 ± 5.3% | 38.2 · 34.2 · 44.7 |
| MobileNetV2 | **67.4 ± 3.7%** | 68.6 · 63.3 · 70.3 | **58.3 ± 2.3%** | 55.7 · 60.1 · 59.1 |
| EfficientNet-B0 | 46.9 ± 7.1% | 55.0 · 41.8 · 43.9 | 26.8 ± 15.0% | 44.1 · 16.7 · 19.7 |

**Every MobileNetV2 seed puts more attention on the lungs than every ResNet18 seed**, on both datasets. Seed by seed, MobileNetV2 − ResNet18 is +20.8, +22.3 and +18.5 points on pediatric X-rays.

Per-image detail for the served models ([`results/kermany/`](results/kermany), [`results/nih/`](results/nih)):
- ResNet18 43.9% vs MobileNetV2 71.2% on pediatric normals. The paired difference is +27.3 points (95% CI +26.0 to +28.6).
- The strongest point of the heatmap lands in the lung 21% vs 60% of the time.

A correction to this project's original claim. By eye, ResNet18 seemed to focus on the shoulders. Measured on the served model, its attention on clavicles and scapulae (4.7%) is below their 6.4% share of the image. Its attention isn't on the shoulders; it's spread over the neck, mediastinum and image borders, and mostly not on the lungs.

### 3. External test (NIH ChestX-ray14, adult, n = 656)

Batch 7 of NIH ChestX-ray14, restricted to images with radiologist-adjudicated labels from Majkowska et al. (Radiology, 2020): 364 *airspace opacity*, 292 with all four expert labels negative. Mostly AP views.

| Model | Pediatric AUC | NIH AUC | Drop |
| :--- | :---: | :---: | :---: |
| ResNet18 | 0.954 ± 0.013 | 0.794 ± 0.023 | −0.160 |
| MobileNetV2 | 0.956 ± 0.025 | **0.822 ± 0.024** | **−0.134** |
| EfficientNet-B0 | 0.960 ± 0.002 | 0.758 ± 0.012 | −0.202 |

- MobileNetV2, the model that looks at the lungs, degrades least on average. Seed by seed, MobileNetV2 − ResNet18 NIH AUC is −0.017, +0.078 and +0.026. That's a trend, not a significant result, with three seeds.
- The pediatric threshold doesn't transfer. At 0.946, the served ResNet18 flags 80% of healthy adults (specificity 0.199).
- The shift mixes a new hospital with a new age group, and it hurts every model badly.

Raw outputs: [`results/`](results), which holds per-image probabilities for every model and seed, per-image attention scores, and the bootstrap comparison for the served models.

## Live demo: [pneumoscan-nine.vercel.app](https://pneumoscan-nine.vercel.app)

- The landing page (`/`) has the 3D exploded hero and a WebGL attention landscape; the full report and the demo live at [`/research.html`](https://pneumoscan-nine.vercel.app/research.html).
- Upload an X-ray, or click a bundled test-set sample.
- Both served models return P(pneumonia) and a Grad-CAM overlay, shown side by side.
- The heatmaps are exact Grad-CAM at no extra cost. All three heads are global-average-pool → linear, so Grad-CAM on the last conv block equals the class activation map `Σ_k W[c,k]·A_k`. The ONNX graphs output it directly (verified correlation = 1.0 against autograd Grad-CAM).
- Uploads are downscaled in the browser to stay under Vercel's 4.5 MB request limit.

## Stack

- **Serving:** FastAPI + ONNX Runtime (CPU) as a Vercel Python function. NumPy/Pillow preprocessing identical to training: grayscale → 3 channels → 224×224 bilinear → ImageNet normalisation.
- **Frontend:** static HTML/CSS/JS in `public/`, served from Vercel's CDN, with light and dark themes.
- **Tests:** a Playwright end-to-end test (desktop and mobile). It uploads the samples, checks the verdicts and heatmaps, and checks for horizontal overflow and JS errors.

## Reproduce

```bash
python -m venv venv && venv\Scripts\activate          # or: source venv/bin/activate
pip install -r requirements.txt                        # serving only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torchxrayvision scipy pandas playwright pytest && playwright install chromium
```

| Step | Command |
| :--- | :--- |
| Train (Kaggle T4, ~10 epochs/model) | [`notebooks/chest-x-ray.ipynb`](notebooks/chest-x-ray.ipynb) |
| Export to ONNX + metrics + operating point | `python scripts/export_onnx.py resnet18_best.pth --arch resnet18 --test-dir chest_xray/test --probs-csv results/kermany_resnet18.csv` |
| Build NIH external set | download `images_007.tar.gz` from the [NIH Box](https://nihcc.app.box.com/v/ChestXray-NIHCC), extract, then `python scripts/prepare_nih.py images nih_test` |
| Retrain with seeds (Kaggle GPU) | [`notebooks/train_seeds_kaggle.ipynb`](notebooks/train_seeds_kaggle.ipynb), or `python scripts/train_seeds.py --data-dir chest_xray` |
| Score seed checkpoints (NIH + attention) | `python scripts/seed_analysis.py --ckpt-dir checkpoints --kermany-test chest_xray/test --nih-dir nih_test --mask-cache masks.npz` |
| AUC comparison, served models (paired bootstrap) | `python scripts/compare_auc.py results` |
| Attention analysis | `python scripts/gradcam_overlap.py --weights-dir <dir with *_best.pth> --test-dir chest_xray/test --out results/kermany` |
| Landing-page 3D layers (X-ray, lung mask, Grad-CAMs) | `python scripts/make_hero_layers.py chest_xray/test/NORMAL/NORMAL2-IM-0111-0001.jpeg --mask-cache masks.npz` |
| Summary PDF (from `docs/summary.html`) | `python scripts/make_summary_pdf.py` |
| Run locally | `uvicorn main:app --port 8000` |
| End-to-end test | `pytest tests` (or `BASE_URL=https://pneumoscan-nine.vercel.app pytest tests`) |

Pushes to `main` deploy to Vercel automatically at **https://pneumoscan-nine.vercel.app**.

## Limitations

- **Three seeds per model.** Enough to show the attention gap is stable and the original AUC lead was luck; too few for formal significance tests between architectures.
- **The external test confounds two shifts:** a different hospital and adults instead of children (and mostly AP views).
- **Lung masks come from a model trained on adults,** not radiologist annotation. It undercounts consolidated lung on pediatric pneumonia X-rays, which is why normal X-rays are the primary comparison.
- **The validation split is 16 images.** Operating points were chosen by cross-validation on the test set, so they're optimistic for a truly new site, as the NIH result shows.
- **Not a medical device.** No FDA or CE clearance.

## Data and citations

- Kermany D.S. et al., "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning," *Cell* 172(5), 2018.
- Wang X. et al., "ChestX-ray8," CVPR 2017 (NIH ChestX-ray14).
- Majkowska A. et al., "Chest Radiograph Interpretation with Deep Learning Models," *Radiology* 294(2), 2020 (expert labels).
- Cohen J.P. et al., "TorchXRayVision," MIDL 2022 (segmentation model).
