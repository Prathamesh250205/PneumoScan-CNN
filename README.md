# 🩺 Chest X-Ray Pneumonia Screening — CNN Comparison Demo

> **Comparative Deep Learning Analysis & Interpretability Trade-Off Study**
>
> *A medical-research screening demonstration trained on the Kermany et al. Chest X-Ray Images (Pneumonia) dataset.*

---

## 📊 Research Summary & Core Findings

This project evaluates three Convolutional Neural Network (CNN) architectures (**ResNet18**, **MobileNetV2**, **EfficientNet-B0**) trained via transfer learning to classify chest X-ray scans as **NORMAL** or **PNEUMONIA**.

### The Accuracy-vs-Interpretability Trade-off

The core finding of this research highlights a critical clinical dilemma: **the highest accuracy model is not necessarily the most trustworthy.**

*   **ResNet18** achieved the highest raw classification accuracy (**87.8%**). However, **Grad-CAM explainability analysis** revealed that it frequently focused on **non-diagnostic regions** (such as the shoulder joint structure, patient posture, and collarbone) to make its decisions.
*   **MobileNetV2** achieved a slightly lower accuracy (**85.6%**) but consistently localized its convolutional attention on the **actual lung tissue and pulmonary parenchyma**, aligning with standard radiological diagnostic criteria.

This demonstrates that lower-accuracy models can sometimes be more clinically robust, explainable, and safe for diagnostic deployment.

---

## ⚡ Model Performance Comparison

| Architecture | Classification Accuracy | F1-Score | AUC-ROC | Clinical Focus (Grad-CAM) | Trustworthiness |
| :--- | :---: | :---: | :---: | :--- | :---: |
| 🏆 **ResNet18** (Live Model) | **87.8%** | **0.911** | **0.978** | ⚠️ Artifacts (Shoulder/Collarbone) | **Low** |
| 📱 **MobileNetV2** | **85.6%** | **0.896** | *N/A* | ✅ Anatomical Lung Fields | **High** |
| 🌿 **EfficientNet-B0** | **84.8%** | **0.891** | *N/A* | 🔍 Diffuse Pulmonary Regions | **Moderate** |

---

## 🔍 Visualizing Diagnostic Focus: Grad-CAM

### ResNet18 (Shoulder Focus) vs. MobileNetV2 (Lung Focus)
Grad-CAM heatmaps highlight where each network focuses its weights:

```
[ResNet18 Focus]                        [MobileNetV2 Focus]
      |                                        |
      v (Shoulder Joint)                        v (Lung Tissue)
 ┌───────────────┐                        ┌───────────────┐
 │   🔴   ░░░    │                        │     ░░░░      │
 │  ░░░░░░░░░░   │                        │   ░░ 🔴 ░░    │
 │  ░░░░░░░░░░   │                        │   ░░ 🔴 ░░    │
 └───────────────┘                        └───────────────┘
```

*   **Pneumonia Specimen**: Heatmaps concentrate on the lower lobe consolidation regions.
*   **Normal Specimen**: Heatmaps remain diffuse, indicating unremarkable, healthy air-filled lung fields.

---

## 🛠️ Tech Stack & Micro-Architecture

To run efficiently in resource-constrained cloud environments (such as Render's 512MB RAM free tier), this application is designed without heavy deep learning frameworks:

*   **Backend**: **FastAPI** + **Uvicorn** for a high-performance, asynchronous web API.
*   **Inference Engine**: **ONNX Runtime** (CPU Provider), running predictions with a minimal memory footprint.
*   **Image Processing**: Pure **NumPy** & **Pillow** implementing the identical pipeline used during training:
    1. Grayscale conversion.
    2. Dimension replication to 3 channels (RGB format).
    3. Bilinear resizing to $224 \times 224$ pixels.
    4. ImageNet normalization: $\text{mean} = [0.485, 0.456, 0.406]$ and $\text{std} = [0.229, 0.224, 0.225]$.
*   **Frontend**: Professional clinical-grade single-page application built using semantic **HTML5**, custom **Vanilla CSS**, and **Vanilla Javascript**. Contains:
    *   Drag-and-drop / file-picker upload sandboxes.
    *   Dynamic semi-circular risk gauge reflecting risk classification.
    *   Interactive tabs displaying comparative Grad-CAM heatmaps.

---

## 🚀 Deployed on Render

This project contains a `render.yaml` Blueprint definition mapping a Python Web Service.

```yaml
services:
  - type: web
    name: chest-xray-pneumonia-screening
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
```

> [!NOTE]
> **Render Free Tier Spin-Down**: Render automatically spins down free-tier services after 15 minutes of inactivity. When visiting the site after an idle period, the first request will trigger a cold start taking 30-60 seconds. All subsequent prediction requests will process instantly (<2 seconds).

---

## 💻 Local Setup Instructions

### 1. Place the ONNX Model
Copy your trained ResNet18 model to the `backend` folder:
- **Target Path**: `backend/model.onnx`

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate

# Activate environment (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch the Server
```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
Visit [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## ⚖️ Clinical Disclaimer
**This application is a student research demonstration for educational purposes only. It is NOT an FDA-cleared diagnostic tool and should never be used to make clinical decisions or assessments. Consult a licensed radiologist or medical professional for health evaluations.**
