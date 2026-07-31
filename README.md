# Chest X-Ray Pneumonia Screening — CNN Comparison Demo

A full-stack clinical research demonstration web application that classifies chest X-rays as **NORMAL** or **PNEUMONIA** using a trained ResNet18 CNN model exported to ONNX format, and details a three-model comparison study (ResNet18, MobileNetV2, EfficientNet-B0) highlighting the **Accuracy-vs-Interpretability Trade-off**.

Developed using **FastAPI** (Python backend) and **Vanilla HTML/CSS/JS** (clinical-designed single page frontend), optimized to run within 512MB RAM on Render's free tier.

---

## Technical Stack & Architecture

- **Backend**: Python FastAPI.
  - Serves frontend static files and `/predict` API.
  - Performs image preprocessing (grayscale, replication to 3 channels, resizing to 224x224, and ImageNet standardization) using standard NumPy and Pillow operations.
  - Loads and runs inference using `onnxruntime` CPU provider (no heavy PyTorch or torchvision dependencies).
- **Frontend**: Responsive single-page interface styled with a premium dark clinical-appropriate theme.
  - Features interactive drag-and-drop or file selection.
  - Displays classification confidence in a dynamic radial gauge and detailed risk breakdown card.
  - Showcases the research findings, comparison table, and interactive Grad-CAM heatmap analysis tabs showing ResNet18 (skeletal artifacts focus) vs. MobileNetV2 (lung tissue focus).
- **Deployment**: Configured with a `render.yaml` Blueprint for single-click deployment as a Python web service.

---

## Local Setup & Run

### 1. Place the ONNX Model
Place your trained ONNX model in the `backend/` directory:
- Path: `backend/model.onnx` (or any `.onnx` filename)

*Note: The backend scans this directory at startup and dynamically loads the first `.onnx` file it finds. If no model is found, the server launches in a degraded state and `/health` will report a warning.*

### 2. Install Dependencies
Ensure you have Python 3.10+ installed. In your terminal, run:
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Run the Server
```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser.

---

## Deployment to Render

This repository includes a `render.yaml` blueprint for easy deployment:

1. Create a new repository on GitHub/GitLab and push your local commits:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of full-stack CNN screen demo"
   git remote add origin <your-github-repo-url>
   git branch -M main
   git push -u origin main
   ```
2. Log in to the [Render Dashboard](https://dashboard.render.com).
3. Click **New** -> **Blueprint**.
4. Connect your GitHub repository and approve.
5. Render will parse the `render.yaml` configuration and deploy the service.

*Note: On Render's Free tier, the service spins down after 15 minutes of inactivity. When spinning up after sleep, the first request may take 30-60 seconds to execute. This is expected behavior.*
