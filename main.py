import os
import glob
import logging
import numpy as np
from PIL import Image
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chest X-Ray Pneumonia Screening — CNN Comparison Demo",
    description="FastAPI service for Chest X-Ray classification using a trained ResNet18 ONNX model.",
    version="1.0.0"
)

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
PUBLIC_DIR = os.path.join(BASE_DIR, "public")

# State variables
model_path = None
session = None
model_error = None

def init_model():
    """Locate and load the first ONNX model found in the backend/ folder."""
    global model_path, session, model_error
    
    if not os.path.exists(BACKEND_DIR):
        os.makedirs(BACKEND_DIR, exist_ok=True)
        logger.info(f"Created backend directory at: {BACKEND_DIR}")
        
    onnx_files = glob.glob(os.path.join(BACKEND_DIR, "*.onnx"))
    
    if onnx_files:
        model_path = onnx_files[0]
        logger.info(f"Found ONNX model at: {model_path}. Loading...")
        try:
            # We enforce CPUExecutionProvider to run reliably on Render's standard resource container and keep memory footprint low
            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            logger.info("ONNX Model loaded successfully.")
            model_error = None
            
            # Print model input/output info
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            logger.info(f"Model Input Node: Name={inputs[0].name}, Shape={inputs[0].shape}, Type={inputs[0].type}")
            logger.info(f"Model Output Node: Name={outputs[0].name}, Shape={outputs[0].shape}, Type={outputs[0].type}")
        except Exception as e:
            model_error = str(e)
            logger.error(f"Error loading ONNX model: {model_error}")
            session = None
    else:
        logger.warning(f"No .onnx files found in '{BACKEND_DIR}'. "
                       f"Please place your model.onnx file in this directory.")
        session = None

# Load model at startup
@app.on_event("startup")
async def startup_event():
    init_model()

@app.get("/health")
def health_check():
    """Health check endpoint to monitor service status and model loading state."""
    global session, model_path, model_error
    
    # Try re-initializing the model if it wasn't loaded (in case file was placed after startup)
    if session is None:
        init_model()
        
    model_loaded = (session is not None)
    # ponytail: size heuristic — real ResNet18 weights are ~45MB, the committed placeholder is ~8KB
    weights_bytes = sum(os.path.getsize(f) for f in glob.glob(model_path + "*")) if model_path else 0
    return {
        "placeholder_weights": model_loaded and weights_bytes < 1_000_000,
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_path": os.path.basename(model_path) if model_path else None,
        "model_error": model_error
    }

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def sigmoid(x):
    """Compute sigmoid values for x."""
    return 1 / (1 + np.exp(-x))

def preprocess(image):
    """Same as the training eval transform: grayscale -> 3ch -> 224x224 bilinear -> [0,1] -> ImageNet norm -> NCHW."""
    img = image.convert("L").convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    return ((x - mean) / std)[None].astype(np.float32)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict pneumonia probability from a chest X-Ray image.
    
    Preprocessing pipeline:
      1. Convert to grayscale.
      2. Replicate to 3 channels.
      3. Resize to 224x224.
      4. Scale pixel values to [0, 1].
      5. Normalize with ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225].
      6. Convert to NCHW float32 tensor and run inference.
    """
    global session
    
    # Check if the model is loaded, try reloading once
    if session is None:
        init_model()
        
    if session is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please place your model.onnx file in the backend directory."
        )
        
    # Validate uploaded file type
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload an image file."
        )
        
    try:
        input_tensor = preprocess(Image.open(file.file))

        # Run inference
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        raw_outputs = session.run([output_name], {input_name: input_tensor})[0]
        
        # Parse outputs based on shape
        # ResNet18 trained via PyTorch ImageFolder outputs shape (1, 2)
        # Class 0: NORMAL, Class 1: PNEUMONIA
        if len(raw_outputs.shape) > 1 and raw_outputs.shape[1] == 2:
            probabilities = softmax(raw_outputs)[0]
            normal_prob = float(probabilities[0])
            pneumonia_prob = float(probabilities[1])
        elif len(raw_outputs.shape) > 1 and raw_outputs.shape[1] == 1:
            # Single logit output for binary classification
            pneumonia_prob = float(sigmoid(raw_outputs[0][0]))
            normal_prob = 1.0 - pneumonia_prob
        else:
            # Flattened output vector
            flat_val = raw_outputs.flatten()
            if len(flat_val) == 2:
                probabilities = softmax(flat_val)
                normal_prob = float(probabilities[0])
                pneumonia_prob = float(probabilities[1])
            else:
                pneumonia_prob = float(sigmoid(flat_val[0]))
                normal_prob = 1.0 - pneumonia_prob
                
        return {
            "normal_probability": normal_prob,
            "pneumonia_probability": pneumonia_prob
        }
        
    except Exception as e:
        logger.error(f"Error during image processing or inference: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference processing failed: {str(e)}"
        )

# Local dev: serve the frontend. On Vercel, public/ is served by the CDN and this dir is not in the function bundle.
if os.path.isdir(PUBLIC_DIR):
    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")
