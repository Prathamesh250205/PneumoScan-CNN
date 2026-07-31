import os
import glob
import logging
import numpy as np
from PIL import Image
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
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
STATIC_DIR = os.path.join(BASE_DIR, "static")

# State variables
model_path = None
session = None

def init_model():
    """Locate and load the first ONNX model found in the backend/ folder."""
    global model_path, session
    
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
            
            # Print model input/output info
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            logger.info(f"Model Input Node: Name={inputs[0].name}, Shape={inputs[0].shape}, Type={inputs[0].type}")
            logger.info(f"Model Output Node: Name={outputs[0].name}, Shape={outputs[0].shape}, Type={outputs[0].type}")
        except Exception as e:
            logger.error(f"Error loading ONNX model: {str(e)}")
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
    global session, model_path
    
    # Try re-initializing the model if it wasn't loaded (in case file was placed after startup)
    if session is None:
        init_model()
        
    model_loaded = (session is not None)
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_path": os.path.basename(model_path) if model_path else None
    }

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def sigmoid(x):
    """Compute sigmoid values for x."""
    return 1 / (1 + np.exp(-x))

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
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload an image file."
        )
        
    try:
        # Load image via Pillow
        image = Image.open(file.file)
        
        # 1. Convert to grayscale
        gray_image = image.convert("L")
        
        # 2. Replicate grayscale value to 3 channels
        rgb_image = gray_image.convert("RGB")
        
        # 3. Resize to 224x224 using Bilinear interpolation
        resized_image = rgb_image.resize((224, 224), Image.Resampling.BILINEAR)
        
        # 4. Scale to [0.0, 1.0] and convert to numpy array
        img_data = np.array(resized_image).astype(np.float32) / 255.0
        
        # Reorder shape from HWC to CHW (3, 224, 224)
        img_data = np.transpose(img_data, (2, 0, 1))
        
        # 5. Normalize with ImageNet parameters
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        normalized_img = (img_data - mean) / std
        
        # 6. Add batch dimension -> (1, 3, 224, 224)
        input_tensor = np.expand_dims(normalized_img, axis=0).astype(np.float32)
        
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

# Serve Frontend Index at /
@app.get("/")
async def read_root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Welcome. The static index.html is missing. Please create it under static/index.html."}

# Mount static directory for JS, CSS, and Images
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
