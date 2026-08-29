import io
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# PROJECT PATHS
# ============================================================

# Current file:
#
#   PramaanSetu/
#       api/
#           index.py
#
# Model:
#
#   PramaanSetu/
#       models/
#           ai-detector/

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models" / "ai-detector"


# ============================================================
# CONFIGURATION
# ============================================================

# Keep this below Vercel's request payload limit.
MAX_UPLOAD_BYTES = 4 * 1024 * 1024

# Vercel Functions run without a CUDA GPU.
# Always use CPU.
DEVICE = torch.device("cpu")


# ============================================================
# GLOBAL MODEL OBJECTS
# ============================================================

processor = None
model = None


# ============================================================
# LOAD MODEL
# ============================================================

def load_local_model():
    """
    Load the AI detector from the model files stored locally
    inside /models/ai-detector.
    """

    global processor
    global model

    print("=" * 70)
    print("PramaanSetu AI Detector")
    print("=" * 70)

    print(f"Project directory: {BASE_DIR}")
    print(f"Model directory: {MODEL_DIR}")
    print(f"Device: {DEVICE}")

    # --------------------------------------------------------
    # Check model directory
    # --------------------------------------------------------

    if not MODEL_DIR.exists():

        raise RuntimeError(
            "AI model directory was not found.\n"
            f"Expected location: {MODEL_DIR}"
        )

    if not MODEL_DIR.is_dir():

        raise RuntimeError(
            f"AI model path is not a directory: {MODEL_DIR}"
        )

    # --------------------------------------------------------
    # Check for model files
    # --------------------------------------------------------

    model_files = list(MODEL_DIR.iterdir())

    print("Model directory contents:")

    for item in model_files:
        print(f"  - {item.name}")

    # --------------------------------------------------------
    # Load image processor
    # --------------------------------------------------------

    try:

        print("Loading image processor...")

        processor = AutoImageProcessor.from_pretrained(
            str(MODEL_DIR),
            local_files_only=True,
        )

        print("Image processor loaded.")

    except Exception as error:

        print(
            "Failed to load image processor:"
        )

        print(repr(error))

        processor = None

        raise RuntimeError(
            "Failed to load the AI image processor."
        ) from error

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    try:

        print("Loading AI model...")

        model = AutoModelForImageClassification.from_pretrained(
            str(MODEL_DIR),
            local_files_only=True,
        )

        print("AI model loaded.")

    except Exception as error:

        print(
            "Failed to load AI model:"
        )

        print(repr(error))

        model = None

        raise RuntimeError(
            "Failed to load the local AI detector."
        ) from error

    # --------------------------------------------------------
    # Move model to CPU
    # --------------------------------------------------------

    model.to(DEVICE)

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    model.eval()

    print("=" * 70)
    print("Local AI detector loaded successfully.")
    print("=" * 70)


# ============================================================
# FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(_: FastAPI):

    load_local_model()

    yield


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="PramaanSetu AI Screening",
    description=(
        "AI-generated image screening API "
        "for the PramaanSetu prototype."
    ),
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],
    allow_headers=["*"],
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():

    return {
        "ready": model is not None,
        "model": "Smogy/SMOGY-Ai-images-detector",
        "mode": "local",
        "device": str(DEVICE),
    }


# ============================================================
# AI DETECTION ENDPOINT
# ============================================================

@app.post("/api/detect")
async def detect_ai_generated_image(
    file: UploadFile = File(...),
):

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if model is None or processor is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "The local AI detector "
                "is not loaded yet."
            ),
        )

    # --------------------------------------------------------
    # Validate content type
    # --------------------------------------------------------

    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):

        raise HTTPException(
            status_code=415,
            detail=(
                "Please upload a JPG, PNG, "
                "or other image file."
            ),
        )

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    try:

        image_bytes = await file.read()

    except Exception as error:

        print(
            f"Failed to read uploaded file: {error!r}"
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not read the uploaded file."
            ),
        ) from error

    # --------------------------------------------------------
    # Empty file
    # --------------------------------------------------------

    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="The selected image is empty.",
        )

    # --------------------------------------------------------
    # File size
    # --------------------------------------------------------

    if len(image_bytes) > MAX_UPLOAD_BYTES:

        raise HTTPException(
            status_code=413,
            detail=(
                "Images must be 4 MB or smaller."
            ),
        )

    # --------------------------------------------------------
    # Decode image
    # --------------------------------------------------------

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image = image.convert("RGB")

    except Exception as error:

        print(
            f"Image decoding failed: {error!r}"
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded file is not "
                "a valid image."
            ),
        ) from error

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    try:

        print(
            f"Running AI detection for: "
            f"{file.filename}"
        )

        # Convert image into model inputs.
        inputs = processor(
            images=image,
            return_tensors="pt",
        )

        # Move tensors to CPU.
        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        # Run model without gradients.
        with torch.inference_mode():

            outputs = model(**inputs)

        # Convert logits to probabilities.
        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )[0]

        predictions = []

        # ----------------------------------------------------
        # Convert model predictions to JSON
        # ----------------------------------------------------

        for index, score in enumerate(
            probabilities
        ):

            label = model.config.id2label.get(
                index,
                str(index),
            )

            predictions.append(
                {
                    "label": str(label),
                    "score": float(score),
                }
            )

        # Highest confidence first.
        predictions.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        print(
            "Model predictions:",
            predictions,
        )

    except Exception as error:

        print(
            "Model inference failed:"
        )

        print(repr(error))

        raise HTTPException(
            status_code=500,
            detail=(
                "The local AI detector failed "
                "while reviewing this image."
            ),
        ) from error

    # ========================================================
    # INTERPRET MODEL LABELS
    # ========================================================

    synthetic = None
    real = None

    for prediction in predictions:

        label_lower = (
            prediction["label"].lower()
        )

        # ----------------------------------------------------
        # AI / synthetic labels
        # ----------------------------------------------------

        if (
            synthetic is None
            and any(
                word in label_lower
                for word in (
                    "ai",
                    "fake",
                    "generated",
                    "synthetic",
                )
            )
        ):

            synthetic = prediction

        # ----------------------------------------------------
        # Real / authentic labels
        # ----------------------------------------------------

        if (
            real is None
            and any(
                word in label_lower
                for word in (
                    "real",
                    "human",
                    "authentic",
                )
            )
        ):

            real = prediction

    # ========================================================
    # CALCULATE RISK
    # ========================================================

    if synthetic is not None:

        risk = synthetic["score"]

    elif real is not None:

        risk = 1.0 - real["score"]

    else:

        raise HTTPException(
            status_code=502,
            detail={
                "message": (
                    "The model returned labels "
                    "this API does not recognise."
                ),
                "predictions": predictions,
            },
        )

    # --------------------------------------------------------
    # Clamp risk between 0 and 1
    # --------------------------------------------------------

    risk = max(
        0.0,
        min(
            1.0,
            float(risk),
        ),
    )

    # ========================================================
    # USER-FACING RESULT
    # ========================================================

    if risk >= 0.50:

        label = (
            "Likely AI-generated or manipulated"
        )

    else:

        label = (
            "No strong AI-generation signal"
        )

    # ========================================================
    # RESPONSE
    # ========================================================

    return {
        "risk": risk,
        "score": risk,
        "percentage": round(
            risk * 100,
            2,
        ),
        "label": label,
        "model": (
            "Smogy/SMOGY-Ai-images-detector"
        ),
        "mode": "local",
        "device": str(DEVICE),
        "predictions": predictions,
        "disclaimer": (
            "This is a screening signal, "
            "not proof that a document or "
            "image is authentic."
        ),
    }


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api.index:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )