import io
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoImageProcessor, AutoModelForImageClassification


# ============================================================================
# Environment
# ============================================================================

load_dotenv()


# ============================================================================
# Paths
# ============================================================================

# index.py is inside /api
# The project root is therefore one directory above it.
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models" / "ai-detector"


# ============================================================================
# Configuration
# ============================================================================

# Vercel Functions have request-size limitations.
# Keep this below the platform limit.
MAX_UPLOAD_BYTES = 4 * 1024 * 1024


# Automatically use GPU if CUDA is available.
# Otherwise use CPU.
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================================
# Global model objects
# ============================================================================

processor = None
model = None


# ============================================================================
# Model loading
# ============================================================================

def load_local_model():
    """
    Load the AI detector from the model files stored in:

        /models/ai-detector
    """

    global processor, model

    if not MODEL_DIR.exists():
        raise RuntimeError(
            f"Local model directory was not found:\n{MODEL_DIR}"
        )

    print("=" * 70)
    print("PramaanSetu AI Detector")
    print("=" * 70)
    print(f"Model directory: {MODEL_DIR}")
    print(f"Using device: {DEVICE}")

    try:
        # Load image processor from local files only.
        processor = AutoImageProcessor.from_pretrained(
            str(MODEL_DIR),
            local_files_only=True,
        )

        # Load model from local files only.
        model = AutoModelForImageClassification.from_pretrained(
            str(MODEL_DIR),
            local_files_only=True,
        )

        # Move model to CPU/GPU.
        model.to(DEVICE)

        # Evaluation mode.
        model.eval()

        print("Local AI detector loaded successfully.")
        print("=" * 70)

    except Exception as error:
        processor = None
        model = None

        print("ERROR: Failed to load local AI detector.")
        print(repr(error))

        raise RuntimeError(
            f"Failed to load the local AI detector: {error}"
        ) from error


# ============================================================================
# FastAPI lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Load the model when the FastAPI application starts.
    """

    load_local_model()

    yield


# ============================================================================
# FastAPI application
# ============================================================================

app = FastAPI(
    title="PramaanSetu AI Screening",
    description="AI-generated image screening API for PramaanSetu",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


# ============================================================================
# CORS
# ============================================================================

# During initial deployment, allow the Vercel frontend
# and local development to communicate with the API.
#
# Once everything is working, this can be restricted
# to your exact Vercel domain.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ============================================================================
# Health check
# ============================================================================

@app.get("/api/health")
def health():
    """
    Basic API and model health check.
    """

    return {
        "ready": model is not None,
        "model": "Smogy/SMOGY-Ai-images-detector",
        "mode": "local",
        "device": str(DEVICE),
    }


# ============================================================================
# AI image detection
# ============================================================================

@app.post("/api/detect")
async def detect_ai_generated_image(
    file: UploadFile = File(...)
):
    """
    Analyze an uploaded image using the locally installed
    AI detector.

    Returns a screening signal.

    This result must NOT be treated as definitive proof
    that an image is authentic or AI-generated.
    """

    # ------------------------------------------------------------------------
    # Check model availability
    # ------------------------------------------------------------------------

    if model is None or processor is None:
        raise HTTPException(
            status_code=503,
            detail="The local AI detector is not loaded yet.",
        )

    # ------------------------------------------------------------------------
    # Validate file type
    # ------------------------------------------------------------------------

    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=415,
            detail="Please upload a JPG, PNG, or other image file.",
        )

    # ------------------------------------------------------------------------
    # Read uploaded image
    # ------------------------------------------------------------------------

    try:
        image_bytes = await file.read()
    except Exception as error:
        print(f"File reading failed: {error!r}")

        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded file.",
        ) from error

    # ------------------------------------------------------------------------
    # Validate empty file
    # ------------------------------------------------------------------------

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The selected image is empty.",
        )

    # ------------------------------------------------------------------------
    # Validate file size
    # ------------------------------------------------------------------------

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Images must be 4 MB or smaller.",
        )

    # ------------------------------------------------------------------------
    # Open image
    # ------------------------------------------------------------------------

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
            detail="The uploaded file is not a valid image.",
        ) from error

    # ------------------------------------------------------------------------
    # Run model inference
    # ------------------------------------------------------------------------

    try:

        # Convert image to model input.
        inputs = processor(
            images=image,
            return_tensors="pt",
        )

        # Move tensors to CPU/GPU.
        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        # Run inference without calculating gradients.
        with torch.inference_mode():
            outputs = model(**inputs)

        # Convert logits into probabilities.
        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )[0]

        predictions = []

        # Convert model output into a clean JSON-compatible structure.
        for index, score in enumerate(probabilities):

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
            f"Local model inference failed: {error!r}"
        )

        raise HTTPException(
            status_code=500,
            detail="The local AI detector failed while reviewing this image.",
        ) from error

    # ------------------------------------------------------------------------
    # Interpret model labels
    # ------------------------------------------------------------------------

    synthetic = None
    real = None

    for prediction in predictions:

        label_lower = prediction["label"].lower()

        # Look for AI/synthetic/fake labels.
        if synthetic is None and any(
            word in label_lower
            for word in (
                "ai",
                "fake",
                "generated",
                "synthetic",
            )
        ):
            synthetic = prediction

        # Look for real/authentic labels.
        if real is None and any(
            word in label_lower
            for word in (
                "real",
                "human",
                "authentic",
            )
        ):
            real = prediction

    # ------------------------------------------------------------------------
    # Calculate AI-generation risk
    # ------------------------------------------------------------------------

    if synthetic is not None:

        risk = synthetic["score"]

    elif real is not None:

        risk = 1.0 - real["score"]

    else:

        raise HTTPException(
            status_code=502,
            detail={
                "message": (
                    "The model returned labels this API "
                    "does not recognise."
                ),
                "predictions": predictions,
            },
        )

    # ------------------------------------------------------------------------
    # Keep risk between 0 and 1
    # ------------------------------------------------------------------------

    risk = max(
        0.0,
        min(
            1.0,
            float(risk),
        ),
    )

    # ------------------------------------------------------------------------
    # User-facing result
    # ------------------------------------------------------------------------

    if risk >= 0.50:

        label = "Likely AI-generated or manipulated"

    else:

        label = "No strong AI-generation signal"

    # ------------------------------------------------------------------------
    # API response
    # ------------------------------------------------------------------------

    return {
        "risk": risk,
        "score": risk,
        "percentage": round(
            risk * 100,
            2,
        ),
        "label": label,
        "model": "Smogy/SMOGY-Ai-images-detector",
        "mode": "local",
        "device": str(DEVICE),
        "predictions": predictions,
        "disclaimer": (
            "This is a screening signal, not proof that a "
            "document or image is authentic."
        ),
    }


# ============================================================================
# Local development
# ============================================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api.index:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )