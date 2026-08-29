import io
import os
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
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models" / "ai-detector"


# ============================================================
# CONFIGURATION
# ============================================================

MAX_UPLOAD_BYTES = 4 * 1024 * 1024

# Render backend uses CPU.
DEVICE = torch.device("cpu")


# ============================================================
# GLOBAL MODEL
# ============================================================

processor = None
model = None


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    global processor
    global model

    print("=" * 70)
    print("PramaanSetu AI Detector")
    print("=" * 70)

    print(f"Project directory: {BASE_DIR}")
    print(f"Model directory: {MODEL_DIR}")
    print(f"Device: {DEVICE}")

    if not MODEL_DIR.exists():
        raise RuntimeError(
            f"Model directory not found: {MODEL_DIR}"
        )

    print("Loading image processor...")

    processor = AutoImageProcessor.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )

    print("Loading AI model...")

    model = AutoModelForImageClassification.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )

    model.to(DEVICE)
    model.eval()

    print("=" * 70)
    print("Local AI detector loaded successfully.")
    print("=" * 70)


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    load_model()

    yield


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="PramaanSetu AI Screening API",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    # Vercel frontend can call this API.
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
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "service": "PramaanSetu AI Screening API",
        "status": "online",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {
        "ready": model is not None,
        "model": "Smogy/SMOGY-Ai-images-detector",
        "mode": "local",
        "device": "cpu",
    }


# ============================================================
# DETECTION
# ============================================================

@app.post("/api/detect")
async def detect(
    file: UploadFile = File(...),
):

    if model is None or processor is None:

        raise HTTPException(
            status_code=503,
            detail="AI detector is not loaded.",
        )

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):

        raise HTTPException(
            status_code=415,
            detail="Please upload an image file.",
        )

    # --------------------------------------------------------
    # Read file
    # --------------------------------------------------------

    try:

        image_bytes = await file.read()

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail="Could not read uploaded image.",
        ) from error

    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    # --------------------------------------------------------
    # Size check
    # --------------------------------------------------------

    if len(image_bytes) > MAX_UPLOAD_BYTES:

        raise HTTPException(
            status_code=413,
            detail="Image must be 4 MB or smaller.",
        )

    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file.",
        ) from error

    # --------------------------------------------------------
    # AI inference
    # --------------------------------------------------------

    try:

        inputs = processor(
            images=image,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )[0]

        predictions = []

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

        predictions.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

    except Exception as error:

        print(
            "Inference error:",
            repr(error),
        )

        raise HTTPException(
            status_code=500,
            detail="AI detection failed.",
        ) from error

    # ========================================================
    # INTERPRET RESULT
    # ========================================================

    synthetic = None
    real = None

    for prediction in predictions:

        label = prediction["label"].lower()

        if synthetic is None and any(
            word in label
            for word in [
                "ai",
                "fake",
                "generated",
                "synthetic",
            ]
        ):

            synthetic = prediction

        if real is None and any(
            word in label
            for word in [
                "real",
                "human",
                "authentic",
            ]
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
                "message": "Unknown model labels.",
                "predictions": predictions,
            },
        )

    risk = max(
        0.0,
        min(
            1.0,
            float(risk),
        ),
    )

    if risk >= 0.50:

        label = "Likely AI-generated or manipulated"

    else:

        label = "No strong AI-generation signal"

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
        "model": "Smogy/SMOGY-Ai-images-detector",
        "mode": "local",
        "device": "cpu",
        "predictions": predictions,
        "disclaimer": (
            "This is a screening signal, not proof "
            "that a document or image is authentic."
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