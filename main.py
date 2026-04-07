import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import onnxruntime as rt
import uvicorn

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="AgriVet Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Class labels ──────────────────────────────────────────────────────────────

VALIDATOR_CLASSES = ["chicken", "dropping", "other"]
DISEASE_CLASSES   = ["cocci", "healthy", "ncd", "salmo"]

# ── Model paths ───────────────────────────────────────────────────────────────

VALIDATOR_MODEL_PATH = "./models/validator.onnx"
DISEASE_MODEL_PATH   = "./models/classification.onnx"

# ── Singleton session holders ─────────────────────────────────────────────────
validator_session: rt.InferenceSession | None = None
disease_session:   rt.InferenceSession | None = None


def get_validator_session() -> rt.InferenceSession:
    global validator_session
    if validator_session is None:
        print("Loading validator ONNX model...")
        validator_session = rt.InferenceSession(
            VALIDATOR_MODEL_PATH,
            providers=["CPUExecutionProvider"],
        )
        print("Validator model loaded ✓")
    return validator_session


def get_disease_session() -> rt.InferenceSession:
    global disease_session
    if disease_session is None:
        print("Loading disease ONNX model...")
        disease_session = rt.InferenceSession(
            DISEASE_MODEL_PATH,
            providers=["CPUExecutionProvider"],
        )
        print("Disease model loaded ✓")
    return disease_session


# ── Shared image preprocessing ────────────────────────────────────────────────

def preprocess_image(image_bytes: bytes, size: int = 224) -> np.ndarray:
    """Decode bytes → RGB PIL image → normalised float32 (1, H, W, 3) array."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)          # shape: (1, 224, 224, 3)


def run_inference(session: rt.InferenceSession, tensor: np.ndarray) -> list[float]:
    """Run a single forward pass and return scores as a plain list."""
    input_name = session.get_inputs()[0].name
    outputs    = session.run(None, {input_name: tensor})
    return outputs[0][0].tolist()               # first output, first batch item


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── /validate endpoint ────────────────────────────────────────────────────────

@app.post("/validate")
async def validate_image(file: UploadFile = File(...)):
    """
    Accepts a JPEG/PNG image and returns whether it is
    'chicken', 'dropping', or 'other'.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images are accepted.")

    image_bytes = await file.read()

    try:
        tensor  = preprocess_image(image_bytes)
        session = get_validator_session()
        scores  = run_inference(session, tensor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    top_idx = int(np.argmax(scores))
    return {
        "label":      VALIDATOR_CLASSES[top_idx],
        "confidence": round(scores[top_idx] * 100, 2),
        "scores":     {cls: round(s * 100, 2) for cls, s in zip(VALIDATOR_CLASSES, scores)},
    }


# ── /classify endpoint ────────────────────────────────────────────────────────
@app.post("/classify")
async def classify_disease(file: UploadFile = File(...)):
    """
    Accepts a JPEG/PNG image and returns the predicted disease:
    'cocci', 'healthy', 'ncd', or 'salmo'.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images are accepted.")

    image_bytes = await file.read()

    try:
        tensor  = preprocess_image(image_bytes)
        session = get_disease_session()
        scores  = run_inference(session, tensor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    top_idx = int(np.argmax(scores))
    return {
        "label":      DISEASE_CLASSES[top_idx],
        "confidence": round(scores[top_idx] * 100, 2),
        "scores":     {cls: round(s * 100, 2) for cls, s in zip(DISEASE_CLASSES, scores)},
    }

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)