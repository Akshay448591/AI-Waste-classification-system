from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = FastAPI(title="E-Waste Classification API")

# ---------------------------
# ✅ CORS (VERY IMPORTANT for Streamlit)
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Load model ONCE at startup
# ---------------------------
model = load_model("model.h5")

# TrashNet classes
classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# ---------------------------
# Image preprocessing
# ---------------------------
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ---------------------------
# Routes
# ---------------------------
@app.get("/")
def home():
    return {"message": "E-Waste Classification API Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        processed_image = preprocess_image(image)

        preds = model.predict(processed_image)
        class_index = int(np.argmax(preds))
        confidence = float(np.max(preds))

        return {
            "class": classes[class_index],
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}