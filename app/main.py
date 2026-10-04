from io import BytesIO
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, ImageOps
from torchvision import transforms

from app.model import LeNet

app = FastAPI(title="MNIST LeNet API", version="1.0")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("/workspace/model/lenet_mnist.pt")

model = LeNet().to(DEVICE)
if MODEL_PATH.exists():
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
else:
    model = None

preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.get("/")
def root():
    return {
        "message": "MNIST classifier is running",
        "device": str(DEVICE),
        "model_loaded": model is not None
    }

@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model file not found. Train the model first.")
    try:
        raw = await file.read()
        img = Image.open(BytesIO(raw)).convert("L")
        # If the image has a white background, invert it to resemble MNIST
        if sum(img.resize((1,1)).getdata()) > 127:
            img = ImageOps.invert(img)
        x = preprocess(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            confidence, digit = probs.max(dim=1)

        return {
            "digit": int(digit.item()),
            "confidence": round(float(confidence.item()), 4),
            "device": str(DEVICE)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")

from fastapi.responses import HTMLResponse

@app.get("/draw", response_class=HTMLResponse)
def draw_page():
    with open("/workspace/app/draw.html", "r", encoding="utf-8") as f:
        return f.read()
