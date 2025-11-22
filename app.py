# api.py
import io
import base64
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import torch
import matplotlib
matplotlib.use("Agg")  # non-GUI backend
import matplotlib.pyplot as plt

from helper_lib import (
    get_model, load_model,
    generate_diffusion_samples,
    generate_ebm_samples
)

app = FastAPI(title="CIFAR-10 Generative Models API")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Load models at startup ---
diffusion_model = get_model('Diffusion', input_channels=3)
diffusion_model = load_model(diffusion_model, path="models/diffusion_mnist.pth")
diffusion_model.to(DEVICE)

ebm_model = get_model('EBM', input_channels=3)
ebm_model = load_model(ebm_model, path="models/ebm_cifar10.pth")
ebm_model.to(DEVICE)


def _figure_to_base64():
    """Helper: take current matplotlib figure and return base64 PNG string."""
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_bytes = buf.read()
    plt.close()
    return base64.b64encode(img_bytes).decode("utf-8")


@app.get("/")
def root():
    return {"message": "CIFAR-10 Diffusion + EBM API is running."}


@app.get("/generate/diffusion")
def generate_diffusion_endpoint(num_samples: int = 16):
    generate_diffusion_samples(
        diffusion_model,
        device=DEVICE,
        num_samples=num_samples,
        diffusion_steps=100
    )
    img_b64 = _figure_to_base64()
    return JSONResponse({"model": "diffusion", "num_samples": num_samples, "image_base64": img_b64})


@app.get("/generate/ebm", response_class=StreamingResponse)
def generate_ebm_endpoint():
    samples = generate_ebm_samples(...)
    buf = io.BytesIO()
    samples.save(buf, format='PNG')
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")