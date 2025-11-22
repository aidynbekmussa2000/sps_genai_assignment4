# CIFAR-10 Generative Models (GAN, Diffusion, EBM) + FastAPI Deployment

This repository contains my **Assignment 4** for the *Applied Generative AI* course.
It includes implementations of:

- **DCGAN**
- **Denoising Diffusion Model (DDPM)**
- **Energy-Based Model (EBM)**
- A full **FastAPI server** exposing generation endpoints (`/generate/diffusion` and `/generate/ebm`)

All models run on **CIFAR-10 (32×32 RGB)** and produce samples using trained weights.

---

## 🚀 Project Structure

```
genAI_sps/
│
├── app.py                    # FastAPI app with /generate/* endpoints
├── Dockerfile                # Container for deployment
├── requirements.txt          # Python dependencies
│
├── helper_lib/               # Core library
│   ├── data_loader.py        # CIFAR-10 data loader
│   ├── model.py              # CNN, VAE, GAN, Diffusion, EBM architectures
│   ├── trainer.py            # Training loops for all models
│   ├── generator.py          # Sampling utilities (GAN, Diffusion, EBM)
│   ├── utils.py              # Save/load helpers
│   └── evaluator.py          # Classifier evaluation helper
│
├── models/                   # Trained weights (may be ignored by .gitignore)
│   ├── CNN_trained.pth
│   ├── diffusion_mnist.pth
│   ├── ebm_cifar10.pth
│   └── vae_latent20_trained.pth
│
├── data/                     # CIFAR-10 dataset (ignored in git)
│
└── assignment4.ipynb         # Full write‑up + model training + experiments
```

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/sps_genai_assignment4.git
cd sps_genai_assignment4
pip install -r requirements.txt
```

Torch + torchvision will download CIFAR‑10 automatically when running the code.

---

## ▶️ Running the FastAPI Server

Start the API locally:

```bash
uvicorn app:app --reload
```

Visit Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

## 📡 API Endpoints

### **Generate Diffusion Samples**
```
GET /generate/diffusion?num_samples=16
```

Returns base64 image grid.

### **Generate EBM Samples**
```
GET /generate/ebm?num_samples=16
```

---

## 🧠 Training

Example EBM training:

```python
from helper_lib import get_model, get_data_loader, train_ebm

loader = get_data_loader('./data', batch_size=64, dataset_type='cifar10')
model = get_model('EBM', input_channels=3)
train_ebm(model, loader, epochs=1)
```

Notebooks:
- `test_gan.ipynb`
- `test_diffusion_and_ebm.ipynb`
- `test_vae.ipynb`

---

## 📝 Notes for Graders

- Large `.pth` files and CIFAR‑10 data are excluded via `.gitignore` to avoid GitHub push errors.
- The project is fully reproducible with the provided training notebooks.
- FastAPI endpoints generate **actual images** via base64.

---
