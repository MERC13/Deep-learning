# GAN test

PyTorch GAN on CIFAR-10. Trains a simple MLP+Conv generator and a conv discriminator, saving samples and weights.

## Setup

```powershell
python -m venv .venv; . .venv\Scripts\Activate.ps1
pip install torch torchvision matplotlib numpy
```

## Data

CIFAR-10 is auto-downloaded to `data/` on first run.

## Train

```powershell
python train.py
```

Artifacts:
- `generated_images_epoch_*.png`
- `generator.pth`, `discriminator.pth`
