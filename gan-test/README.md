# GAN test

PyTorch GAN on CIFAR-10. Trains a simple MLP+Conv generator and a conv discriminator, saving samples and weights.

## Setup (PowerShell)

```powershell
python -m venv .venv; . .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you prefer to install core packages manually, select the correct PyTorch wheel for your platform/CUDA version. If activation fails due to execution policy, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
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
