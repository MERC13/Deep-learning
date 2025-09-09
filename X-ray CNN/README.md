# X-ray CNN

Transfer learning with VGG16 for pneumonia detection on the Chest X-Ray dataset.

After 3 epochs of 326 images:

![image](https://github.com/user-attachments/assets/1aaceac2-2c8c-4820-8faf-0e76db43354c)

## Setup

```powershell
python -m venv .venv; . .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

Place dataset under `chest_xray/chest_xray/` with subfolders `train/`, `val/`, and `test/`.

## Train

```powershell
python train.py
```
