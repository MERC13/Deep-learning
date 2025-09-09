# Deep-learning

A collection of machine learning and deep learning experiments and mini-projects. Each subfolder is self-contained. This README gives a quick map and how to run each project.

## Projects

- GAN test: PyTorch GAN on CIFAR-10. Trains a simple generator/discriminator and saves samples and weights.
- gymtest: Q-learning on CartPole-v1 using Gym. Renders to a window.
- LSTMstock: LSTM forecasting using TensorFlow/Keras and yfinance. Includes a Dockerfile.
- Rankings Predictor: Keras model predicting team win counts from tournament pairings.
- research-assistant: Data ingestion/parsing utilities and artifacts (docs WIP).
- stockproject: Stock prediction training pipeline plus a simple Flask app to serve predictions.
- X-ray CNN: Transfer-learning with VGG16 for pneumonia detection from chest X-rays.

## Setup

Use a separate virtual environment per project. Python 3.10 recommended where possible.

```bash
# Windows PowerShell
python -m venv .venv; . .venv/Scripts/Activate.ps1
pip install -r <project>/requirements.txt
```

See each project's README for specific steps.

## Conventions

- Code style: 4-space indentation, LF line endings (see .editorconfig)
- Artifacts: data/, logs/, and model files are git-ignored by default
- Repro: Each project has a requirements.txt with minimal dependencies

## Notes

- Large datasets and trained weights are ignored; you may need to re-download or retrain.
- GPU-accelerated packages (TensorFlow/PyTorch) may require platform-specific installs; adjust versions if needed.
