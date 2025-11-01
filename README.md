# Deep-learning

A collection of machine learning and deep learning experiments and mini-projects. Each subfolder is intended to be self-contained. This top-level README gives a short map, quick setup hints, and links to subproject READMEs.

## Quick project map

- `Agents/` — small agent examples and a research-assistant pipeline (LangChain, LangGraph, AutoGen demos).
- `atariRL/` — Deep Q-Network training for Atari games (PyTorch + Gymnasium).
- `fantasy-football-extension/` — transformer models + Flask API and browser extension for fantasy football overlays.
- `GAN test/` — PyTorch GAN (CIFAR-10).
- `GPT2/` — small GPT-2 experiments and text generation examples.
- `gymtest/` — simple RL / Q-learning demos (CartPole examples).
- `learning-LLM/` — small utilities and experiments for LLM fine-tuning / LoRA.
- `LSTMstock/` — LSTM forecasting (TensorFlow/Keras).
- `Rankings Predictor/` — Keras model for tournament ranking prediction.
- `research-assistant/` — PDF ingestion → embeddings → LLM summarization pipeline.
- `SNN/` — spiking neural network experiments.
- `stockproject/` — training pipeline + small Flask app for stock predictions.
- `X-ray CNN/` — transfer learning (VGG16) for chest X-ray classification.

For more details, open the `README.md` inside each project folder.

## Recommended setup

Use a separate virtual environment per project. Python 3.10+ is recommended for most folders. From PowerShell (example):

```powershell
# create and activate a venv
python -m venv .venv; . .venv\Scripts\Activate.ps1

# install the project's dependencies
pip install -r <project>/requirements.txt
```

Replace `<project>` with the folder name (for example, `atariRL` or `Agents/LangChainAgents`). Some projects require extra system packages (CUDA, ffmpeg, etc.) — see the project README.

## Contributing

- File issues for bugs or feature requests.
- When submitting PRs: keep changes limited to one project when possible and include a short test or example demonstrating the change.
- Follow the existing code style (4-space indent). Run linters/tests in the subproject if present.

## Data and artifacts

Large datasets, logs, and model weights are intentionally git-ignored. Check each project's `.gitignore` and `data/` or `saved_models/` directories for expected artifact locations. If a project requires external downloads, the project README will usually include links or scripts.

## Contact / Questions

If something is unclear, open an issue with the project name and a short reproduction step. Maintainers will triage according to available context.
