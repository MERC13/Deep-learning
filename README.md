# Deep Learning & AI Portfolio

A collection of machine learning, deep learning, and AI engineering projects, ranging from LLM safety research to reinforcement learning and full-stack AI applications.

## 🏆 Featured Projects

### 1. [Fantasy Football Extension](fantasy-football-extension/)
**Stack:** Python, Flask, Transformers, Chrome Extension
Transformer-powered fantasy football predictions with a browser extension overlay.
- **Key Features:** Position-specific Transformer models, real-time API, Chrome/Edge extension for Yahoo/ESPN.
- **Tech:** FT-Transformer, Temporal Transformer, Flask, JavaScript.

### 2. [AI Agents & Pipelines](agents/)
**Stack:** LangChain, LangGraph, AutoGen
A collection of agentic workflows and a research assistant pipeline.
- **Key Features:** Multi-agent chat (AutoGen), ReAct/RAG patterns (LangChain), and an end-to-end PDF research assistant.
- **Demos:** Automated research summaries, tool-using agents.

### 3. [Atari Reinforcement Learning](rl-atari/)
**Stack:** PyTorch, Gymnasium
Implementation of Deep Q-Networks (DQN) to master Atari games.
- **Key Features:** Experience replay, target networks, frame stacking.
- **Performance:** Beats human baseline on Breakout and Pong.

---

## 📂 Project Index

| Project | Description | Tech Stack |
|---------|-------------|------------|
| **[Agents](agents/)** | LangChain & AutoGen demos, research assistant pipeline. | LangChain, AutoGen |
| **[CNN X-Ray](cnn-xray/)** | Chest X-ray classification using Transfer Learning (VGG16). | PyTorch/Keras, CNN |
| **[Fantasy Football](fantasy-football-extension/)** | Browser extension & API for fantasy football overlays. | Flask, JS, Transformers |
| **[GAN Test](gan-test/)** | Generative Adversarial Network experiments on CIFAR-10. | PyTorch, GAN |
| **[GPT-2 Experiments](gpt2/)** | Text generation and fine-tuning experiments. | Transformers, PyTorch |
| **[Gym Test](gym-test/)** | Simple RL algorithms (Q-learning) on CartPole. | Gymnasium, RL |
| **[Learning LLM](learning-llm/)** | Utilities for LLM fine-tuning and LoRA. | PEFT, Transformers |
| **[LSTM Stock](lstm-stock/)** | Time-series forecasting experiments. | TensorFlow, LSTM |
| **[Mechanistic Interpretability](mechanistic-interpretability/)** | Transformer Lens and DPO experiments. | TransformerLens, PyTorch |
| **[Rankings Predictor](rankings-predictor/)** | Tournament ranking prediction model. | Keras, Pandas |
| **[SNN](snn/)** | Spiking Neural Network experiments. | SNN |

## 🛠️ General Setup

Most projects are self-contained. To get started:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/Deep-learning.git
    cd Deep-learning
    ```

2.  **Navigate to a project:**
    ```bash
    cd agentic-misalignment
    ```

3.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🤝 Contributing

Open to collaboration! Please open an issue or submit a PR for any improvements.
