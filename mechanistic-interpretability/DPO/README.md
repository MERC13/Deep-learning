# Direct Preference Optimization (DPO) for AI Alignment

This project implements Direct Preference Optimization (DPO) to address AI misalignment by training language models to prefer human-aligned responses over misaligned ones.

## What is DPO?

Direct Preference Optimization is a method for fine-tuning language models using preference data (preferred vs. rejected responses) without needing a separate reward model. It directly optimizes the policy to increase the likelihood of preferred responses while decreasing the likelihood of rejected ones.

## Step-by-Step Guide

### Step 1: Environment Setup
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Your Dataset
DPO requires preference pairs (chosen vs. rejected responses). You can use:
- Anthropic HH-RLHF dataset
- Stanford Human Preferences (SHP)
- Custom preference data

### Step 3: Configure Training
Edit `config/dpo_config.yaml` with your preferences:
- Model selection (e.g., GPT-2, LLaMA, Mistral)
- Training hyperparameters
- Dataset paths

### Step 4: Train the Model
```bash
python main.py --config config/dpo_config.yaml
```

### Step 5: Evaluate Results
The evaluation script will compare your DPO-trained model against the base model.

## Key Concepts

### DPO Loss Function
The DPO loss is defined as:
$$L_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x,y_w,y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

Where:
- $\pi_\theta$ is the policy being trained
- $\pi_{ref}$ is the reference policy (frozen base model)
- $y_w$ is the preferred (winning) response
- $y_l$ is the rejected (losing) response
- $\beta$ is the temperature parameter

### Why DPO Addresses Misalignment

1. **Direct optimization**: No need for a reward model that could itself be misaligned
2. **Preference-based**: Learns directly from human preferences
3. **Conservative**: The KL divergence term keeps the model close to the reference policy
4. **Interpretable**: The loss directly reflects preference strength

## Dataset Format

Your preference dataset should have the following structure:
```json
{
    "prompt": "User question or instruction",
    "chosen": "Preferred response (aligned)",
    "rejected": "Rejected response (misaligned)"
}
```

## Monitoring Training

The project integrates with Weights & Biases (wandb) to track:
- Training loss
- Preference accuracy
- KL divergence from reference model
- Sample generations

## References

- [DPO Paper](https://arxiv.org/abs/2305.18290): "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
- [TRL Library](https://github.com/huggingface/trl): Hugging Face's Transformer Reinforcement Learning library
