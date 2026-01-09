# 🏥 TriFetch AI: RLHF Control Room

An Online RLHF (Reinforcement Learning from Human Feedback) Workbench for medical AI. This tool simulates the process of a medical expert ranking model outputs and calculates the optimization updates (DPO Loss and GRPO Advantages) required to steer the model.

## Quick Start

### 1. Install Dependencies

```sh
pip install -r requirements.txt
```

### 2. Run the App

```sh
streamlit run app.py
```

### 3. Use the Workbench

1. **Select Model** — Choose a model from the sidebar
2. **Select Case** — Pick a patient case (1-5)
3. **Generate Traces** — Click to generate 3 reasoning traces
4. **Rank Traces** — Act as the doctor: rank Best, Middle, Worst
5. **Compute Loss** — Click to calculate DPO Loss and GRPO Advantages

Use **Clear All** in the sidebar to reset.

## Configuration

All settings are in `config.yaml`:

```yaml
# Change default model (one line switch)
default: "qwen-0.5b"

# DPO hyperparameter
dpo:
  beta: 0.1

# Available models
models:
  smollm-135m:
    name: "HuggingFaceTB/SmolLM-135M-Instruct"
    description: "SmolLM 135M (Fast)"
  qwen-0.5b:
    name: "Qwen/Qwen2-0.5B-Instruct"
    description: "Qwen2 0.5B (Best)"
```

### Adding a New Model

1. Find the model on [HuggingFace](https://huggingface.co/models)
2. Add it to `config.yaml`:

```yaml
models:
  my-new-model:
    name: "organization/model-name"
    description: "My New Model"
```

1. Set it as default: `default: "my-new-model"`

## Project Structure

```
├── app.py           # Streamlit UI
├── sampler.py       # Model interface & trace generation
├── optimizer.py     # DPO & GRPO calculations
├── config.yaml      # Model & hyperparameter settings
├── sample1-5.json   # Patient cases
├── requirements.txt # Dependencies
└── README.md
```

## Optimization Algorithms

### DPO (Direct Preference Optimization)

Calculates how much to adjust the model based on preferred vs rejected traces:

```
Loss = -log(sigmoid(β * (policy_margin - reference_margin)))
```

### GRPO (Group Relative Policy Optimization)

Normalizes rewards across the group of traces:

```
Advantage = (reward - mean) / std
```

## Requirements

- Python 3.8+
- About 1GB disk space (for model weights)
- Works on CPU, MPS (Mac), or CUDA (GPU)