# Football Question Answering Model

A specialized question-answering model fine-tuned on football (soccer) data using the Llama 3.1 8B Instruct model with Unsloth optimization.

## Overview

This project fine-tunes the Meta Llama 3.1 8B Instruct model on a football question-answering dataset to create a specialized AI that can accurately answer questions about football matches, scores, players, and other football-related information.

Key features:
- Leverages Unsloth for 2x faster training and inference
- Uses QLoRA (4-bit quantization with Low-Rank Adaptation) for efficient fine-tuning
- Optimized for football-specific question answering

## Requirements

- Python 3.8+
- PyTorch
- CUDA-compatible GPU
- Unsloth library
- Transformers library
- TRL (Transformer Reinforcement Learning)
- Datasets

## Installation

```bash
pip install unsloth
pip install transformers datasets trl
```

For the latest Unsloth features:

```bash
pip uninstall unsloth -y
pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
```

## Usage

### Loading the Fine-Tuned Model

```python
from unsloth import FastLanguageModel
from transformers import TextStreamer

# Load the model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model",  # Path to your saved model
    max_seq_length = 2048,
    load_in_4bit = True,
)

# Enable faster inference
FastLanguageModel.for_inference(model)

# Prepare user question
messages = [
    {"role": "user", "content": "What was the result of the match between Barcelona and Real Madrid in La Liga 2022/2023?"}
]

# Format input
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt"
).to("cuda")

# Generate response with streaming
text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    input_ids = inputs,
    streamer = text_streamer,
    max_new_tokens = 128,
    use_cache = True,
    temperature = 1.5,
    min_p = 0.1
)
```

## Training Details

### Dataset

This model was trained on the "StaAhmed/Football_Question_Answers" dataset, which contains:
- Football questions
- Corresponding accurate answers
- Coverage of various leagues, teams, and competitions

### Training Configuration

- **Base Model**: Meta-Llama-3.1-8B-Instruct
- **Quantization**: 4-bit (QLoRA)
- **LoRA Parameters**:
  - Rank: 16
  - Alpha: 16
  - Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Training Parameters**:
  - Batch Size: 2
  - Gradient Accumulation Steps: 4
  - Learning Rate: 2e-4
  - Scheduler: Linear
  - Training Steps: 60
  - Seed: 3407

## Model Performance

The model is specialized in answering football-related questions, including:
- Match results between specific teams
- Tournament outcomes
- Team/player statistics
- Historical football data

Example questions the model can answer:
- "What was the result of the match between Barcelona and Real Madrid in La Liga 2022/2023?"
- "Who won the Premier League in 2020/2021?"
- "How many goals did Liverpool score in the 2020/2021 season?"

## Limitations

- The model's knowledge is limited to the football data it was trained on
- May not have complete information about very recent matches or events
- Best suited for factual football questions rather than subjective analysis

## Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth) for optimization techniques
- Meta for the Llama 3.1 base model
- StaAhmed for the Football Question Answers dataset