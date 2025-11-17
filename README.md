# Conversational Chatbot

A GPT-based conversational chatbot trained on the TinyChat dataset using PyTorch. This project implements a custom GPT model for text generation and conversation.

## Features

- Custom GPT model implementation with configurable parameters
- Trained on the TinyChat dataset for conversational responses
- Supports text generation with context-aware prompting
- CUDA acceleration for faster training and inference

## Project Structure

- `components/model.py` - GPT model implementation
- `components/dataset.py` - Dataset handling and preprocessing
- `components/tokenizer.py` - Text encoding and decoding utilities
- `test_chat.ipynb` - Jupyter notebook for model testing and text generation
- `train_script_3.py` - Training script for the model

## Setup and Installation

This project uses `uv` for fast Python package management. To set up the project:

1. Install `uv` if you don't have it:
   ```bash
   pip install uv
   ```

2. Clone the repository:
   ```bash
   git clone <repository-url>
   cd chatbot
   ```

3. Install dependencies using uv:
   ```bash
   uv sync
   ```

## Running the Model

To run the chatbot model using the test notebook:

1. Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Run the test notebook with uv:
   ```bash
   uv run jupyter notebook test_chat.ipynb
   ```

   Or if you prefer to run it directly:
   ```bash
   uv run python -m notebook test_chat.ipynb
   ```

## Model Configuration

The model uses the following hyperparameters (configurable in `test_chat.ipynb`):
- Block size: 128
- Number of layers: 16
- Number of attention heads: 8
- Embedding dimension: 256
- Dropout probability: 0.1
- Batch size: 8
- Learning rate: 3e-4
- Maximum iterations: 5000

## Usage

The model can generate conversational responses. To force a conversational flow, format your prompt like this:
```
Prompt: How are you doing [/INST] I am good. [INST] Why?
```

This will make the model generate another response following the previous exchange.

## Dependencies

- torch>=2.9.0
- transformers>=4.57.1
- datasets>=4.2.0
- tiktoken>=0.12.0
- gradio>=5.49.1
- accelerate>=1.11.0
- trl>=0.24.0

## Demo

Check out the live demo: https://huggingface.co/spaces/flappybird1084/chatbot
