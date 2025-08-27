#!/bin/bash

echo "Starting bootstrap script..."

# Clone the repository
echo "Cloning qwenimage-runpod repository..."
git clone git@github.com:QwenLM/Qwen-Image.git /workspace/Qwen-Image
cd /workspace/Qwen-Image

# Install Python dependencies
echo "Installing Python dependencies from requirements.txt..."
pip install git+https://github.com/huggingface/diffusers
pip install --no-cache-dir -r requirements.txt

# Install flash_attn wheel
echo "Installing flash_attn wheel..."
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.2cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

# Execute the application
echo "Running app.py..."
python app.py
