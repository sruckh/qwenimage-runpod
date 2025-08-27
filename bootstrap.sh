#!/bin/bash

echo "Starting bootstrap script..."

echo "Installing Python dependencies from requirements.txt..."
pip install --no-cache-dir -r requirements.txt

echo "Installing flash_attn wheel..."
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.2cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

echo "Running app.py..."
python app.py
