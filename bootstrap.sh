#!/bin/bash

# Install git
apt-get update && apt-get install -y git

# Clone the repository
git clone https://github.com/sruckh/qwenimage-runpod.git /workspace
cd /workspace

# Install Python dependencies
pip install --no-cache-dir -r requirements.txt
pip install git+https://github.com/huggingface/diffusers
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Execute the application
python /workspace/app.py