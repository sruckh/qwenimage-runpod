#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Update package lists and install necessary tools
apt-get update
apt-get install -y git curl software-properties-common python3-apt # Added curl and python3-apt

# Add deadsnakes PPA for Python 3.10
add-apt-repository ppa:deadsnakes/ppa -y # Added -y for non-interactive
apt-get update

# Install Python 3.10 and venv
apt-get install -y python3.10 python3.10-dev python3.10-venv

# Set Python 3.10 as default python3
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
update-alternatives --set python3 /usr/bin/python3.10

# Install pip for Python 3.10
python3.10 -m ensurepip --default-pip # Use ensurepip to install pip
python3.10 -m pip install --upgrade pip # Upgrade pip

# Clone the repository into a subdirectory
if [ ! -d "/workspace" ]; then
  mkdir /workspace
fi
git clone https://github.com/sruckh/qwenimage-runpod.git /workspace/qwenimage-runpod
cd /workspace/qwenimage-runpod

# Install PyTorch
python3.10 -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126

# Install Python dependencies
python3.10 -m pip install --no-cache-dir -r requirements.txt
python3.10 -m pip install git+https://github.com/huggingface/diffusers

pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Execute the application
python3.10 /workspace/qwenimage-runpod/app.py