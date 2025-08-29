#!/bin/bash

# Install git
apt-get update && apt-get install -y git

# Install Python 3.10 and pip
apt-get install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt-get update
apt-get install -y python3.10 python3.10-dev python3.10-venv
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
update-alternatives --set python3 /usr/bin/python3.10
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Clone the repository
git clone https://github.com/sruckh/qwenimage-runpod.git /workspace
cd /workspace

# Install PyTorch
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126

# Install Python dependencies
pip install --no-cache-dir -r requirements.txt
pip install git+https://github.com/huggingface/diffusers

# Install flash_attn wheel
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Execute the application
python /workspace/app.py