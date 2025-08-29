#!/bin/bash

# Install git
apt-get update && apt-get install -y git

# Clone the repository
git clone https://github.com/sruckh/qwenimage-runpod.git /workspace
cd /workspace

# Install Python dependencies
pip install --no-cache-dir -r requirements.txt
pip install git+https://github.com/huggingface/diffusers

# Patch diffusers to remove xpu reference
sed -i 's/"xpu": torch.xpu.device_count,/\/\/"xpu": torch.xpu.device_count,/g' /usr/local/lib/python3.10/dist-packages/diffusers/utils/torch_utils.py
sed -i 's/"xpu": torch.xpu.empty_cache,/\/\/"xpu": torch.xpu.empty_cache,/g' /usr/local/lib/python3.10/dist-packages/diffusers/utils/torch_utils.py

pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Execute the application
python /workspace/app.py