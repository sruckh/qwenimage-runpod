FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt .

COPY bootstrap.sh .


COPY . .

ENV HF_TOKEN=""
ENV HF_HOME="/workspace/huggingface"

CMD ["bash", "bootstrap.sh"]