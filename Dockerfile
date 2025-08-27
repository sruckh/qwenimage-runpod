FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /workspace

COPY bootstrap.sh .

ENV HF_TOKEN=""
ENV HF_HOME="/workspace/huggingface"

CMD ["bash", "bootstrap.sh"]