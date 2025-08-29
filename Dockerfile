FROM nvcr.io/nvidia/cuda:12.6.0-runtime-ubuntu24.04

WORKDIR /

ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COPY bootstrap.sh .

CMD ["./bootstrap.sh"]