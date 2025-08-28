FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /

ENV PYTHONUNBUFFERED=1

COPY bootstrap.sh .

CMD ["./bootstrap.sh"]