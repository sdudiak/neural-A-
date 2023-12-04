FROM jupyter/base-notebook:latest

# Run as root
USER root

# Install apt dependencies
RUN apt update && \
    apt install -y \
    git \
    # opencv dependencies
    ffmpeg \
    libsm6 \
    libxext6

COPY requirements.txt /tmp/requirements.txt
RUN conda install -c anaconda pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

USER user


