FROM jupyter/base-notebook:latest

# Run as root
USER root

# Install apt dependencies
RUN apt update && \
    apt install -y \
    git \
    unzip \
    wget \
    # opencv dependencies
    ffmpeg \
    libsm6 \
    libxext6

COPY requirements.txt /tmp/requirements.txt

RUN conda install -c anaconda pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt


WORKDIR /workspaces
RUN  mkdir -p datasets/raw/street && \
    wget https://www.movingai.com/benchmarks/street/street-png.zip -P datasets/raw/street/ && \
    unzip datasets/raw/street/street-png.zip -d datasets/raw/street/ && \
    rm datasets/raw/street/street-png.zip

    


