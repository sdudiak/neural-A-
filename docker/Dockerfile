FROM jupyter/base-notebook:latest

# Run as root
USER root

# Install apt dependencies
RUN apt update && \
    apt install -y \
    git 

# Setup pip with conda
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


RUN conda install -c anaconda pip && \
    pip install \
    matplotlib \
    opencv-python \ 
    numpy \
    sphinx \
    black \
    lightning \
    segmentation-models-pytorch


