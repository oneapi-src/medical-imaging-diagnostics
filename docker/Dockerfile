ARG BASE_NAME=continuumio/miniconda3
ARG BASE_TAG=22.11.1
FROM ${BASE_NAME}:${BASE_TAG} as devkitminimal

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends --fix-missing \
    gcc \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    numactl

SHELL ["/bin/bash", "-c"] 

ENV CONDA_VERSION=23.5.0

ENV LISTEN_IP=0.0.0.0

RUN conda install -n base conda==${CONDA_VERSION} && \
    conda install -n base conda-libmamba-solver && \
    conda config --set solver libmamba && \
    conda init --system bash
    

FROM devkitminimal as medical-image-diag

WORKDIR /workspace
COPY env env
COPY docker/envinstall_all.sh envinstall_all.sh
RUN bash envinstall_all.sh

RUN sed -i "s/conda\ activate\ base/conda\ activate\ medical-image-diag-tf/" ~/.bashrc
RUN groupadd -r user && useradd -r -g user -m user 
USER $USER