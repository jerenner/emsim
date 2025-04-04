FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel
# MinkowskiEngine is incompatible with cuda 12.4 even with the edits below

WORKDIR /app
RUN useradd -ms /bin/bash user

ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
        git \
        build-essential \
        ninja-build \
        libopenblas-dev \
        # python3-dev \
    && apt-get clean all \
    && rm -rf /var/lib/apt/lists/*

RUN conda update conda mamba \
    && mamba install -c conda-forge -y \
        "sparse>=0.15" \
        hydra-core \
        h5py \
        cython \
        yaml \
        scipy \
        matplotlib \
        pandas \
        fire \
        deprecation \
        pytest \
        pytest-env \
        && mamba clean -ay
RUN pip install --no-cache-dir \
        lightning==2.4 \
        transformers==4.46 \
        timm==1.0.11 \
        tensorboardx \
        pycocotools \
        torchmetrics \
        stempy

# Could try just using the arch for A100
ARG TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6 8.9"
ARG TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

RUN git clone --recursive https://github.com/NVIDIA/MinkowskiEngine /app/MinkowskiEngine \
    && cd /app/MinkowskiEngine \
    # latest commit as of this writing
    && git checkout 02fc608 \
    # Adding these includes lets MinkowskiEngine compile against CUDA 12.1
    # See: https://github.com/NVIDIA/MinkowskiEngine/issues/543#issuecomment-1773458776
    && sed -i '31i\#include <thrust/execution_policy.h>' ./src/convolution_kernel.cuh \
    && sed -i '39i\#include <thrust/unique.h>\n#include <thrust/remove.h>' ./src/coordinate_map_gpu.cu \
    && sed -i '38i\#include <thrust/execution_policy.h>\n#include <thrust/reduce.h>\n#include <thrust/sort.h>' src/spmm.cu \
    && sed -i '30i\#include <thrust/execution_policy.h>' src/3rdparty/concurrent_unordered_map.cuh \
    && python setup.py install --force_cuda --blas=openblas \
    && cd - \
    && rm -rf /app/MinkowskiEngine

COPY entrypoint.sh /entrypoint.sh
RUN chmod 755 /entrypoint.sh
RUN chown user /entrypoint.sh
RUN chown -R user /app
ENTRYPOINT [ "./entrypoint.sh" ]
USER user

# This env var should be set during runtime. Pytorch warns that the setting "is not supported on this platform" but it still seems to work
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
