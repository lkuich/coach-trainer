# Build an image that can do training in SageMaker
FROM nvidia/cuda:9.0-base-ubuntu16.04

MAINTAINER Loren Kuich <loren@lkuich.com>

ENV NCCL_VERSION=2.3.5-2+cuda9.0
ENV CUDNN_VERSION=7.3.1.20-1+cuda9.0
ENV TF_TENSORRT_VERSION=4.1.2

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        cuda-command-line-tools-9-0 \
        cuda-cublas-9-0 \
        cuda-cufft-9-0 \
        cuda-curand-9-0 \
        cuda-cusolver-9-0 \
        cuda-cusparse-9-0 \
        libcudnn7=${CUDNN_VERSION} \
        libnccl2=${NCCL_VERSION} \
        libgomp1 \
        python3 \
        python3-dev \
        python3-setuptools \
        python3-wheel \
        python3-pip \
        && \
    apt-get update && apt-get install -y --no-install-recommends \
        nvinfer-runtime-trt-repo-ubuntu1604-4.0.1-ga-cuda9.0 && \
    apt-get update && apt-get install -y --no-install-recommends \
        libnvinfer4=${TF_TENSORRT_VERSION}-1+cuda9.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm /usr/lib/x86_64-linux-gnu/libnvinfer_plugin* && \
    rm /usr/lib/x86_64-linux-gnu/libnvcaffe_parser* && \
    rm /usr/lib/x86_64-linux-gnu/libnvparsers*

# Here we get all python packages.
# There's substantial overlap between scipy and numpy that we eliminate by
# linking them together. Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.
# tensorflowjs==0.8.0
RUN pip3 install \
    tensorflow-gpu==1.12 \
    tensorflowjs==0.8.0 \
    tensorflow-hub \
    pillow \
    scipy \
    mlagents_envs==0.8.2 \
    && rm -rf /root/.cache

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY train /opt/program
WORKDIR /opt/program
