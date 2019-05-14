# Build an image that can do training in SageMaker
FROM nvidia/cuda:10.0-base-ubuntu18.04

MAINTAINER Loren Kuich <loren@lkuich.com>

ENV CUDNN_VERSION=7.4.1.5-1+cuda10.0
ENV CUDNN_DEV_VERSION=7.4.1.5-1+cuda10.0

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libcudnn7=${CUDNN_VERSION} \
        libcudnn7-dev=${CUDNN_DEV_VERSION} \
        python3 \
        python3-dev \
        python3-setuptools \
        python3-wheel \
        python3-pip \
        && \

    apt-get update && \
        apt-get install nvinfer-runtime-trt-repo-ubuntu1804-5.0.2-ga-cuda10.0 \
        && apt-get update \
        && apt-get install -y --no-install-recommends libnvinfer-dev=5.0.2-1+cuda10.0 \
	&& \

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
RUN pip3 install tensorflow-gpu==2.0.0-alpha0 tensorflow-hub pillow scipy tensorflowjs && rm -rf /root/.cache

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
