# Build an image that can do training in SageMaker
FROM nvidia/cuda:10.0-base-ubuntu18.04

MAINTAINER Loren Kuich <loren@lkuich.com>

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        cuda-command-line-tools-10-0 \
        libcudnn7=7.6.2.24-1+cuda10.0  \
        libcudnn7-dev=7.6.2.24-1+cuda10.0 \
        python3 \
        python3-dev \
        python3-setuptools \
        python3-wheel \
        python3-pip \
        && \
    apt-get update && apt-get install -y --no-install-recommends \
        libnvinfer5=5.1.5-1+cuda10.0 \
        libnvinfer-dev=5.1.5-1+cuda10.0 && \
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
    absl-py==0.8.0  \
    astor==0.8.0 \
    atomicwrites==1.3.0 \
    attrs==19.1.0 \
    cloudpickle==0.8.1 \
    gast==0.2.2 \
    google-pasta==0.1.7 \
    grpcio==1.23.0 \
    h5py==2.8.0 \
    importlib-metadata==0.20 \
    Keras==2.2.4 \
    Keras-Applications==1.0.8 \
    Keras-Preprocessing==1.1.0 \
    Markdown==3.1.1 \
    mlagents-envs==0.9.3 \
    more-itertools==7.2.0 \
    numpy==1.16.4 \
    Pillow==5.4.1 \
    pluggy==0.12.0 \
    prompt-toolkit==1.0.14 \
    protobuf==3.9.1 \
    py==1.8.0 \
    Pygments==2.4.2 \
    PyInquirer==1.0.3 \
    pytest==3.10.1 \
    PyYAML==5.1.2 \
    regex==2019.8.19 \
    scipy==1.3.0 \
    six==1.11.0 \
    tensorboard==1.14.0 \
    tensorflow-estimator==1.14.0 \
    tensorflow==1.14.0 \
    tensorflow-gpu==1.14.0 \
    tensorflow-hub==0.5.0 \
    tensorflowjs==1.2.9 \
    termcolor==1.1.0 \
    wcwidth==0.1.7 \
    Werkzeug==0.15.6 \
    wrapt==1.11.2 \
    zipp==0.6.0 \
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