# Build an image that can do training in SageMaker
FROM tensorflow/tensorflow:1.14.0-gpu-py3

MAINTAINER Loren Kuich <loren@lkuich.com>

RUN pip3 install \
    mlagents-envs==0.9.3 \
    tensorflow-hub==0.5.0 \
    keras
RUN pip3 install tensorflowjs==1.2.6 --no-dependencies

RUN rm -rf /root/.cache

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