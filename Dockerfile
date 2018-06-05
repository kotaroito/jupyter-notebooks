FROM ubuntu:16.04

# Install deb packages
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Install python packages
RUN pip3 install --no-cache-dir pandas scipy jupyter
RUN pip3 install --no-cache-dir chainer==4.1.0
