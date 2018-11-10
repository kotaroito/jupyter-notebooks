FROM ubuntu:16.04

# Install deb packages
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    git curl \
    cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

WORKDIR /tmp

# Install python packages
ADD requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Bash Kernel
RUN python3 -m bash_kernel.install