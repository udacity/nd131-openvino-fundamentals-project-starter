FROM ubuntu:18.04
# based on openvino Dockerfile from https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_docker_linux.html
USER root
WORKDIR /

ENV WORKSPACE=/workspace
ARG DEBIAN_FRONTEND=noninteractive

RUN useradd -ms /bin/bash openvino && \
    chown openvino -R /home/openvino
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                  autoconf \
                  automake \
                  build-essential \
                  cmake \
                  cpio \
                  curl \
                  gnupg2 \
                  libdrm2 \
                  libglib2.0-0 \
                  lsb-release \
                  libgtk-3-0 \
                  libtool \
                  python3-pip \
                  udev \
                  unzip \
    && rm -rf /var/lib/apt/lists/*

ARG DOWNLOAD_LINK=http://registrationcenter-download.intel.com/akdlm/irc_nas/13231/l_openvino_toolkit_p_2019.0.000.tgz
WORKDIR /tmp
ENV INSTALL_DIR=/opt/intel/openvino
RUN curl -LOJ "${DOWNLOAD_LINK}" && \
    tar -xzf ./*.tgz && \
    cd l_openvino_toolkit* && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    rm -rf /tmp/* && \
    $INSTALL_DIR/install_dependencies/install_openvino_dependencies.sh

# Additional requirements as specified by the project requirements https://github.com/udacity/nd131-openvino-fundamentals-project-starter/blob/master/linux-setup.md

RUN curl -sSL https://deb.nodesource.com/gpgkey/nodesource.gpg.key | apt-key add - \
  && export VERSION=node_6.x \
  && export DISTRO="$(lsb_release -s -c)" \
  && echo "deb https://deb.nodesource.com/$VERSION $DISTRO main" | tee /etc/apt/sources.list.d/nodesource.list \
  && echo "deb-src https://deb.nodesource.com/$VERSION $DISTRO main" | tee -a /etc/apt/sources.list.d/nodesource.list \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
    nodejs \
    npm \
    python3-dev \
    python3-pip \
    python3-setuptools \
    libzmq3-dev \
    libkrb5-dev \
    ffmpeg \
    cmake \
    vim \
  && pip3 install \
    numpy \
    paho-mqtt \
    opencv-python-headless \
    tensorflow \
    ipython \
  && pip3 install -r /opt/intel/openvino/deployment_tools/model_optimizer/requirements.txt \
  && echo "source /opt/intel/openvino/bin/setupvars.sh" >> /root/.bashrc

# ENVIRONMENT SETUP
COPY nd131-openvino-fundamentals-project-starter $WORKSPACE/
WORKDIR $WORKSPACE

RUN cd $WORKSPACE/webservice/server \
  && npm install \
  && cd $WORKSPACE/webservice/ui \
  && npm install \
  && npm install mosca

