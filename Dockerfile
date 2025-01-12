FROM ubuntu:24.10

RUN apt-get update && \
  apt-get install -y software-properties-common && \
  add-apt-repository multiverse && \
  apt-get update

RUN apt-get install -y \
  build-essential \
  binutils \
  cmake \
  dpkg-dev \
  g++ \
  gcc \
  libssl-dev \
  git \
  libx11-dev \
  libxext-dev \
  libxft-dev \
  libxpm-dev \
  libtbb-dev \
  libvdt-dev \
  libgif-dev \
  gfortran \
  libpcre3-dev \
  libglu1-mesa-dev \
  libglew-dev \
  libftgl-dev \
  libfftw3-dev \
  libcfitsio-dev \
  libgraphviz-dev \
  libavahi-compat-libdnssd-dev \
  libldap2-dev \
  libxml2-dev \
  libkrb5-dev \
  libgsl-dev \
  qtwebengine5-dev \
  nlohmann-json3-dev \
  libmysqlclient-dev \
  libgl2ps-dev \
  liblzma-dev \
  libxxhash-dev \
  liblz4-dev \
  libzstd-dev \
  libcurl4-openssl-dev \
  libssl-dev \
  curl \
  which \
  texlive-full \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh


ENV PATH="/root/.local/bin/:$PATH"
RUN uv venv /opt/venv --python=3.13
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /opt
RUN git clone --branch latest-stable https://github.com/root-project/root.git

WORKDIR /opt/root
RUN mkdir root_build
WORKDIR /opt/root/root_build
RUN cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -D xrootd=OFF -D proof=ON -D davix=OFF -D clad=OFF && \
  cmake --build . -- -j"$(nproc)" && \
  cmake --install .

ENV ROOTSYS=/usr/local
ENV PATH=$ROOTSYS/bin:$PATH
ENV LD_LIBRARY_PATH=$ROOTSYS/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH=$ROOTSYS/lib:$PYTHONPATH

RUN mkdir -p /var/log/luigi && chmod 777 /var/log/luigi

COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8082

WORKDIR /work

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
