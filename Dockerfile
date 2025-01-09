FROM almalinux:9

RUN dnf install -y epel-release && dnf clean all && \
  dnf config-manager --set-enabled crb

RUN dnf groupinstall -y "Development Tools" && \
  dnf install -y \
  git \
  cmake \
  gcc-c++ \
  libX11-devel \
  libXpm-devel \
  libXft-devel \
  libXext-devel \
  openssl-devel \
  which \
  && dnf clean all

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
RUN cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -D xrootd=OFF -D proof=ON && \
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

WORKDIR /opt
RUN git clone https://github.com/denehoffman/thesis_analysis.git
WORKDIR /opt/thesis_analysis
RUN uv pip install -e .

WORKDIR /work

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
