FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04

WORKDIR /workspace/distributed_rl
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update \
    && apt-get -y --no-install-recommends install curl redis cmake zlib1g-dev python3 python3-pip python3.8-venv unrar wget \
    && rm --recursive --force /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -

COPY . .
ENV PATH $PATH:/root/.local/bin

RUN poetry config virtualenvs.create false \
    && poetry run pip install -U pip \
    && apt purge -y python3-pip \
    && poetry install
RUN cp config/redis.conf /etc/redis/.

RUN mkdir roms && \
    cd roms && \
    wget http://www.atarimania.com/roms/Roms.rar && \
    unrar e Roms.rar -y && \
    rm Roms.rar && \
    python3 -m atari_py.import_roms .

ENTRYPOINT /bin/bash
