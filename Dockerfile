# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 as builder

# install python 3.12
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# set python alternatives
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# install rust toolchain for maturin builds
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# install uv for fast python dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# copy project files
COPY README.md pyproject.toml uv.lock ./
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY native ./native

# build the application with all dependencies
# set cmake args for llama-cpp-python cuda support
ENV CMAKE_ARGS="-DGGML_CUDA=on" \
    FORCE_CMAKE=1
RUN uv sync --frozen --no-dev

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# install python 3.12
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# set python alternatives
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

WORKDIR /app

# copy built application from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

# set environment variables
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=info

ENTRYPOINT ["bookmarks"]
CMD ["--help"]
