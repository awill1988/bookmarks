# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

# install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    curl \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# install rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY common ./common
COPY bookmarks ./bookmarks

# build release binary
RUN cargo build --release

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy binary from builder
COPY --from=builder /app/target/release/bookmarks /usr/local/bin/bookmarks

# create cache and data directories with proper ownership
RUN mkdir -p /app/.cache /app/data && \
    chown -R 65534:65534 /app

# set environment variables
ENV LOG_LEVEL=info

# switch to nobody user (65534:65534)
USER 65534:65534

ENTRYPOINT ["bookmarks"]
CMD ["--help"]
