# Everything the loop needs on one machine with one NVIDIA GPU: CUDA 13 for nvcc and CUPTI,
# kern built from the pinned submodule, humanize, and two agent CLIs (others: `hmz exec --help`).
# Run as your own user (README): tools live outside any home, HOME is a world-writable directory.
FROM ubuntu:24.04 AS kern
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates build-essential pkg-config libssl-dev && \
    curl -LsSf https://sh.rustup.rs | sh -s -- -y --profile minimal
COPY third_party/kern /kern
RUN cd /kern && /root/.cargo/bin/cargo build --release -p kern-run

FROM nvidia/cuda:13.3.1-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive HOME=/home/agent \
    UV_TOOL_DIR=/opt/uv/tools UV_TOOL_BIN_DIR=/usr/local/bin UV_PYTHON_INSTALL_DIR=/opt/uv/python
RUN apt-get update && apt-get install -y --no-install-recommends git tmux curl ca-certificates python3 nodejs npm && \
    rm -rf /var/lib/apt/lists/* && mkdir -p /home/agent && chmod 1777 /home/agent
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh && \
    uv tool install git+https://github.com/humanfia/humanize.git@v0.2.2 && \
    uv tool install 'huggingface_hub[cli]' && \
    npm install -g @anthropic-ai/claude-code @openai/codex && npm cache clean --force && \
    chmod -R a+rX /opt
COPY --from=kern /kern/target/release/kern /usr/local/bin/kern
WORKDIR /work
CMD ["bash"]
