# Dockerfile for building a container that sets up the ai-toolkit environment
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04


ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone the repository (alternatively, build with the repo as build context and COPY . /app)
RUN git clone https://github.com/ostris/ai-toolkit.git /app

# Create a venv and install Python deps. We put venv in /opt/venv and prepend it to PATH.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip

# Install torch + torchvision with CUDA 13.0 support, then the project's requirements
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Create a config dir and set an env var pointing to the style config expected by the project.
RUN mkdir -p /app/config

# Expose working directory and default to an interactive shell. Users can override CMD to run training.
WORKDIR /app
CMD ["bash"]

# Build notes:
# - To build: docker build -t burgerizer:latest .
# - To run and mount a local config file: \
#     docker run --gpus all -v $(pwd)/config/burger_style.yaml:/app/config/burger_style.yaml -it burgerizer:latest
# - If you prefer not to clone at build time, replace the RUN git clone line with: COPY . /app
