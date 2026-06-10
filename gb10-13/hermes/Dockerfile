FROM nousresearch/hermes-agent:v2026.6.5

# Switch to root to install system packages
USER root

# Install common networking and CLI tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    iputils-ping \
    curl \
    wget \
    dnsutils \
    net-tools \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment, expose `python` command, and install Python deps
# Copy requirements first so Docker layer caches unless requirements change
COPY requirements.txt /tmp/requirements.txt

# Create venv at /opt/venv, upgrade pip, install requirements into it, and
# symlink `python` and `pip` into `/usr/local/bin` so they are on PATH.
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip install -r /tmp/requirements.txt \
    && ln -sf /opt/venv/bin/python /usr/local/bin/python \
    && ln -sf /opt/venv/bin/pip /usr/local/bin/pip \
    && rm /tmp/requirements.txt
# Ensure the virtualenv's `bin` directory is first in PATH so `python` uses it
ENV PATH="/opt/venv/bin:$PATH"

# Add the hermes command to the PATH for the hermes user
ENV PATH="/opt/hermes/.venv/bin:$PATH"

RUN uv pip install hindsight-client discord.py

# Switch back to the default unprivileged user for security
USER hermes

# Cherry-pick a patch from the main branch that adds support for bank_id in hindsight tools
RUN cd /opt/hermes \
    && curl -sSL https://github.com/NousResearch/hermes-agent/pull/28017.patch | git apply --reject || true
