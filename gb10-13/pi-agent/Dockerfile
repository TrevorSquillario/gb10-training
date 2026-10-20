FROM node:24-bookworm-slim

ENV PI_HOME=/home/pi
ENV HOME=/home/pi

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment and expose `python` command
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
    && ln -sf /opt/venv/bin/python /usr/local/bin/python \
    && ln -sf /opt/venv/bin/pip /usr/local/bin/pip

ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies from hermes requirements
COPY requirements.txt /tmp/requirements.txt
RUN /opt/venv/bin/pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Create pi user and home directory structure
RUN addgroup --system pi && \
    adduser --system --ingroup pi pi && \
    mkdir -p ${PI_HOME}/.pi/agent/skills && \
    mkdir -p ${PI_HOME}/.pi/agent/agents && \
    mkdir -p ${PI_HOME}/.config/mcp && \
    chown -R pi:pi ${PI_HOME}

# Install Pi agent globally
RUN npm install -g --ignore-scripts @earendil-works/pi-coding-agent

# Install Pi packages one at a time (pi install doesn't accept multiple)
#RUN for pkg in $PI_PACKAGES; do /usr/local/bin/pi install "$pkg"; done

# Copy configuration files
COPY agent/models.json ${PI_HOME}/.pi/agent/models.json
COPY agent/settings.json ${PI_HOME}/.pi/agent/settings.json
COPY config/mcp/mcp.json ${PI_HOME}/.config/mcp/mcp.json

# Ensure proper ownership
RUN chown -R pi:pi ${PI_HOME}

# Copy and configure entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR ${PI_HOME}

ENTRYPOINT ["/entrypoint.sh"]
# Keep container alive so users can exec into it and run `pi` interactively
CMD ["sleep", "infinity"]
