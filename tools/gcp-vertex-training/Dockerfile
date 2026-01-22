# =============================================================================
# OmniTransfer Training Container for Google Vertex AI
# =============================================================================
# Build: docker build -t gcr.io/$GCP_PROJECT/omnitransfer-train:latest .
# Push: docker push gcr.io/$GCP_PROJECT/omnitransfer-train:latest
# =============================================================================

FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV WORKSPACE=/workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    curl \
    wget \
    vim \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK
RUN curl -sSL https://sdk.cloud.google.com | bash
ENV PATH="${PATH}:/root/google-cloud-sdk/bin"

# Install uv for fast Python package management
RUN pip install uv

# Clone repository
WORKDIR ${WORKSPACE}
RUN git clone https://github.com/johndpope/OmniTransfer-hack.git

# Install Python dependencies
WORKDIR ${WORKSPACE}/OmniTransfer-hack
RUN uv sync

# Install additional training dependencies
RUN pip install wandb lpips open_clip_torch

# Verify flash-attn (pre-installed in NGC image)
RUN python -c "from flash_attn import flash_attn_func; print('flash-attn OK')"

# Copy training scripts
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Set working directory
WORKDIR ${WORKSPACE}/OmniTransfer-hack/packages/ltx-trainer

# Default command
ENTRYPOINT ["/start.sh"]
