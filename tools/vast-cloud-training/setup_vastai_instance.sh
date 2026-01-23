#!/bin/bash
# Setup script for Vast.ai OmniTransfer training instance
# Run this AFTER connecting via SSH

set -e

echo "=========================================="
echo "Setting up OmniTransfer Training Environment"
echo "=========================================="

# Install system dependencies
echo "[1/7] Installing system packages..."
apt-get update && apt-get install -y git curl unzip awscli

# Install uv for Python package management
echo "[2/7] Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
source ~/.bashrc 2>/dev/null || true

# Clone repository
echo "[3/7] Cloning ltx2-omnitransfer repository..."
cd /workspace
if [ ! -d "ltx2-omnitransfer" ]; then
    git clone https://github.com/johndpope/ltx2-omnitransfer.git
fi
cd ltx2-omnitransfer

# Setup Python environment
echo "[4/7] Setting up Python environment..."
cd packages/ltx-trainer
uv sync

# Configure AWS credentials (you'll need to set these)
echo "[5/7] Configuring AWS credentials..."
mkdir -p ~/.aws
if [ -z "$AWS_ACCESS_KEY_ID" ]; then
    echo "WARNING: AWS_ACCESS_KEY_ID not set. Set it with:"
    echo "  export AWS_ACCESS_KEY_ID=your_key"
    echo "  export AWS_SECRET_ACCESS_KEY=your_secret"
else
    cat > ~/.aws/credentials << EOF
[default]
aws_access_key_id = $AWS_ACCESS_KEY_ID
aws_secret_access_key = $AWS_SECRET_ACCESS_KEY
EOF
    echo "AWS credentials configured."
fi

# Download training data from S3
echo "[6/7] Downloading training data from S3..."
mkdir -p /data/omnitransfer_training
aws s3 sync s3://imf-infinity-latents/omnitransfer/training_data/ /data/omnitransfer_training/ --no-progress

# Download LTX-2 model
echo "[7/7] Downloading LTX-2 model..."
mkdir -p /data/models
if [ ! -f "/data/models/ltx-2-19b-dev.safetensors" ]; then
    echo "You need to download the LTX-2 model manually."
    echo "Options:"
    echo "  1. From HuggingFace: huggingface-cli download Lightricks/LTX-2 ltx-2-19b-dev.safetensors --local-dir /data/models"
    echo "  2. From S3 if you have it stored there"
fi

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download LTX-2 model if not already done"
echo "2. Run training:"
echo "   cd /workspace/ltx2-omnitransfer/packages/ltx-trainer"
echo "   uv run python scripts/train.py configs/ltx2_omnitransfer_stage1_test.yaml"
echo ""
echo "Data location: /data/omnitransfer_training/"
echo "Model location: /data/models/"
