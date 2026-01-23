#!/bin/bash
# Full Vast.ai Setup Script for OmniTransfer Training
#
# Usage:
#   1. Copy this script to the instance: scp -P PORT setup_vastai_full.sh root@ssh.vast.ai:~
#   2. SSH into the instance: ssh -p PORT root@ssh.vast.ai
#   3. Set environment variables (see below)
#   4. Run: bash setup_vastai_full.sh
#
# Required environment variables:
#   export AWS_ACCESS_KEY_ID="your_aws_key"
#   export AWS_SECRET_ACCESS_KEY="your_aws_secret"
#   export HF_TOKEN="your_huggingface_token"  # Optional but recommended
#   export WANDB_API_KEY="your_wandb_key"     # For training logs

set -e

# Configuration
WORKSPACE="/workspace"
DATA_DIR="/data"
MODEL_DIR="$DATA_DIR/models"
TRAINING_DATA_DIR="$DATA_DIR/omnitransfer_training"
S3_BUCKET="s3://imf-infinity-latents/omnitransfer/training_data/"

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║         OmniTransfer Vast.ai Training Environment Setup              ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check for required environment variables
check_env() {
    if [ -z "${!1}" ]; then
        echo "⚠️  WARNING: $1 is not set"
        return 1
    else
        echo "✓ $1 is set"
        return 0
    fi
}

echo "[0/7] Checking environment variables..."
check_env "AWS_ACCESS_KEY_ID" || AWS_MISSING=1
check_env "AWS_SECRET_ACCESS_KEY" || AWS_MISSING=1
check_env "HF_TOKEN" || HF_MISSING=1
check_env "WANDB_API_KEY" || WANDB_MISSING=1

if [ "$AWS_MISSING" = "1" ]; then
    echo ""
    echo "❌ AWS credentials are REQUIRED for S3 data sync."
    echo "   Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
    exit 1
fi

echo ""

# =============================================================================
# Step 1: Install system dependencies
# =============================================================================
echo "[1/7] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq git curl unzip htop nvtop

# Install AWS CLI
if ! command -v aws &> /dev/null; then
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip -q awscliv2.zip
    ./aws/install
    rm -rf awscliv2.zip aws/
fi
echo "✓ System dependencies installed"

# =============================================================================
# Step 2: Install uv for Python package management
# =============================================================================
echo "[2/7] Installing uv package manager..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi
echo "✓ uv installed"

# =============================================================================
# Step 3: Configure AWS credentials
# =============================================================================
echo "[3/7] Configuring AWS credentials..."
mkdir -p ~/.aws
cat > ~/.aws/credentials << EOF
[default]
aws_access_key_id = $AWS_ACCESS_KEY_ID
aws_secret_access_key = $AWS_SECRET_ACCESS_KEY
EOF
cat > ~/.aws/config << EOF
[default]
region = us-west-2
EOF
echo "✓ AWS credentials configured"

# =============================================================================
# Step 4: Clone repository
# =============================================================================
echo "[4/7] Setting up repository..."
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

if [ ! -d "ltx2-omnitransfer" ]; then
    git clone https://github.com/johndpope/ltx2-omnitransfer.git
    echo "✓ Repository cloned"
else
    cd ltx2-omnitransfer
    git pull origin main
    cd ..
    echo "✓ Repository updated"
fi

cd ltx2-omnitransfer/packages/ltx-trainer

# =============================================================================
# Step 5: Setup Python environment
# =============================================================================
echo "[5/7] Setting up Python environment..."
uv sync
echo "✓ Python environment ready"

# =============================================================================
# Step 6: Download training data from S3
# =============================================================================
echo "[6/7] Downloading training data from S3..."
mkdir -p "$TRAINING_DATA_DIR"
aws s3 sync "$S3_BUCKET" "$TRAINING_DATA_DIR/" --no-progress
echo "✓ Training data downloaded"
echo "   Files:"
ls -la "$TRAINING_DATA_DIR/"

# =============================================================================
# Step 7: Download LTX-2 model weights
# =============================================================================
echo "[7/7] Downloading LTX-2 model weights..."
mkdir -p "$MODEL_DIR"

# Install huggingface_hub
pip install -q huggingface_hub

# Login to HuggingFace if token is set
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
fi

# Download LTX-2 checkpoint
if [ ! -f "$MODEL_DIR/ltx-2-19b-dev.safetensors" ]; then
    echo "   Downloading LTX-2 transformer weights (~38GB)..."
    huggingface-cli download Lightricks/LTX-2 \
        ltx-2-19b-dev.safetensors \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False
else
    echo "   ✓ LTX-2 weights already exist"
fi

# Download Gemma text encoder
if [ ! -d "$MODEL_DIR/gemma" ] || [ ! -f "$MODEL_DIR/gemma/model.safetensors" ]; then
    echo "   Downloading Gemma text encoder..."
    mkdir -p "$MODEL_DIR/gemma"
    huggingface-cli download Lightricks/LTX-2 \
        text_encoder/config.json \
        text_encoder/model.safetensors \
        text_encoder/special_tokens_map.json \
        text_encoder/tokenizer.json \
        text_encoder/tokenizer_config.json \
        --local-dir "$MODEL_DIR/gemma_temp" \
        --local-dir-use-symlinks False
    mv "$MODEL_DIR/gemma_temp/text_encoder"/* "$MODEL_DIR/gemma/"
    rm -rf "$MODEL_DIR/gemma_temp"
else
    echo "   ✓ Gemma text encoder already exists"
fi

echo "✓ Model weights downloaded"

# =============================================================================
# Configure W&B
# =============================================================================
if [ -n "$WANDB_API_KEY" ]; then
    echo ""
    echo "Configuring Weights & Biases..."
    pip install -q wandb
    wandb login "$WANDB_API_KEY"
    echo "✓ W&B configured"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                        Setup Complete!                                ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Directories:"
echo "   Repository:     $WORKSPACE/ltx2-omnitransfer"
echo "   Training Data:  $TRAINING_DATA_DIR"
echo "   Models:         $MODEL_DIR"
echo ""
echo "📊 Disk usage:"
du -sh "$TRAINING_DATA_DIR" "$MODEL_DIR" 2>/dev/null || true
echo ""
echo "🚀 To start training:"
echo "   cd $WORKSPACE/ltx2-omnitransfer/packages/ltx-trainer"
echo "   uv run python scripts/train.py configs/ltx2_omnitransfer_stage1_a100_80gb.yaml"
echo ""
echo "🔍 Monitor GPU:"
echo "   nvtop"
echo ""
echo "📈 View logs:"
echo "   Open https://wandb.ai to see training progress"
echo ""
