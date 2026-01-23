#!/bin/bash
# Download LTX-2 model weights and Gemma text encoder
# Run this on the Vast.ai instance after setup

set -e

MODEL_DIR="${MODEL_DIR:-/data/models}"
mkdir -p "$MODEL_DIR"

echo "=========================================="
echo "Downloading LTX-2 Model Weights"
echo "=========================================="
echo "Target directory: $MODEL_DIR"
echo ""

# Check if huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface_hub..."
    pip install huggingface_hub
fi

# Login to HuggingFace if token is set
if [ -n "$HF_TOKEN" ]; then
    echo "Logging into HuggingFace..."
    huggingface-cli login --token "$HF_TOKEN"
fi

# Download LTX-2 main checkpoint
echo "[1/3] Downloading LTX-2 transformer weights..."
if [ ! -f "$MODEL_DIR/ltx-2-19b-dev.safetensors" ]; then
    huggingface-cli download Lightricks/LTX-2 \
        ltx-2-19b-dev.safetensors \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False
    echo "✓ LTX-2 weights downloaded"
else
    echo "✓ LTX-2 weights already exist"
fi

# Download Gemma text encoder
echo "[2/3] Downloading Gemma text encoder..."
if [ ! -d "$MODEL_DIR/gemma" ]; then
    mkdir -p "$MODEL_DIR/gemma"
    huggingface-cli download Lightricks/LTX-2 \
        text_encoder/config.json \
        text_encoder/model.safetensors \
        text_encoder/special_tokens_map.json \
        text_encoder/tokenizer.json \
        text_encoder/tokenizer_config.json \
        --local-dir "$MODEL_DIR/gemma" \
        --local-dir-use-symlinks False
    # Move files from text_encoder subdirectory
    if [ -d "$MODEL_DIR/gemma/text_encoder" ]; then
        mv "$MODEL_DIR/gemma/text_encoder"/* "$MODEL_DIR/gemma/"
        rmdir "$MODEL_DIR/gemma/text_encoder"
    fi
    echo "✓ Gemma text encoder downloaded"
else
    echo "✓ Gemma text encoder already exists"
fi

# Download VAE (for visualization/LPIPS)
echo "[3/3] Downloading VAE weights..."
if [ ! -f "$MODEL_DIR/ltx-2-19b-dev.safetensors" ]; then
    echo "VAE weights are included in the main checkpoint"
fi
echo "✓ VAE weights ready (included in main checkpoint)"

echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""
echo "Model locations:"
echo "  Transformer: $MODEL_DIR/ltx-2-19b-dev.safetensors"
echo "  Text Encoder: $MODEL_DIR/gemma/"
echo ""
echo "Total disk usage:"
du -sh "$MODEL_DIR"/*
echo ""

# Verify files
echo "Verifying downloads..."
if [ -f "$MODEL_DIR/ltx-2-19b-dev.safetensors" ] && [ -d "$MODEL_DIR/gemma" ]; then
    echo "✓ All models verified!"
else
    echo "✗ Some models are missing. Check the output above."
    exit 1
fi
