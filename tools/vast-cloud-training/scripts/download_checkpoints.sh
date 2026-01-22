#!/bin/bash
# =============================================================================
# Download Training Checkpoints from S3
# =============================================================================
# Run this LOCALLY to download checkpoints from cloud training
# Usage: ./download_checkpoints.sh [output_dir]
# =============================================================================

set -e

# Load environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../.env"

if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
else
    echo "ERROR: .env file not found at $ENV_FILE"
    echo "Copy terraform.tfvars.example to .env and configure your credentials."
    exit 1
fi

# Check required variables
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ] || [ -z "$S3_BUCKET" ]; then
    echo "ERROR: Missing required AWS credentials"
    exit 1
fi

# Parse arguments
LOCAL_DIR="${1:-./outputs/cloud_checkpoints}"
S3_PREFIX="${S3_PREFIX:-checkpoints/omnitransfer}"

# Configure AWS
export AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY
AWS_REGION="${S3_REGION:-us-east-1}"

echo "=== Download OmniTransfer Checkpoints ==="
echo "S3: s3://$S3_BUCKET/$S3_PREFIX/"
echo "Local: $LOCAL_DIR"
echo ""

# Check what's available
echo "Available checkpoints on S3:"
aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/" --region "$AWS_REGION" || {
    echo "No checkpoints found at s3://$S3_BUCKET/$S3_PREFIX/"
    exit 1
}

# Create output directory
mkdir -p "$LOCAL_DIR"

# Confirm download
read -p "Download all checkpoints? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Download
echo ""
echo "Downloading..."
aws s3 sync "s3://$S3_BUCKET/$S3_PREFIX/" "$LOCAL_DIR" --region "$AWS_REGION"

echo ""
echo "=== Download Complete ==="
echo "Checkpoints saved to: $LOCAL_DIR"
echo ""

# List downloaded checkpoints
echo "Downloaded checkpoints:"
ls -la "$LOCAL_DIR"
