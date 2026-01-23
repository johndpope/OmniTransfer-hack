#!/bin/bash
# =============================================================================
# Upload OmniTransfer Training Data to S3
# =============================================================================
# This script uploads the preprocessed latents, conditions, and metadata to S3
# for use with cloud training on Vast.ai
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load credentials
CREDS_FILE="${CREDS_FILE:-$REPO_ROOT/packages/ltx-trainer/credentials.env}"
if [ -f "$CREDS_FILE" ]; then
    echo "Loading credentials from: $CREDS_FILE"
    export $(grep -v '^#' "$CREDS_FILE" | xargs)
else
    echo "ERROR: Credentials file not found: $CREDS_FILE"
    echo "Copy credentials.env.example to credentials.env and fill in your keys"
    exit 1
fi

# Configuration
S3_BUCKET="${S3_BUCKET:-imf-infinity-latents}"
S3_REGION="${S3_REGION:-us-east-1}"
S3_PREFIX="omnitransfer/training_data"

# Local data paths
DATA_DIR="${DATA_DIR:-/media/2TB/omnitransfer_unified_5task}"
VIDEOS_DIR="${VIDEOS_DIR:-$REPO_ROOT/packages/ltx-trainer/omnitransfer_official_videos}"

echo "=== OmniTransfer S3 Upload ==="
echo "Bucket: s3://$S3_BUCKET"
echo "Region: $S3_REGION"
echo "Data:   $DATA_DIR"
echo ""

# Configure AWS CLI
aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID"
aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
aws configure set region "$S3_REGION"

# Test S3 access
echo "Testing S3 access..."
aws s3 ls "s3://$S3_BUCKET" --region "$S3_REGION" > /dev/null 2>&1 || {
    echo "Creating bucket..."
    aws s3 mb "s3://$S3_BUCKET" --region "$S3_REGION" || true
}

# Upload preprocessed data (latents, conditions, references, metadata)
echo ""
echo "=== Uploading Preprocessed Training Data ==="
echo "Source: $DATA_DIR"
echo "Target: s3://$S3_BUCKET/$S3_PREFIX/"

if [ -d "$DATA_DIR" ]; then
    # Calculate total size
    TOTAL_SIZE=$(du -sh "$DATA_DIR" | cut -f1)
    echo "Total size: $TOTAL_SIZE"

    # Sync with progress
    aws s3 sync "$DATA_DIR" "s3://$S3_BUCKET/$S3_PREFIX/" \
        --region "$S3_REGION" \
        --exclude "*.log" \
        --exclude "*.pyc" \
        --exclude "__pycache__/*"

    echo "Preprocessed data uploaded successfully!"
else
    echo "ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi

# Optionally upload raw videos (larger, ~42MB)
echo ""
read -p "Upload raw videos too? (~42MB) [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "=== Uploading Raw Videos ==="
    if [ -d "$VIDEOS_DIR" ]; then
        aws s3 sync "$VIDEOS_DIR" "s3://$S3_BUCKET/omnitransfer/raw_videos/" \
            --region "$S3_REGION"
        echo "Raw videos uploaded!"
    else
        echo "Warning: Videos directory not found: $VIDEOS_DIR"
    fi
fi

# Summary
echo ""
echo "=== Upload Complete ==="
echo ""
echo "S3 Structure:"
aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/" --region "$S3_REGION" --human-readable

echo ""
echo "To download on Vast.ai instance:"
echo "  aws s3 sync s3://$S3_BUCKET/$S3_PREFIX/ /data/omnitransfer_training/"
echo ""
echo "Total S3 usage:"
aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/" --region "$S3_REGION" --recursive --summarize | tail -2
