#!/bin/bash
# =============================================================================
# Upload Videos to S3 for OmniTransfer Training
# =============================================================================
# Usage: ./upload_videos.sh /path/to/local/videos
# =============================================================================

set -e

# Load environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../.env"

if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
else
    echo "ERROR: .env file not found at $ENV_FILE"
    echo "Copy .env.example to .env and configure your credentials."
    exit 1
fi

# Check required variables
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ] || [ -z "$S3_BUCKET" ]; then
    echo "ERROR: Missing required AWS credentials in .env"
    exit 1
fi

# Parse arguments
LOCAL_DIR="${1:-}"
if [ -z "$LOCAL_DIR" ]; then
    echo "Usage: $0 /path/to/videos"
    exit 1
fi

if [ ! -d "$LOCAL_DIR" ]; then
    echo "ERROR: Directory not found: $LOCAL_DIR"
    exit 1
fi

# Configure AWS
export AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY
AWS_REGION="${S3_REGION:-us-east-1}"
S3_PREFIX="${S3_PREFIX:-videos}"
STORAGE_CLASS="${S3_STORAGE_CLASS:-STANDARD}"

echo "=== OmniTransfer S3 Video Upload ==="
echo "Local: $LOCAL_DIR"
echo "S3: s3://$S3_BUCKET/$S3_PREFIX/"
echo "Region: $AWS_REGION"
echo "Storage Class: $STORAGE_CLASS"
echo ""

# Count files
VIDEO_COUNT=$(find "$LOCAL_DIR" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.webm" \) | wc -l)
TOTAL_SIZE=$(du -sh "$LOCAL_DIR" 2>/dev/null | cut -f1)

echo "Videos found: $VIDEO_COUNT"
echo "Total size: $TOTAL_SIZE"
echo ""

# Confirm upload
read -p "Proceed with upload? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Upload with progress
echo ""
echo "Uploading..."
aws s3 sync "$LOCAL_DIR" "s3://$S3_BUCKET/$S3_PREFIX/" \
    --region "$AWS_REGION" \
    --storage-class "$STORAGE_CLASS" \
    --exclude "*.pt" \
    --exclude "*.pth" \
    --exclude "*.safetensors" \
    --include "*.mp4" \
    --include "*.avi" \
    --include "*.mov" \
    --include "*.webm" \
    --include "*.png" \
    --include "*.jpg" \
    --include "*.jpeg" \
    --include "*.json" \
    --include "*.txt"

echo ""
echo "=== Upload Complete ==="
echo "Videos available at: s3://$S3_BUCKET/$S3_PREFIX/"
echo ""
echo "List uploaded files:"
echo "  aws s3 ls s3://$S3_BUCKET/$S3_PREFIX/ --recursive --human-readable"
