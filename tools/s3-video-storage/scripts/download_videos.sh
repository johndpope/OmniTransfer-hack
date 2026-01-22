#!/bin/bash
# =============================================================================
# Download Videos from S3 for OmniTransfer Training
# =============================================================================
# Usage: ./download_videos.sh /path/to/local/output
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
    echo "Usage: $0 /path/to/output/directory"
    exit 1
fi

# Configure AWS
export AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY
AWS_REGION="${S3_REGION:-us-east-1}"
S3_PREFIX="${S3_PREFIX:-videos}"

echo "=== OmniTransfer S3 Video Download ==="
echo "S3: s3://$S3_BUCKET/$S3_PREFIX/"
echo "Local: $LOCAL_DIR"
echo "Region: $AWS_REGION"
echo ""

# Check what's available on S3
echo "Checking S3 contents..."
REMOTE_COUNT=$(aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/" --recursive --region "$AWS_REGION" | wc -l)
REMOTE_SIZE=$(aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/" --recursive --summarize --region "$AWS_REGION" | grep "Total Size" | awk '{print $3, $4}')

echo "Remote files: $REMOTE_COUNT"
echo "Remote size: $REMOTE_SIZE"
echo ""

# Create output directory
mkdir -p "$LOCAL_DIR"

# Confirm download
read -p "Proceed with download? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Download with progress
echo ""
echo "Downloading..."
aws s3 sync "s3://$S3_BUCKET/$S3_PREFIX/" "$LOCAL_DIR" \
    --region "$AWS_REGION"

echo ""
echo "=== Download Complete ==="
echo "Files saved to: $LOCAL_DIR"
echo ""

# Count downloaded files
LOCAL_COUNT=$(find "$LOCAL_DIR" -type f | wc -l)
LOCAL_SIZE=$(du -sh "$LOCAL_DIR" 2>/dev/null | cut -f1)

echo "Downloaded: $LOCAL_COUNT files ($LOCAL_SIZE)"
