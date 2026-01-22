#!/bin/bash
# =============================================================================
# Upload Training Data to S3 for Cloud Training
# =============================================================================
# Run this LOCALLY to upload preprocessed training data to S3
# Usage: ./upload_training_data.sh /path/to/processed/data
# =============================================================================

set -e

# Load environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../.env"

if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
else
    echo "ERROR: .env file not found at $ENV_FILE"
    echo "Create .env with your AWS credentials."
    exit 1
fi

# Check required variables
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ] || [ -z "$S3_BUCKET" ]; then
    echo "ERROR: Missing required AWS credentials"
    exit 1
fi

# Parse arguments
LOCAL_DIR="${1:-}"
if [ -z "$LOCAL_DIR" ]; then
    echo "Usage: $0 /path/to/processed/training/data"
    echo ""
    echo "Example directories:"
    echo "  /media/2TB/omnitransfer_unified_5task"
    echo "  /media/2TB/movie_weaver_training"
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
STORAGE_CLASS="${S3_STORAGE_CLASS:-STANDARD}"

# Determine S3 prefix from directory name
DATASET_NAME=$(basename "$LOCAL_DIR")
S3_PREFIX="${S3_PREFIX:-processed/$DATASET_NAME}"

echo "=== Upload OmniTransfer Training Data ==="
echo "Local: $LOCAL_DIR"
echo "S3: s3://$S3_BUCKET/$S3_PREFIX/"
echo "Region: $AWS_REGION"
echo ""

# Show local structure
echo "Local directories:"
for dir in "$LOCAL_DIR"/*/; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -type f \( -name "*.pt" -o -name "*.json" \) 2>/dev/null | wc -l)
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "  $(basename "$dir"): $count files ($size)"
    fi
done
echo ""

# Count totals
LOCAL_COUNT=$(find "$LOCAL_DIR" -type f \( -name "*.pt" -o -name "*.json" -o -name "*.txt" \) | wc -l)
LOCAL_SIZE=$(du -sh "$LOCAL_DIR" 2>/dev/null | cut -f1)
echo "Total: $LOCAL_COUNT files ($LOCAL_SIZE)"
echo ""

# Confirm
read -p "Upload to S3? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Upload
echo ""
echo "=== Uploading to S3 ==="
aws s3 sync "$LOCAL_DIR" "s3://$S3_BUCKET/$S3_PREFIX/" \
    --region "$AWS_REGION" \
    --storage-class "$STORAGE_CLASS"

echo ""
echo "=== Upload Complete ==="
echo ""
echo "Training data available at: s3://$S3_BUCKET/$S3_PREFIX/"
echo ""
echo "To use in cloud training, set in terraform.tfvars:"
echo "  s3_data_prefix = \"$S3_PREFIX\""
