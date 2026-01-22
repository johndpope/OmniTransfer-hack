#!/bin/bash
# =============================================================================
# Sync OmniTransfer Training Data with S3
# =============================================================================
# Bidirectional sync of processed training data (latents, conditions, etc.)
# Usage: ./sync_training_data.sh /path/to/training/data
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
    echo "Usage: $0 /path/to/training/data"
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
UPLOAD_ONLY="${UPLOAD_ONLY:-false}"
DELETE_REMOTE="${DELETE_REMOTE:-false}"
STORAGE_CLASS="${S3_STORAGE_CLASS:-STANDARD}"

# Determine S3 prefix from directory name
DATASET_NAME=$(basename "$LOCAL_DIR")
S3_PREFIX="${S3_PREFIX:-processed/$DATASET_NAME}"

echo "=== OmniTransfer Training Data Sync ==="
echo "Local: $LOCAL_DIR"
echo "S3: s3://$S3_BUCKET/$S3_PREFIX/"
echo "Region: $AWS_REGION"
echo "Upload only: $UPLOAD_ONLY"
echo ""

# Show local structure
echo "Local directories:"
for dir in "$LOCAL_DIR"/*/; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -type f -name "*.pt" 2>/dev/null | wc -l)
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "  $(basename "$dir"): $count files ($size)"
    fi
done
echo ""

# Count totals
LOCAL_COUNT=$(find "$LOCAL_DIR" -type f \( -name "*.pt" -o -name "*.json" -o -name "*.txt" \) | wc -l)
LOCAL_SIZE=$(du -sh "$LOCAL_DIR" 2>/dev/null | cut -f1)
echo "Total local: $LOCAL_COUNT files ($LOCAL_SIZE)"
echo ""

# Build sync command
SYNC_ARGS="--region $AWS_REGION --storage-class $STORAGE_CLASS"

if [ "$DELETE_REMOTE" = "true" ]; then
    echo "WARNING: DELETE_REMOTE is enabled - files not in local will be deleted from S3!"
    SYNC_ARGS="$SYNC_ARGS --delete"
fi

# Confirm
read -p "Proceed with sync? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Upload local to S3
echo ""
echo "=== Uploading to S3 ==="
aws s3 sync "$LOCAL_DIR" "s3://$S3_BUCKET/$S3_PREFIX/" $SYNC_ARGS

# Download from S3 (if not upload only)
if [ "$UPLOAD_ONLY" != "true" ]; then
    echo ""
    echo "=== Downloading from S3 ==="
    aws s3 sync "s3://$S3_BUCKET/$S3_PREFIX/" "$LOCAL_DIR" --region "$AWS_REGION"
fi

echo ""
echo "=== Sync Complete ==="
echo ""
echo "Your data is available at:"
echo "  Local: $LOCAL_DIR"
echo "  S3: s3://$S3_BUCKET/$S3_PREFIX/"
echo ""
echo "Download to another machine:"
echo "  aws s3 sync s3://$S3_BUCKET/$S3_PREFIX/ /path/to/local/data"
