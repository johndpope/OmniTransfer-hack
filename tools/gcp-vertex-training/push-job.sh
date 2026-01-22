#!/bin/bash
# =============================================================================
# Submit OmniTransfer Training Job to Google Vertex AI
# =============================================================================
# Prerequisites:
#   1. gcloud CLI installed and authenticated
#   2. Docker image built and pushed to GCR (run build.sh first)
#   3. Training data uploaded to GCS bucket
#
# Usage: ./push-job.sh [--config job_config.yaml]
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${1:-job_config.yaml}"

# Load environment variables
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
fi

# Check required variables
if [ -z "$GCP_PROJECT" ]; then
    echo "Error: GCP_PROJECT environment variable is not set."
    echo "Set it with: export GCP_PROJECT=your-project-id"
    exit 1
fi

if [ -z "$GCS_BUCKET" ]; then
    echo "Error: GCS_BUCKET environment variable is not set."
    echo "Set it with: export GCS_BUCKET=your-bucket-name"
    exit 1
fi

GCP_REGION="${GCP_REGION:-us-central1}"

echo "=== OmniTransfer Vertex AI Training ==="
echo "Project: $GCP_PROJECT"
echo "Region: $GCP_REGION"
echo "Bucket: gs://$GCS_BUCKET"
echo "Config: $CONFIG_FILE"
echo ""

# Substitute environment variables in config
TEMP_CONFIG=$(mktemp)
envsubst < "$SCRIPT_DIR/$CONFIG_FILE" > "$TEMP_CONFIG"

echo "Resolved configuration:"
cat "$TEMP_CONFIG"
echo ""

# Confirm
read -p "Submit training job? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    rm "$TEMP_CONFIG"
    echo "Cancelled."
    exit 0
fi

# Submit job
echo ""
echo "Submitting job to Vertex AI..."
JOB_OUTPUT=$(gcloud ai custom-jobs create \
    --project="$GCP_PROJECT" \
    --region="$GCP_REGION" \
    --display-name="omnitransfer-$(date +%Y%m%d-%H%M%S)" \
    --config="$TEMP_CONFIG" 2>&1)

rm "$TEMP_CONFIG"

# Extract job name
JOB_NAME=$(echo "$JOB_OUTPUT" | grep "CustomJob" | sed -n 's/.*CustomJob \[\(.*\)\].*/\1/p')

if [ -z "$JOB_NAME" ]; then
    echo "Error: Could not extract job name"
    echo "Output: $JOB_OUTPUT"
    exit 1
fi

echo ""
echo "=== Job Submitted ==="
echo "Job: $JOB_NAME"
echo ""
echo "Monitor at:"
echo "  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=$GCP_PROJECT"
echo ""
echo "View logs:"
echo "  gcloud ai custom-jobs stream-logs $JOB_NAME --project=$GCP_PROJECT --region=$GCP_REGION"
echo ""

# Optional: Start polling for completion
if [ -f "$SCRIPT_DIR/utils/poll.py" ]; then
    echo "Starting job monitor..."
    python3 "$SCRIPT_DIR/utils/poll.py" "$JOB_NAME" --project "$GCP_PROJECT" --region "$GCP_REGION"
fi
