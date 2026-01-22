#!/bin/bash
# =============================================================================
# Build and Push OmniTransfer Training Container to GCR
# =============================================================================
# Prerequisites: Docker, gcloud CLI authenticated
# Usage: ./build.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load environment
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
fi

if [ -z "$GCP_PROJECT" ]; then
    echo "Error: GCP_PROJECT not set"
    echo "Usage: GCP_PROJECT=your-project ./build.sh"
    exit 1
fi

IMAGE_NAME="gcr.io/$GCP_PROJECT/omnitransfer-train"
TAG="${1:-latest}"

echo "=== Building OmniTransfer Training Container ==="
echo "Image: $IMAGE_NAME:$TAG"
echo ""

# Configure Docker for GCR
gcloud auth configure-docker gcr.io --quiet

# Build
echo "Building image..."
docker build -t "$IMAGE_NAME:$TAG" "$SCRIPT_DIR"

# Push
echo ""
echo "Pushing to GCR..."
docker push "$IMAGE_NAME:$TAG"

echo ""
echo "=== Build Complete ==="
echo "Image available at: $IMAGE_NAME:$TAG"
echo ""
echo "Next steps:"
echo "  1. Upload training data: gsutil -m cp -r /path/to/data gs://\$GCS_BUCKET/omnitransfer/processed/"
echo "  2. Submit job: ./push-job.sh"
