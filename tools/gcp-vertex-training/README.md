# Google Vertex AI Training for OmniTransfer

Train OmniTransfer on Google Cloud Vertex AI with A100/H100 GPUs.

## Prerequisites

1. **Google Cloud CLI** installed and authenticated:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Docker** installed for building training images

3. **Enable APIs**:
   ```bash
   gcloud services enable aiplatform.googleapis.com
   gcloud services enable containerregistry.googleapis.com
   ```

## Quick Start

### 1. Set Environment Variables

```bash
export GCP_PROJECT="your-project-id"
export GCS_BUCKET="your-training-bucket"
export WANDB_API_KEY="your-wandb-key"  # Optional
export HF_TOKEN="your-hf-token"        # Optional
```

Or create a `.env` file:
```bash
cp .env.example .env
# Edit .env with your values
```

### 2. Create GCS Bucket

```bash
gsutil mb -l us-central1 gs://$GCS_BUCKET
```

### 3. Upload Training Data

```bash
# Upload preprocessed OmniTransfer data
gsutil -m cp -r /path/to/omnitransfer_unified_5task/* gs://$GCS_BUCKET/omnitransfer/processed/

# Optional: Upload model weights (faster than downloading from HF)
gsutil cp /path/to/ltx-2-19b-dev.safetensors gs://$GCS_BUCKET/models/
gsutil -m cp -r /path/to/gemma gs://$GCS_BUCKET/models/gemma/
```

### 4. Build and Push Docker Image

```bash
./build.sh
```

### 5. Submit Training Job

```bash
./push-job.sh
```

## Configuration

### GPU Options

Edit `job_config.yaml` to change GPU type:

| Machine Type | GPU | VRAM | Cost (approx) |
|-------------|-----|------|---------------|
| `a2-highgpu-1g` | A100 40GB | 40GB | ~$3.67/hr |
| `a2-ultragpu-1g` | A100 80GB | 80GB | ~$5.12/hr |
| `a3-highgpu-1g` | H100 80GB | 80GB | ~$10.20/hr |

### Training Parameters

Environment variables in `job_config.yaml`:
- `TRAINING_STEPS`: Number of training steps (default: 10000)
- `CHECKPOINT_INTERVAL`: Save checkpoint every N steps (default: 500)
- `WANDB_PROJECT`: W&B project name
- `GCS_DATA_PREFIX`: GCS path to training data
- `GCS_CHECKPOINT_PREFIX`: GCS path for checkpoint uploads

## Monitoring

### View Job Status

```bash
gcloud ai custom-jobs list --project=$GCP_PROJECT --region=us-central1
```

### Stream Logs

```bash
gcloud ai custom-jobs stream-logs JOB_NAME --project=$GCP_PROJECT --region=us-central1
```

### Web Console

https://console.cloud.google.com/vertex-ai/training/custom-jobs

### Weights & Biases

If configured, training metrics are logged to:
https://wandb.ai/YOUR_ENTITY/omnitransfer-vertex

## Cost Estimates

| GPU | Time for 10k steps | Estimated Cost |
|-----|-------------------|----------------|
| A100 40GB | ~10-14 hours | ~$40-50 |
| A100 80GB | ~8-12 hours | ~$45-60 |
| H100 80GB | ~4-6 hours | ~$45-60 |

*Costs vary by region and spot availability*

## File Structure

```
gcp-vertex-training/
├── Dockerfile           # Training container
├── build.sh            # Build and push container
├── push-job.sh         # Submit training job
├── start.sh            # Container entrypoint
├── job_config.yaml     # Vertex AI job configuration
├── .env.example        # Environment template
└── utils/
    └── poll.py         # Job status monitor
```

## Troubleshooting

### "Quota exceeded"
Request GPU quota increase at:
https://console.cloud.google.com/iam-admin/quotas

### Container fails to start
Check container logs:
```bash
gcloud ai custom-jobs stream-logs JOB_NAME --project=$GCP_PROJECT
```

### OOM errors
- Switch to A100 80GB or H100
- Enable INT8 quantization in config
- Reduce batch size
