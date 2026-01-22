# S3 Video Storage for OmniTransfer Training

Upload and download training videos to/from AWS S3 for distributed training workflows.

## Setup

1. Copy the sample environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your AWS credentials:
   ```bash
   AWS_ACCESS_KEY_ID=AKIA...
   AWS_SECRET_ACCESS_KEY=your-secret-key
   S3_BUCKET=omnitransfer-training-videos
   S3_REGION=us-east-1
   ```

3. Install AWS CLI:
   ```bash
   pip install awscli boto3
   ```

## Usage

### Upload Videos to S3

```bash
# Upload a directory of videos
./scripts/upload_videos.sh /media/2TB/omnitransfer_training/videos

# Upload with custom prefix
S3_PREFIX=raw_demos ./scripts/upload_videos.sh /path/to/videos
```

### Download Videos from S3

```bash
# Download all videos
./scripts/download_videos.sh /media/2TB/downloaded_videos

# Download specific prefix
S3_PREFIX=website_demos ./scripts/download_videos.sh /path/to/output
```

### Sync Training Data

Bidirectional sync for keeping local and S3 in sync:

```bash
# Sync entire training dataset
./scripts/sync_training_data.sh /media/2TB/omnitransfer_unified_5task

# Upload only (don't download)
UPLOAD_ONLY=true ./scripts/sync_training_data.sh /path/to/data
```

## Directory Structure on S3

```
s3://omnitransfer-training-videos/
├── raw_demos/              # Original demo videos
│   ├── website_demos/      # OmniTransfer website examples
│   ├── movie_weaver/       # Movie Weaver demos
│   └── custom/             # Your custom videos
├── processed/              # Encoded latents ready for training
│   ├── latents/
│   ├── conditions/
│   └── reference_latents/
└── checkpoints/            # Training checkpoints
    └── lora_weights/
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AWS_ACCESS_KEY_ID` | Yes | - | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | Yes | - | AWS secret key |
| `S3_BUCKET` | Yes | - | S3 bucket name |
| `S3_REGION` | No | us-east-1 | AWS region |
| `S3_PREFIX` | No | "" | Prefix path in bucket |
| `UPLOAD_ONLY` | No | false | Skip downloads in sync |
| `DELETE_REMOTE` | No | false | Delete remote files not in local |

## Pricing Estimate

| Data Volume | Upload | Download | Storage/month |
|-------------|--------|----------|---------------|
| 10 GB | ~$0.09 | ~$0.09 | ~$0.23 |
| 100 GB | ~$0.90 | ~$0.90 | ~$2.30 |
| 1 TB | ~$9.00 | ~$9.00 | ~$23.00 |

*Prices approximate for US East region with S3 Standard storage*

## Credits

Adapted from [vast-infinitystar](https://github.com/johndpope/imf-infinity) for OmniTransfer training workflows.
