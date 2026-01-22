# Vast.ai Cloud Training for OmniTransfer

This directory contains Terraform configuration and scripts for running OmniTransfer training on Vast.ai cloud GPUs.

## Why Cloud Training?

- **A100 80GB**: Much faster than local RTX cards for LTX-2 19B training
- **No VRAM constraints**: Full LoRA rank 64, video logging, etc.
- **Cost-effective**: ~$1.50-2.50/hr for A100 80GB on-demand
- **Checkpoint protection**: Auto-sync to S3 every 30 minutes

## Prerequisites

1. **Vast.ai Account**: https://vast.ai/
2. **AWS S3 Bucket**: For training data and checkpoints
3. **Terraform**: `brew install terraform` or `apt install terraform`
4. **Vast.ai CLI**: `pip install vastai`

## Quick Start

### 1. Configure Credentials

```bash
# Set up Vast.ai CLI
vastai set api-key YOUR_API_KEY

# Create terraform.tfvars from example
cp terraform.tfvars.example terraform.tfvars

# Edit with your credentials
vim terraform.tfvars
```

### 2. Upload Training Data to S3

```bash
# Create .env for local scripts
cp .env.example .env
vim .env  # Add AWS credentials

# Upload preprocessed training data
./scripts/upload_training_data.sh /media/2TB/omnitransfer_unified_5task
```

### 3. Launch Cloud Instance

```bash
# Initialize Terraform
terraform init

# Preview what will be created
terraform plan

# Create the instance
terraform apply
```

### 4. Start Training

```bash
# Get SSH connection
vastai ssh-url <instance_id>

# SSH into the instance
ssh -p <port> root@<host>

# Start training
cd /workspace/ltx2-omnitransfer
bash tools/vast-cloud-training/scripts/train_omnitransfer.sh
```

### 5. Monitor Training

- **tmux**: Training runs in tmux session, attach with `tmux attach -t training`
- **W&B**: Watch progress at https://wandb.ai/your-project
- **GPU**: `watch -n1 nvidia-smi`
- **Logs**: `tail -f outputs/omnitransfer_cloud/training.log`

### 6. Download Checkpoints

```bash
# Download checkpoints locally
./scripts/download_checkpoints.sh ./outputs/cloud_checkpoints
```

### 7. Cleanup

```bash
# Destroy the instance when done
terraform destroy
```

## Configuration Reference

### terraform.tfvars Options

| Variable | Default | Description |
|----------|---------|-------------|
| `vast_api_key` | (required) | Your Vast.ai API key |
| `aws_access_key_id` | (required) | AWS access key for S3 |
| `aws_secret_access_key` | (required) | AWS secret key for S3 |
| `s3_bucket` | omnitransfer-training | S3 bucket name |
| `preferred_gpu` | A100 | GPU model (A100, H100) |
| `interruptible` | false | Use spot pricing (cheaper but risky) |
| `num_gpus` | 1 | Number of GPUs |
| `disk_gb` | 200 | Disk space in GB |
| `training_steps` | 10000 | Training steps |
| `max_runtime_hours` | 24 | Auto-shutdown timer |
| `wandb_api_key` | "" | W&B API key for logging |

### GPU Selection

| GPU | VRAM | ~Price/hr | Notes |
|-----|------|-----------|-------|
| A100 SXM4 | 80GB | $1.50-2.50 | Best value for training |
| H100 SXM5 | 80GB | $3.00-4.00 | Faster but pricier |
| A100 PCIe | 40GB | $1.00-1.50 | May need smaller batch |

## Training Configuration

The cloud training script generates a config optimized for A100 80GB:

```yaml
lora:
  rank: 64        # Full rank (vs 32 on 32GB local)
  alpha: 64

optimization:
  batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1e-5

training_strategy:
  log_reconstructions: true    # Visual debugging enabled
  log_video_comparisons: true  # Video comparisons enabled
```

## Checkpoint Protection

The training script implements multiple safeguards:

1. **Periodic S3 Sync**: Every 30 minutes, checkpoints are uploaded to S3
2. **Auto-shutdown**: Instance shuts down after `max_runtime_hours`
3. **Pre-shutdown Upload**: Before shutdown, final checkpoints are synced

## Resuming Training

If training is interrupted:

```bash
# Resume from latest checkpoint
bash scripts/train_omnitransfer.sh --resume /path/to/checkpoint

# Or it will auto-detect existing checkpoints and prompt
```

## Cost Estimation

| Duration | A100 On-Demand | A100 Spot |
|----------|----------------|-----------|
| 10k steps (~8h) | ~$16-20 | ~$8-12 |
| 30k steps (~24h) | ~$48-60 | ~$24-36 |
| Full training | ~$100-150 | ~$50-75 |

*Note: Spot instances can be preempted. Use on-demand for critical training.*

## Troubleshooting

### Instance won't start
```bash
# Check available offers
vastai search offers "gpu_ram>=80 reliability>=0.95"
```

### Training OOM
- Reduce `gradient_accumulation_steps` in the generated config
- Disable video logging: set `log_video_comparisons: false`

### Checkpoints not syncing
```bash
# Manual sync
aws s3 sync /outputs/omnitransfer_cloud s3://your-bucket/checkpoints/ --region us-east-1
```

### Can't SSH
```bash
# Get fresh SSH URL
vastai ssh-url <instance_id>

# Check instance status
vastai show instances
```
