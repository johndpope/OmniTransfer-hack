terraform {
  required_version = ">= 1.5"
}

# =============================================================================
# Vast.ai GPU Instance for OmniTransfer Training
# =============================================================================
# Cloud training for LTX-2 OmniTransfer on A100 80GB GPUs
# Uploads checkpoints to S3, downloads training data from S3
#
# Prerequisites:
#   pip install vastai
#   vastai set api-key YOUR_API_KEY
# =============================================================================

variable "vast_api_key" {
  description = "Your Vast.ai API key"
  type        = string
  sensitive   = true
}

variable "hf_token" {
  description = "HuggingFace token for model downloads"
  type        = string
  sensitive   = true
  default     = ""
}

variable "aws_access_key_id" {
  description = "AWS Access Key ID for S3"
  type        = string
  sensitive   = true
}

variable "aws_secret_access_key" {
  description = "AWS Secret Access Key for S3"
  type        = string
  sensitive   = true
}

variable "s3_bucket" {
  description = "S3 bucket name for training data and checkpoints"
  type        = string
  default     = "omnitransfer-training"
}

variable "s3_region" {
  description = "AWS region for S3"
  type        = string
  default     = "us-east-1"
}

variable "wandb_api_key" {
  description = "Weights & Biases API key for experiment tracking"
  type        = string
  sensitive   = true
  default     = ""
}

variable "wandb_project" {
  description = "W&B project name"
  type        = string
  default     = "omnitransfer-unified"
}

variable "gpu_query" {
  description = "Vast.ai search query for GPU offers"
  type        = string
  # A100 80GB optimal for OmniTransfer training with INT8 quantization
  default = "gpu_ram>=80 num_gpus>=1 inet_down>=200 disk_space>=200 reliability>=0.95 rentable=true"
}

variable "preferred_gpu" {
  description = "Preferred GPU model (A100_SXM4, H100_SXM5, etc.)"
  type        = string
  default     = "A100"
}

variable "interruptible" {
  description = "Use interruptible/spot instances (cheaper but can be preempted)"
  type        = bool
  default     = false  # Default to on-demand for training stability
}

variable "num_gpus" {
  description = "Number of GPUs to request"
  type        = number
  default     = 1
}

variable "disk_gb" {
  description = "Disk space in GB (need space for model + data + checkpoints)"
  type        = number
  default     = 200
}

variable "image" {
  description = "Docker image to use (NGC images have flash-attn pre-installed)"
  type        = string
  # NVIDIA NGC image with PyTorch 24.01 has flash-attn pre-installed
  default     = "nvcr.io/nvidia/pytorch:24.01-py3"
}

variable "max_price_per_hour" {
  description = "Maximum price per hour in USD"
  type        = number
  default     = 2.50
}

variable "instance_label" {
  description = "Label for the instance"
  type        = string
  default     = "omnitransfer-training"
}

variable "training_config" {
  description = "Training config YAML file to use"
  type        = string
  default     = "ltx2_omnitransfer_unified_5task.yaml"
}

variable "training_steps" {
  description = "Number of training steps"
  type        = number
  default     = 10000
}

variable "max_runtime_hours" {
  description = "Auto-shutdown after X hours (0 to disable)"
  type        = number
  default     = 24
}

variable "checkpoint_interval" {
  description = "Checkpoint save interval (steps)"
  type        = number
  default     = 500
}

variable "s3_data_prefix" {
  description = "S3 prefix for training data"
  type        = string
  default     = "processed/omnitransfer_unified_5task"
}

variable "github_repo" {
  description = "GitHub repo to clone"
  type        = string
  default     = "johndpope/ltx2-omnitransfer"
}

variable "github_branch" {
  description = "Git branch to use"
  type        = string
  default     = "main"
}

# Store instance ID for destroy operations
resource "local_file" "instance_id" {
  filename = "${path.module}/.instance_id"
  content  = ""

  lifecycle {
    ignore_changes = [content]
  }
}

# Main provisioning resource
resource "null_resource" "vast_ai_instance" {
  depends_on = [local_file.instance_id]

  triggers = {
    version = "1"
    # Store API key in triggers so it's available at destroy time
    vast_api_key = var.vast_api_key
  }

  provisioner "local-exec" {
    command = <<-EOT
      #!/bin/bash
      set -e

      echo "=== Vast.ai OmniTransfer Training Setup ==="
      echo ""

      if ! command -v vastai &> /dev/null; then
        echo "Error: vastai CLI not found. Install with: pip install vastai"
        exit 1
      fi

      QUERY="${var.gpu_query}"
      if [ "${var.preferred_gpu}" != "" ]; then
        QUERY="gpu_name=${var.preferred_gpu} $QUERY"
      fi
      QUERY="num_gpus>=${var.num_gpus} $QUERY"

      echo "Searching for offers with query: $QUERY"
      OFFERS=$(vastai search offers "$QUERY" --raw 2>/dev/null || echo "[]")

      if [ "$OFFERS" = "[]" ] || [ -z "$OFFERS" ]; then
        echo "No offers found with preferred GPU. Trying fallback..."
        QUERY="${var.gpu_query} num_gpus>=${var.num_gpus}"
        OFFERS=$(vastai search offers "$QUERY" --raw 2>/dev/null || echo "[]")
      fi

      OFFER_ID=$(echo "$OFFERS" | jq -r 'sort_by(.dph_total) | .[0].id // empty' 2>/dev/null)
      OFFER_GPU=$(echo "$OFFERS" | jq -r 'sort_by(.dph_total) | .[0].gpu_name // empty' 2>/dev/null)
      OFFER_PRICE=$(echo "$OFFERS" | jq -r 'sort_by(.dph_total) | .[0].dph_total // empty' 2>/dev/null)

      if [ -z "$OFFER_ID" ]; then
        echo "Error: No suitable GPU offers found!"
        exit 1
      fi

      echo "Selected offer: ID=$OFFER_ID GPU=$OFFER_GPU Price=\$$OFFER_PRICE/hr"

      # Environment variables for the instance
      ENV_VARS="-e AWS_ACCESS_KEY_ID=${var.aws_access_key_id} -e AWS_SECRET_ACCESS_KEY=${var.aws_secret_access_key} -e S3_BUCKET=${var.s3_bucket} -e S3_REGION=${var.s3_region} -e S3_DATA_PREFIX=${var.s3_data_prefix} -e TRAINING_CONFIG=${var.training_config} -e TRAINING_STEPS=${var.training_steps} -e MAX_RUNTIME_HOURS=${var.max_runtime_hours} -e CHECKPOINT_INTERVAL=${var.checkpoint_interval} -e GITHUB_REPO=${var.github_repo} -e GITHUB_BRANCH=${var.github_branch}"

      if [ "${var.hf_token}" != "" ]; then
        ENV_VARS="$ENV_VARS -e HF_TOKEN=${var.hf_token}"
      fi

      if [ "${var.wandb_api_key}" != "" ]; then
        ENV_VARS="$ENV_VARS -e WANDB_API_KEY=${var.wandb_api_key} -e WANDB_PROJECT=${var.wandb_project}"
      fi

      INTERRUPTIBLE_FLAG=""
      if [ "${var.interruptible}" = "true" ]; then
        INTERRUPTIBLE_FLAG="--bid"
      fi

      # Onstart command - NGC image has flash-attn pre-installed
      ONSTART_CMD="bash -c 'apt-get update && apt-get install -y git git-lfs awscli tmux htop && pip install uv wandb && git clone https://github.com/${var.github_repo}.git /workspace/ltx2-omnitransfer && cd /workspace/ltx2-omnitransfer && git checkout ${var.github_branch} && uv sync && python -c \"from flash_attn import flash_attn_func; print(flash_attn OK)\" && echo \"Setup complete - ready for training!\"'"

      RESULT=$(vastai create instance $OFFER_ID \
        --image "${var.image}" \
        --disk ${var.disk_gb} \
        --ssh \
        --direct \
        --label "${var.instance_label}" \
        --onstart-cmd "$ONSTART_CMD" \
        $INTERRUPTIBLE_FLAG \
        $ENV_VARS \
        --raw 2>&1)

      echo "Create result: $RESULT"

      INSTANCE_ID=$(echo "$RESULT" | jq -r '.new_contract // empty' 2>/dev/null)
      if [ -z "$INSTANCE_ID" ]; then
        INSTANCE_ID=$(echo "$RESULT" | jq -r '.id // empty' 2>/dev/null)
      fi

      if [ ! -z "$INSTANCE_ID" ]; then
        echo "$INSTANCE_ID" > ${path.module}/.instance_id
        echo "Instance ID: $INSTANCE_ID"
      fi

      echo ""
      echo "=== Waiting for instance to start ==="
      sleep 30
      vastai show instances --raw | jq '.[] | select(.label == "${var.instance_label}")' 2>/dev/null || vastai show instances

      echo ""
      echo "=== Next Steps ==="
      echo "1. SSH into instance: vastai ssh-url <instance_id>"
      echo "2. Run training: bash /workspace/ltx2-omnitransfer/tools/vast-cloud-training/scripts/train_omnitransfer.sh"
    EOT

    environment = {
      VASTAI_API_KEY = var.vast_api_key
    }
  }

  provisioner "local-exec" {
    when    = destroy
    command = <<-EOT
      #!/bin/bash
      if [ -f "${path.module}/.instance_id" ]; then
        INSTANCE_ID=$(cat ${path.module}/.instance_id)
        if [ ! -z "$INSTANCE_ID" ]; then
          echo "Destroying instance: $INSTANCE_ID"
          vastai destroy instance $INSTANCE_ID --raw || true
          rm -f ${path.module}/.instance_id
        fi
      fi
    EOT

    environment = {
      VASTAI_API_KEY = self.triggers.vast_api_key
    }
  }
}

output "usage_info" {
  value = <<-EOT

    === Vast.ai OmniTransfer Training ===

    Check instance: vastai show instances
    SSH connection: vastai ssh-url <instance_id>

    On the instance, run:
      cd /workspace/ltx2-omnitransfer
      bash tools/vast-cloud-training/scripts/train_omnitransfer.sh

    Training checkpoints uploaded to: s3://${var.s3_bucket}/checkpoints/
    Training data downloaded from: s3://${var.s3_bucket}/${var.s3_data_prefix}/

    Monitor training: https://wandb.ai/${var.wandb_project}

    Destroy: terraform destroy
  EOT
}
