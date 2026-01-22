#!/usr/bin/env python3
"""Poll Vertex AI job status until completion."""

import argparse
import subprocess
import sys
import time


def get_job_state(job_name: str, project: str, region: str) -> str:
    """Get current job state from Vertex AI."""
    try:
        result = subprocess.run(
            [
                "gcloud", "ai", "custom-jobs", "describe", job_name,
                "--project", project,
                "--region", region,
                "--format", "value(state)"
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error getting job state: {e.stderr}")
        return "UNKNOWN"


def main():
    parser = argparse.ArgumentParser(description="Poll Vertex AI job status")
    parser.add_argument("job_name", help="Full job resource name")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--region", default="us-central1", help="GCP region")
    parser.add_argument("--interval", type=int, default=60, help="Poll interval in seconds")
    args = parser.parse_args()

    print(f"Monitoring job: {args.job_name}")
    print(f"Poll interval: {args.interval}s")
    print("")

    terminal_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}

    while True:
        state = get_job_state(args.job_name, args.project, args.region)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        if state in terminal_states:
            emoji = "" if state == "JOB_STATE_SUCCEEDED" else ""
            print(f"[{timestamp}] {emoji} Job completed: {state}")
            sys.exit(0 if state == "JOB_STATE_SUCCEEDED" else 1)
        else:
            print(f"[{timestamp}] Job state: {state}")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
