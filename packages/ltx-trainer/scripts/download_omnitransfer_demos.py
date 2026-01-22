#!/usr/bin/env python3
"""Download OmniTransfer demo videos from the website.

The OmniTransfer website has demo videos showing:
- Reference image (input)
- Reference video (effect/motion source)
- Output video (the generated result - our ground truth!)

This script downloads these to create proper training triplets.

Usage:
    python scripts/download_omnitransfer_demos.py --output-dir /media/2TB/omnitransfer_website_demos
"""

import argparse
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


def get_video_urls_from_page(url: str) -> dict:
    """Extract video URLs from the OmniTransfer page.

    The page uses lazy-loading, so we need to find data-src attributes.
    """
    print(f"Fetching page: {url}")

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    videos = {
        "effect_transfer": [],
        "motion_transfer": [],
        "camera_transfer": [],
        "id_transfer": [],
        "style_transfer": [],
    }

    # Find all video elements
    for video in soup.find_all('video'):
        # Check for data-src in source elements
        for source in video.find_all('source'):
            src = source.get('data-src') or source.get('src')
            if src:
                full_url = urljoin(url, src)
                print(f"Found video: {full_url}")

    # Also check for direct video src with data-src
    for video in soup.find_all('video'):
        src = video.get('data-src') or video.get('src')
        if src:
            full_url = urljoin(url, src)
            print(f"Found video (direct): {full_url}")

    # Look in scripts for video paths
    for script in soup.find_all('script'):
        if script.string:
            # Find mp4/webm references
            matches = re.findall(r'["\']([^"\']*\.(mp4|webm))["\']', script.string)
            for match in matches:
                video_path = match[0]
                if not video_path.startswith('http'):
                    video_path = urljoin(url, video_path)
                print(f"Found in script: {video_path}")

    return videos


def download_video(url: str, output_path: Path) -> bool:
    """Download a video file."""
    try:
        print(f"Downloading: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"  Saved to: {output_path}")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def download_with_ytdlp(url: str, output_dir: Path) -> bool:
    """Use yt-dlp to download videos from the page."""
    try:
        cmd = [
            "yt-dlp",
            "--no-playlist",
            "-o", str(output_dir / "%(title)s.%(ext)s"),
            url
        ]
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print(f"yt-dlp failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download OmniTransfer demo videos")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/media/2TB/omnitransfer_website_demos"),
        help="Output directory for downloaded videos",
    )
    parser.add_argument(
        "--url",
        default="https://pangzecheung.github.io/OmniTransfer/",
        help="OmniTransfer website URL",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("OmniTransfer Demo Video Downloader")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print()

    # Method 1: Parse HTML for video URLs
    print("Method 1: Parsing HTML for video URLs...")
    videos = get_video_urls_from_page(args.url)

    # Method 2: Try yt-dlp
    print("\nMethod 2: Trying yt-dlp...")
    download_with_ytdlp(args.url, args.output_dir)

    print("\n" + "=" * 60)
    print("MANUAL STEPS REQUIRED:")
    print("=" * 60)
    print("""
The OmniTransfer website uses lazy-loading for videos, which makes
automated downloading difficult. Please manually download the demo videos:

1. Open https://pangzecheung.github.io/OmniTransfer/ in your browser
2. Open Developer Tools (F12) → Network tab
3. Scroll through each demo section to trigger video loading
4. Filter by "media" or "mp4"
5. Right-click each video and "Save as..."

Organize the downloads as:
    {output_dir}/
        effect_transfer/
            ref_image_1.jpg       # Reference image (input)
            ref_video_1.mp4       # Effect source video
            output_1.mp4          # OmniTransfer result (GROUND TRUTH)
        motion_transfer/
            ref_image_1.jpg
            ref_video_1.mp4
            output_1.mp4
        ...

Then run the dataset preparation script to encode them.
""".format(output_dir=args.output_dir))


if __name__ == "__main__":
    main()
