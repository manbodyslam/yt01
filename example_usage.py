"""
Example usage of YouTube Thumbnail Generator

This script demonstrates how to use the API programmatically
"""

import requests
import json
from pathlib import Path


def generate_thumbnail_example():
    """
    Example: Generate thumbnail via API
    """
    # API endpoint
    api_url = "http://localhost:8000"

    # Check if API is running
    try:
        response = requests.get(f"{api_url}/health")
        print(f"✓ API Status: {response.json()['status']}")
    except requests.exceptions.ConnectionError:
        print("✗ Error: API is not running. Please start with: python main.py")
        return

    # Generate thumbnail
    print("\nGenerating thumbnail...")

    payload = {
        "title": "เรื่องราวสุดฮา! ตอนที่ 1",
        "subtitle": "EP.1 - การผจญภัยเริ่มต้น",
        "num_characters": 3
    }

    response = requests.post(
        f"{api_url}/generate",
        json=payload
    )

    if response.status_code == 200:
        result = response.json()

        print(f"\n✓ Success!")
        print(f"  Thumbnail: {result['filename']}")
        print(f"  Path: {result['thumbnail_path']}")
        print(f"\nMetadata:")
        print(json.dumps(result['metadata'], indent=2, ensure_ascii=False))

        # Download thumbnail
        download_url = f"{api_url}/thumbnail/{result['filename']}"
        print(f"\nDownload URL: {download_url}")

    else:
        print(f"\n✗ Error: {response.status_code}")
        print(response.text)


def list_thumbnails_example():
    """
    Example: List all thumbnails
    """
    api_url = "http://localhost:8000"

    response = requests.get(f"{api_url}/thumbnails")

    if response.status_code == 200:
        thumbnails = response.json()

        print(f"\nFound {len(thumbnails)} thumbnails:")
        for thumb in thumbnails:
            print(f"  - {thumb}")
    else:
        print(f"Error: {response.status_code}")


def direct_pipeline_example():
    """
    Example: Use pipeline directly (without API)
    """
    from main import ThumbnailPipeline

    print("\nRunning pipeline directly...")

    pipeline = ThumbnailPipeline()

    result = pipeline.generate(
        title="ทดสอบระบบ",
        subtitle="Test Subtitle",
        num_characters=2
    )

    if result['success']:
        print(f"\n✓ Success!")
        print(f"  Thumbnail: {result['filename']}")
        print(f"  Path: {result['thumbnail_path']}")
    else:
        print(f"\n✗ Error: {result['error']}")


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("YouTube Thumbnail Generator - Example Usage")
    print("=" * 60)

    if len(sys.argv) > 1 and sys.argv[1] == "direct":
        # Direct pipeline usage
        direct_pipeline_example()
    else:
        # API usage
        print("\nMake sure you have:")
        print("1. Put images in workspace/raw/")
        print("2. Started the API server: python main.py")
        print("\nPress Enter to continue...")
        input()

        generate_thumbnail_example()
        list_thumbnails_example()

    print("\n" + "=" * 60)
