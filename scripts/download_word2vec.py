"""
Download Vietnamese Word2Vec model from sonvx/word2vecVN repository.
"""

import os
import sys
from pathlib import Path
import urllib.request

# Model options
MODELS = {
    "1": {
        "name": "Baomoi 400D (window-size 5)",
        "url": "https://thiaisotajppub.s3-ap-northeast-1.amazonaws.com/publicfiles/baomoi.model.bin",
        "filename": "baomoi_400d.model.bin",
        "size": "~500MB"
    },
    "2": {
        "name": "Baomoi 300D (window-size 2)",
        "url": "https://thiaisotajppub.s3-ap-northeast-1.amazonaws.com/publicfiles/baomoi.window2.vn.model.bin.gz",
        "filename": "baomoi_300d.model.bin.gz",
        "size": "~300MB (compressed)"
    },
    "3": {
        "name": "Vietnamese Wiki",
        "url": "https://thiaisotajppub.s3-ap-northeast-1.amazonaws.com/publicfiles/wiki.vi.model.bin.gz",
        "filename": "wiki.vi.model.bin.gz",
        "size": "~1GB (compressed)"
    }
}

def download_with_progress(url, output_path):
    """Download file with progress bar."""
    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            downloaded_mb = count * block_size / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\rDownloading: {percent}% ({downloaded_mb:.1f}MB / {total_mb:.1f}MB)")
            sys.stdout.flush()

    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")

    urllib.request.urlretrieve(url, output_path, reporthook)
    print("\nDownload complete!")

def main():
    # Create models directory
    models_dir = Path(__file__).parent.parent / "models" / "word2vec"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Vietnamese Word2Vec Model Downloader")
    print("=" * 60)
    print("\nAvailable models:")
    for key, model in MODELS.items():
        print(f"{key}. {model['name']} - {model['size']}")

    print("\nRecommended: Option 1 (Baomoi 400D)")
    choice = input("\nSelect model (1-3) [1]: ").strip() or "1"

    if choice not in MODELS:
        print("Invalid choice!")
        sys.exit(1)

    model = MODELS[choice]
    output_path = models_dir / model["filename"]

    if output_path.exists():
        overwrite = input(f"\nFile already exists: {output_path}\nOverwrite? (y/n) [n]: ").strip().lower()
        if overwrite != 'y':
            print("Download cancelled.")
            sys.exit(0)

    print(f"\nSelected: {model['name']}")
    print(f"Size: {model['size']}")

    try:
        download_with_progress(model["url"], output_path)
        print(f"\nModel saved to: {output_path}")
        print("\nYou can now use this model with:")
        print(f"  --word2vec-model {output_path}")
    except Exception as e:
        print(f"\nError downloading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
