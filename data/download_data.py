"""
download_data.py
Simple helper to download GTZAN dataset metadata or provide instructions.
This script does not host copyrighted datasets. It will either:
 - download from a user-provided URL (if provided), or
 - print instructions to download GTZAN manually, then organize files.
"""

import argparse
import os
import shutil
from pathlib import Path

def prepare_dirs(target_dir):
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    (Path(target_dir) / "raw").mkdir(exist_ok=True)

def main(target_dir):
    prepare_dirs(target_dir)
    print("This script will help you prepare dataset folders for GTZAN or similar datasets.")
    print("GTZAN is not redistributed here. You can download it from:")
    print(" - http://marsyas.info/downloads/datasets.html (GTZAN)\n")
    print("After downloading, place the .wav files inside:")
    print(f"  {os.path.join(target_dir, 'raw')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=str, default="data", help="Where to place dataset folders")
    args = parser.parse_args()
    main(args.target_dir)
