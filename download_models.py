"""
Model Downloader Utility
Downloads the pre-trained MediaPipe task models required for gesture recognition and face detection.
"""

import os
import sys
import urllib.request


MODELS = {
    "gesture_recognizer.task": "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/gesture_recognizer.task",
    "blaze_face_short_range.task": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
}


def download_with_progress(url, dest_path):
    print(f"Downloading: {os.path.basename(dest_path)}...")
    print(f"URL: {url}")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\rProgress: [{downloaded_mb:.2f} MB / {total_mb:.2f} MB] ({percent:.1f}%)")
            sys.stdout.flush()
        else:
            sys.stdout.write(f"\rDownloaded {downloaded / (1024 * 1024):.2f} MB")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
        print("\n[SUCCESS] Download completed.\n")
    except Exception as e:
        print(f"\n[ERROR] Failed to download {os.path.basename(dest_path)}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)


def main():
    print("=" * 60)
    print("       Face Attendance System - Model Downloader")
    print("=" * 60)

    for filename, url in MODELS.items():
        if os.path.exists(filename):
            print(f"[FOUND] {filename} already exists. Skipping.")
        else:
            download_with_progress(url, filename)

    print("-" * 60)
    print("InsightFace 'buffalo_l' Model Information:")
    print("InsightFace automatically downloads its 'buffalo_l' model pack (~288MB)")
    print("on the first run of 'enroll_employee.py' or 'attendance_system.py'.")
    print("Default cache directory: ~/.insightface/models/buffalo_l/")
    print("\nIf automatic download is slow or fails, download manually:")
    print("1. Download zip from:")
    print("   https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip")
    print("2. Extract the files into your user folder:")
    print("   - Windows: C:\\Users\\<Username>\\.insightface\\models\\buffalo_l\\")
    print("   - Linux/macOS: ~/.insightface/models/buffalo_l/")
    print("=" * 60)


if __name__ == "__main__":
    main()
