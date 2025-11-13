#!/usr/bin/env python3
"""
Video Frame Counter
Recursively finds all .avi and .mp4 files in a directory and exports their frame counts to CSV.
"""

import argparse
import csv
from pathlib import Path
import cv2


def find_video_files(root_path, extensions=('.avi', '.mp4')):
    """
    Recursively find all video files with specified extensions.

    Args:
        root_path: Root directory to search
        extensions: Tuple of file extensions to search for

    Returns:
        List of Path objects for found video files
    """
    video_files = []
    root = Path(root_path)

    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root_path}")

    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root_path}")

    print(f"Searching for video files with extensions: {', '.join(extensions)}")
    for ext in extensions:
        # Use rglob for recursive search
        found_files = list(root.rglob(f"*{ext}"))
        for file in found_files:
            print(f"  Found: {file}")
        video_files.extend(found_files)

    return sorted(video_files)


def get_frame_count(video_path):
    """
    Get the total frame count of a video file.

    Args:
        video_path: Path to video file

    Returns:
        Total number of frames, or -1 if unable to read
    """
    try:
        print(f"  Opening video file: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  [ERROR] Could not open video file: {video_path}")
            return -1

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"  [SUCCESS] Frame count: {frame_count}")
        return frame_count
    except Exception as e:
        print(f"  [ERROR] Exception while processing {video_path}: {e}")
        return -1


def process_videos(root_path, output_csv):
    """
    Process all video files and export frame counts to CSV.

    Args:
        root_path: Root directory to search for videos
        output_csv: Output CSV filename
    """
    print(f"=" * 80)
    print(f"Starting video frame counter")
    print(f"Search path: {root_path}")
    print(f"Output CSV: {output_csv}")
    print(f"=" * 80)
    print()

    video_files = find_video_files(root_path)

    if not video_files:
        print("\nNo video files found.")
        return

    print(f"\n{'=' * 80}")
    print(f"Total files found: {len(video_files)}")
    print(f"{'=' * 80}\n")

    # Prepare data for CSV
    results = []
    for idx, video_path in enumerate(video_files, 1):
        print(f"[{idx}/{len(video_files)}] Processing: {video_path}")
        frame_count = get_frame_count(video_path)
        results.append({
            'file_path': str(video_path),
            'frame_count': frame_count
        })
        print()

    # Write to CSV
    print(f"{'=' * 80}")
    print(f"Writing results to CSV: {output_csv}")
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file_path', 'frame_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(results)

    # Calculate total frames (excluding failed files with -1)
    total_frames = sum(r['frame_count'] for r in results if r['frame_count'] > 0)
    failed_count = sum(1 for r in results if r['frame_count'] == -1)

    print(f"[SUCCESS] Results exported to: {output_csv}")
    print(f"Total files processed: {len(results)}")
    if failed_count > 0:
        print(f"Failed files: {failed_count}")
    print(f"Total frames across all files: {total_frames:,}")
    print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(
        description='Count frames in video files and export to CSV'
    )
    parser.add_argument(
        '--path',
        required=True,
        help='Root directory to search for video files'
    )
    parser.add_argument(
        '--csv',
        required=True,
        help='Output CSV filename'
    )

    args = parser.parse_args()

    try:
        process_videos(args.path, args.csv)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
