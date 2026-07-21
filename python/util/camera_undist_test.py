#!/usr/bin/env python3
"""
Camera Undistortion Test Utility

Applies camera calibration parameters from a JSON file to undistort a video,
saving the calibrated result video with '_calibrated' postfix added to the filename.

Usage Example:
    python camera_undist_test.py --video input_video.avi --json camera_calib.json
    # Generates: input_video_calibrated.avi
"""

import argparse
import json
import os
import sys
from pathlib import Path
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Undistort video using camera calibration parameters from JSON file."
    )
    parser.add_argument(
        "-v", "--video",
        type=str,
        required=True,
        help="Path to input AVI video file"
    )
    parser.add_argument(
        "-j", "--json",
        type=str,
        required=True,
        help="Path to calibration JSON file containing camera_matrix and distortion_coefficients"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Optional custom output video path. If not specified, '<filename>_calibrated.avi' is created."
    )
    parser.add_argument(
        "-a", "--alpha",
        type=float,
        default=0.0,
        help="Free scaling parameter between 0.0 (crop black borders) and 1.0 (retain all pixels). Default: 0.0"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display original and undistorted video side-by-side during processing"
    )
    return parser.parse_args()


def load_calibration_json(json_path: str):
    if not os.path.exists(json_path):
        print(f"[Error] Calibration JSON file not found: {json_path}")
        sys.exit(1)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "camera_matrix" not in data or "distortion_coefficients" not in data:
            print(f"[Error] JSON file missing 'camera_matrix' or 'distortion_coefficients': {json_path}")
            sys.exit(1)

        camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
        dist_coeffs = np.array(data["distortion_coefficients"], dtype=np.float64)

        return camera_matrix, dist_coeffs
    except Exception as e:
        print(f"[Error] Failed to parse JSON calibration file: {e}")
        sys.exit(1)


def generate_output_filename(video_path: str, custom_output: str = None) -> str:
    if custom_output:
        return custom_output

    p = Path(video_path)
    output_name = f"{p.stem}_calibrated{p.suffix}"
    return str(p.parent / output_name)


def undistort_video(video_path: str, json_path: str, output_path: str = None, alpha: float = 0.0, show_window: bool = False):
    camera_matrix, dist_coeffs = load_calibration_json(json_path)
    out_video_path = generate_output_filename(video_path, output_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Failed to open input video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    # Determine VideoWriter FourCC codec
    ext = Path(out_video_path).suffix.lower()
    if ext == ".mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")

    out_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
    if not out_writer.isOpened():
        # Fallback codec
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    print("=" * 70)
    print("Camera Video Undistortion Test")
    print("=" * 70)
    print(f"Input Video       : {video_path}")
    print(f"Calibration JSON  : {json_path}")
    print(f"Output Video      : {out_video_path}")
    print(f"Video Info        : {width}x{height} @ {fps:.2f} FPS ({total_frames} frames)")
    print(f"Alpha Scaling     : {alpha}")
    print("Camera Matrix (K):")
    print(np.array2string(camera_matrix, formatter={'float_kind': lambda x: f"{x:12.4f}"}))
    print("Distortion Coeffs:")
    print(np.array2string(dist_coeffs.ravel(), formatter={'float_kind': lambda x: f"{x:12.6f}"}))
    print("=" * 70)

    # Compute optimal camera matrix and undistortion maps
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (width, height), alpha, (width, height)
    )
    mapx, mapy = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_camera_matrix, (width, height), cv2.CV_32FC1
    )

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Apply undistortion remap
        undist_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

        # Write to output video
        out_writer.write(undist_frame)

        if frame_count % 30 == 0 or frame_count == total_frames:
            print(f"  Processing frame {frame_count:5d} / {total_frames:5d} ({(frame_count / total_frames * 100):5.1f}%)")

        if show_window:
            # Display side-by-side (Original vs Undistorted) resized for viewing
            vis_orig = cv2.resize(frame, (width // 2, height // 2))
            vis_undist = cv2.resize(undist_frame, (width // 2, height // 2))
            combined = np.hstack((vis_orig, vis_undist))
            cv2.putText(combined, "Original", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(combined, "Undistorted", (width // 2 + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("Original vs Calibrated Undistorted", combined)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                print("\n[Canceled by User]")
                break

    cap.release()
    out_writer.release()
    if show_window:
        cv2.destroyAllWindows()

    print("\n" + "=" * 70)
    print(f"[Success] Undistorted video successfully saved to:\n  {out_video_path}")
    print("=" * 70)


def main():
    args = parse_args()
    undistort_video(
        video_path=args.video,
        json_path=args.json,
        output_path=args.output,
        alpha=args.alpha,
        show_window=args.show
    )


if __name__ == "__main__":
    main()
