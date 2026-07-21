#!/usr/bin/env python3
"""
Camera Calibration Utility using OpenCV

Calibrates camera intrinsic parameters and distortion coefficients using an AVI video of a checkerboard pattern.

Usage Example:
    python camera_calibration.py --video calibration.avi --cols 9 --rows 6 --square_size 25.0 --output camera_calib.json
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate camera intrinsic matrix and distortion coefficients from checkerboard video."
    )
    parser.add_argument(
        "-v", "--video",
        type=str,
        required=True,
        help="Path to input AVI video file"
    )
    parser.add_argument(
        "-c", "--cols",
        type=int,
        required=True,
        help="Number of inner corners along checkerboard width (columns)"
    )
    parser.add_argument(
        "-r", "--rows",
        type=int,
        required=True,
        help="Number of inner corners along checkerboard height (rows)"
    )
    parser.add_argument(
        "-s", "--square_size",
        type=float,
        required=True,
        help="Size of a single checkerboard square in mm"
    )
    parser.add_argument(
        "-i", "--interval",
        type=int,
        default=10,
        help="Frame sampling interval (process every N-th frame, default: 10)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="camera_calibration.json",
        help="Path to save output JSON calibration file (default: camera_calibration.json)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display detected checkerboard corners in a window during processing"
    )
    return parser.parse_args()


def calibrate_camera_from_video(
    video_path: str,
    cols: int,
    rows: int,
    square_size_mm: float,
    sample_interval: int = 10,
    show_window: bool = False
):
    if not os.path.exists(video_path):
        print(f"[Error] Video file not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Failed to open video file: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("=" * 70)
    print("Camera Calibration Program")
    print("=" * 70)
    print(f"Video File       : {video_path}")
    print(f"Resolution       : {width}x{height} @ {fps:.2f} FPS")
    print(f"Total Frames     : {total_frames}")
    print(f"Checkerboard     : {cols} x {rows} inner corners")
    print(f"Square Size      : {square_size_mm} mm")
    print(f"Frame Interval   : Every {sample_interval} frame(s)")
    print("=" * 70)

    # Termination criteria for sub-pixel corner refinement
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare 3D object points in real world coordinates (in mm)
    # E.g. (0,0,0), (25,0,0), (50,0,0)...
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm

    # Arrays to store 3D object points and 2D image points from all valid frames
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    frame_idx = 0
    detected_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Skip frames according to sampling interval
        if frame_idx % sample_interval != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find checkerboard corners
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)

        if found:
            detected_count += 1
            # Refine corner locations to sub-pixel accuracy
            corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)

            objpoints.append(objp)
            imgpoints.append(corners_subpix)

            print(f"  [Frame {frame_idx:5d}/{total_frames:5d}] Checkerboard detected! (Total valid frames: {detected_count})")

            if show_window:
                vis_frame = cv2.drawChessboardCorners(frame.copy(), (cols, rows), corners_subpix, found)
                cv2.imshow("Checkerboard Detection", vis_frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to cancel
                    break
        else:
            print(f"  [Frame {frame_idx:5d}/{total_frames:5d}] Checkerboard not found.")

    cap.release()
    if show_window:
        cv2.destroyAllWindows()

    if detected_count == 0:
        print("\n[Error] No valid checkerboard corners detected in any sampled frames.")
        sys.exit(1)

    print("\nProcessing camera calibration...")

    # Calibrate camera using OpenCV
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (width, height), None, None
    )

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    dist_list = dist_coeffs.ravel().tolist()

    print("\n" + "=" * 70)
    print("CALIBRATION RESULTS")
    print("=" * 70)
    print(f"RMS Reprojection Error : {rms:.6f} pixels")
    print(f"Image Resolution       : {width} x {height}")
    print("\nCamera Matrix (K):")
    print(f"  fx = {fx:12.4f},  fy = {fy:12.4f}")
    print(f"  cx = {cx:12.4f},  cy = {cy:12.4f}")
    print("  Full Matrix:")
    print(np.array2string(camera_matrix, formatter={'float_kind': lambda x: f"{x:12.4f}"}))

    print("\nDistortion Coefficients (k1, k2, p1, p2, k3):")
    print(np.array2string(dist_coeffs.ravel(), formatter={'float_kind': lambda x: f"{x:12.6f}"}))
    print("=" * 70)

    calib_data = {
        "reprojection_error_rms": rms,
        "image_size": {
            "width": width,
            "height": height
        },
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_list,
        "intrinsics": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy
        },
        "checkerboard_config": {
            "cols": cols,
            "rows": rows,
            "square_size_mm": square_size_mm,
            "num_calibration_frames": detected_count
        }
    }

    return calib_data


def main():
    args = parse_args()
    calib_result = calibrate_camera_from_video(
        video_path=args.video,
        cols=args.cols,
        rows=args.rows,
        square_size_mm=args.square_size,
        sample_interval=args.interval,
        show_window=args.show
    )

    # Save to JSON output
    output_path = args.output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(calib_result, f, indent=4)

    print(f"\n[Success] Calibration parameters saved to: {output_path}")


if __name__ == "__main__":
    main()
