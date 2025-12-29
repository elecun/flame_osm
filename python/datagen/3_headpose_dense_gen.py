#!/usr/bin/env python3
"""
Head Pose Generator
Processes face landmarks CSV and generates 3D head pose estimation (position and orientation).
Uses PnP algorithm to estimate head pose from 2D facial landmarks.
"""

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np


def detect_csv_encoding(file_path: Path) -> str:
    """
    Detect CSV file encoding by trying multiple encodings.

    Args:
        file_path: Path to CSV file

    Returns:
        Detected encoding string
    """
    encodings = ['utf-8', 'cp949', 'euc-kr', 'latin-1']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read()
            return encoding
        except UnicodeDecodeError:
            continue

    return 'utf-8'


def extract_id_from_filename(filename: str) -> Optional[int]:
    """
    Extract numeric ID from the end of filename.

    Args:
        filename: Filename to extract ID from (e.g., 'cam_0.avi', 'face_kps_0.csv')

    Returns:
        Numeric ID or None if not found
    """
    match = re.search(r'_(\d+)\.[^.]+$', filename)
    if match:
        return int(match.group(1))
    return None


def find_file_pairs(directory: Path, selected_ids: Optional[List[int]] = None) -> Dict[int, Dict[str, Path]]:
    """
    Find and pair AVI and face landmarks CSV files by their ID suffix.

    Args:
        directory: Directory to search for files
        selected_ids: Optional list of camera IDs to filter (e.g., [0, 2, 4, 6])

    Returns:
        Dictionary mapping ID to {'avi': Path, 'csv': Path}
    """
    print(f"{'=' * 80}")
    print(f"Searching for file pairs in: {directory}")
    print(f"{'=' * 80}\n")

    # Find all AVI files in camera subdirectory
    camera_dir = directory / "camera"
    print(f"Searching for AVI files in camera directory...")
    avi_files = {}
    all_avi = list(camera_dir.glob('*.avi'))
    print(f"Found {len(all_avi)} AVI file(s)\n")

    for avi_file in all_avi:
        if avi_file.name.startswith('._'):
            continue
        file_id = extract_id_from_filename(avi_file.name)
        if file_id is not None:
            # Filter by selected IDs if provided
            if selected_ids is not None and file_id not in selected_ids:
                continue
            avi_files[file_id] = avi_file
            print(f"  Found AVI file: {avi_file.name} (ID: {file_id})")

    # Find all face_kps CSV files in base directory
    print(f"\nSearching for face_kps CSV files...")
    csv_files = {}
    all_csv = list(directory.glob('face_kps_*.csv'))
    print(f"Found {len(all_csv)} face_kps CSV file(s)\n")

    for csv_file in all_csv:
        if csv_file.name.startswith('._'):
            continue
        file_id = extract_id_from_filename(csv_file.name)
        if file_id is not None:
            # Filter by selected IDs if provided
            if selected_ids is not None and file_id not in selected_ids:
                continue
            csv_files[file_id] = csv_file
            print(f"  Found CSV file: {csv_file.name} (ID: {file_id})")

    print(f"\nAVI files with IDs: {sorted(avi_files.keys())}")
    print(f"CSV files with IDs: {sorted(csv_files.keys())}")

    # Pair files by ID
    pairs = {}
    print(f"\n{'=' * 40}")
    print("Pairing files by ID:")
    print(f"{'=' * 40}")

    for file_id in avi_files.keys():
        if file_id in csv_files:
            pairs[file_id] = {
                'avi': avi_files[file_id],
                'csv': csv_files[file_id]
            }
            print(f"  [PAIRED] ID {file_id}: {avi_files[file_id].name} <-> {csv_files[file_id].name}")
        else:
            print(f"  [WARNING] ID {file_id}: AVI file has no matching face_kps CSV")

    for file_id in csv_files.keys():
        if file_id not in avi_files:
            print(f"  [WARNING] ID {file_id}: CSV file has no matching AVI")

    print(f"\n{'=' * 80}")
    print(f"Total pairs found: {len(pairs)}")
    print(f"{'=' * 80}\n")

    return pairs


def read_face_landmarks_csv(csv_path: Path) -> Tuple[List[float], np.ndarray, Optional[List[int]]]:
    """
    Read face landmarks from CSV file.
    Supports MediaPipe Face Mesh format (landmark_N_x, landmark_N_y, landmark_N_z).
    Extracts only 2D landmarks (x, y).

    Args:
        csv_path: Path to face_kps CSV file

    Returns:
        Tuple of (timestamps, landmarks_array, valid_mask)
        landmarks_array shape: (num_frames, num_landmarks, 2)
    """
    encoding = detect_csv_encoding(csv_path)
    timestamps = []
    landmarks_list = []
    valid_mask = None

    with open(csv_path, 'r', encoding=encoding, errors='replace', newline='') as f:
        raw_lines = f.readlines()

    filtered_lines = []
    nul_lines = 0
    for idx, line in enumerate(raw_lines, start=1):
        if '\x00' in line:
            nul_lines += 1
            print(f"[WARN] Skipping line {idx}: contains NUL")
            continue
        filtered_lines.append(line)

    reader = csv.reader(filtered_lines)
    try:
        header = next(reader)
    except StopIteration:
        raise csv.Error(f"Empty CSV after removing {nul_lines} NUL-containing lines")

    # Find landmark column indices for MediaPipe Face Mesh format
    # Header format: timestamp, landmark_0_x, landmark_0_y, landmark_0_z, landmark_1_x, ...
    landmark_indices = []
    col_idx = 1  # Start after timestamp

    while col_idx < len(header):
        col_name = header[col_idx]
        # Check if this is an x coordinate (landmark_N_x)
        if '_x' in col_name and col_idx + 1 < len(header):
            next_col = header[col_idx + 1]
            # Verify next column is y coordinate
            if '_y' in next_col:
                landmark_indices.append((col_idx, col_idx + 1))
                # Skip x, y, z columns (assuming z exists)
                col_idx += 3
            else:
                col_idx += 1
        else:
            col_idx += 1

    num_landmarks = len(landmark_indices)

    if num_landmarks == 0:
        # Fallback: assume format is x, y, z triplets
        num_cols = len(header) - 1
        num_landmarks = num_cols // 3
        landmark_indices = [(1 + i * 3, 1 + i * 3 + 1) for i in range(num_landmarks)]

    valid_idx = None
    if 'valid_mask' in header:
        valid_idx = header.index('valid_mask')
        valid_mask = []

    line_num = 1  # header already consumed
    for row in reader:
        line_num += 1
        if not row:
            continue
        try:
            timestamp = float(row[0])
            # Parse 2D landmarks (x, y only)
            landmarks = []
            for x_idx, y_idx in landmark_indices:
                x = float(row[x_idx])
                y = float(row[y_idx])
                landmarks.append([x, y])
        except Exception as e:
            print(f"[WARN] Skipping line {line_num}: {e}")
            continue

        timestamps.append(timestamp)
        landmarks_list.append(landmarks)
        if valid_mask is not None and valid_idx is not None:
            try:
                valid_mask.append(int(float(row[valid_idx])))
            except Exception:
                valid_mask.append(0)

    landmarks_array = np.array(landmarks_list)
    return timestamps, landmarks_array, valid_mask


def get_3d_model_points():
    """
    Get 3D model points for MediaPipe Face Mesh landmarks.
    Coordinates are in mm, optimized for MediaPipe Face Mesh 468-point model.

    Returns:
        np.ndarray: 3D coordinates of facial landmarks
    """
    # 3D model points optimized for MediaPipe Face Mesh
    # Order: nose_tip(4), chin(152), left_eye_outer(33), right_eye_outer(263), mouth_left(61), mouth_right(291)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip (4)
        (0.0, -63.6, -12.5),         # Chin (152)
        (-43.3, 32.7, -26.0),        # Left eye left corner (33)
        (43.3, 32.7, -26.0),         # Right eye right corner (263)
        (-28.9, -28.9, -24.1),       # Left mouth corner (61)
        (28.9, -28.9, -24.1)         # Right mouth corner (291)
    ], dtype=np.float64)

    return model_points


def get_camera_matrix(_image_shape):
    """
    Return calibrated camera intrinsics and distortion coefficients.

    Args:
        _image_shape: Tuple of (height, width) (unused when calibration is provided)

    Returns:
        camera_matrix, dist_coeffs
    """
    camera_matrix = np.array(
        [
            [1061.014, 0.0, 477.172],
            [0.0, 1167.336, 944.795],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.array([[-0.446958, 0.281446, 0.004579, -0.055549]], dtype=np.float64)
    return camera_matrix, dist_coeffs


def estimate_head_pose(landmarks_2d: np.ndarray, image_shape: Tuple[int, int],
                       prev_rvec: Optional[np.ndarray] = None,
                       prev_tvec: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estimate head pose from 2D facial landmarks.

    Args:
        landmarks_2d: 2D facial landmarks (MediaPipe Face Mesh with 468 points)
        image_shape: Image shape (height, width)
        prev_rvec: Previous rotation vector for temporal consistency (optional)
        prev_tvec: Previous translation vector for temporal consistency (optional)

    Returns:
        Tuple of (rotation_vector, translation_vector, euler_angles)
        euler_angles: (pitch, yaw, roll) in degrees
    """
    # MediaPipe Face Mesh has 468 landmarks
    # We need at least 264 landmarks to access index 263
    if len(landmarks_2d) < 292:
        return None, None, None

    # Get 3D model points
    model_points = get_3d_model_points()

    # Get corresponding 2D image points (specific landmark indices)
    # Based on MediaPipe Face Mesh 468-point model:
    # nose_tip=4, chin=152, left_eye_left=33, right_eye_right=263, left_mouth=61, right_mouth=291
    landmark_indices = [4, 152, 33, 263, 61, 291]
    image_points = np.array([landmarks_2d[i] for i in landmark_indices], dtype=np.float64)

    # Check for NaN values
    if np.any(np.isnan(image_points)):
        return None, None, None

    # Camera internals
    camera_matrix, dist_coeffs = get_camera_matrix(image_shape)

    # Solve PnP with temporal consistency; use ITERATIVE when a good initial guess exists
    # If previous Z was excessive (>1500), re-init with EPNP to recover from bad landmarks.
    use_guess = prev_rvec is not None and prev_tvec is not None and float(prev_tvec[2]) <= 1500
    if use_guess:
        flags = cv2.SOLVEPNP_ITERATIVE
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            useExtrinsicGuess=True,
            rvec=prev_rvec,
            tvec=prev_tvec,
            flags=flags,
        )
    else:
        # Re-init (first frame or previous pose was implausible)
        flags = cv2.SOLVEPNP_EPNP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=flags,
        )

    if not success:
        return None, None, None

    # Convert rotation vector to Euler angles
    rotation_mat, _ = cv2.Rodrigues(rotation_vector)

    # Extract Euler angles (in radians)
    sy = np.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2])
        yaw = np.arctan2(-rotation_mat[2, 0], sy)
        roll = np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0])
    else:
        pitch = np.arctan2(-rotation_mat[1, 2], rotation_mat[1, 1])
        yaw = np.arctan2(-rotation_mat[2, 0], sy)
        roll = 0

    # Convert to degrees
    euler_angles = np.array([
        np.degrees(pitch),
        np.degrees(yaw),
        np.degrees(roll)
    ])

    return rotation_vector, translation_vector, euler_angles


def draw_head_pose_axes(frame: np.ndarray, landmarks_2d: np.ndarray,
                       rotation_vector: np.ndarray, translation_vector: np.ndarray,
                       image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Draw 3D head pose axes on a frame.

    Args:
        frame: Input frame to draw on
        landmarks_2d: 2D facial landmarks
        rotation_vector: Rotation vector from PnP
        translation_vector: Translation vector from PnP
        image_shape: Image shape (height, width)

    Returns:
        Frame with head pose axes drawn
    """
    vis_frame = frame.copy()

    # Get camera matrix
    camera_matrix, dist_coeffs = get_camera_matrix(image_shape)

    # Draw 3D coordinate axes
    # Define 3D axis points (in mm)
    axis_length = 50
    axis_points_3d = np.array([
        (0, 0, 0),              # Origin
        (axis_length, 0, 0),    # X-axis (Red)
        (0, axis_length, 0),    # Y-axis (Green)
        (0, 0, axis_length)     # Z-axis (Blue)
    ], dtype=np.float64)

    # Project 3D points to 2D
    axis_points_2d, _ = cv2.projectPoints(
        axis_points_3d,
        rotation_vector,
        translation_vector,
        camera_matrix,
        dist_coeffs
    )

    # Convert to integer coordinates
    origin = tuple(axis_points_2d[0].ravel().astype(int))
    x_axis = tuple(axis_points_2d[1].ravel().astype(int))
    y_axis = tuple(axis_points_2d[2].ravel().astype(int))
    z_axis = tuple(axis_points_2d[3].ravel().astype(int))

    # Draw a semi-transparent origin marker and a directional line showing the
    # forward (Z) direction so it's easier to identify the yaw origin.
    overlay = vis_frame.copy()

    # Origin marker (semi-transparent filled circle)
    origin_radius = max(8, int(min(image_shape) * 0.02))
    origin_color = (50, 50, 50)  # dark gray
    cv2.circle(overlay, origin, origin_radius, origin_color, -1)

    # Directional line from origin toward positive Z (projected z_axis)
    dir_color = (200, 200, 200)
    dir_thickness = max(4, int(origin_radius / 2))
    cv2.line(overlay, origin, z_axis, dir_color, dir_thickness)

    # Blend overlay with original frame for translucency
    alpha = 0.45
    cv2.addWeighted(overlay, alpha, vis_frame, 1 - alpha, 0, vis_frame)

    # Draw axes on top for clarity
    line_thickness = 3
    cv2.line(vis_frame, origin, x_axis, (0, 0, 255), line_thickness)  # X-axis: Red
    cv2.line(vis_frame, origin, y_axis, (0, 255, 0), line_thickness)  # Y-axis: Green
    cv2.line(vis_frame, origin, z_axis, (255, 0, 0), line_thickness)  # Z-axis: Blue

    # Draw origin border to make it pop
    cv2.circle(vis_frame, origin, origin_radius, (255, 255, 255), 1)

    # Add axis labels
    cv2.putText(vis_frame, 'X', x_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(vis_frame, 'Y', y_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis_frame, 'Z', z_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return vis_frame


def add_reference_axes_icon(frame: np.ndarray, size: int = 120, margin: int = 12) -> np.ndarray:
    """
    Draw a small reference axes icon (roll/pitch/yaw = 0) on the bottom-left corner.
    """
    h, w = frame.shape[:2]
    icon = frame.copy()

    # Icon background
    x0, y0 = margin, h - size - margin
    x1, y1 = x0 + size, y0 + size
    cv2.rectangle(icon, (x0, y0), (x1, y1), (20, 20, 20), -1)

    # Origin at bottom-left of icon
    origin = (x0 + int(size * 0.25), y1 - int(size * 0.25))
    axis_len = int(size * 0.35)

    # Draw reference axes (no rotation)
    cv2.arrowedLine(icon, origin, (origin[0] + axis_len, origin[1]), (0, 0, 255), 2, tipLength=0.15)   # X
    cv2.arrowedLine(icon, origin, (origin[0], origin[1] - axis_len), (0, 255, 0), 2, tipLength=0.15)   # Y
    cv2.arrowedLine(icon, origin, (origin[0] + int(axis_len * 0.7), origin[1] - int(axis_len * 0.6)), (255, 0, 0), 2, tipLength=0.15)  # Z

    cv2.putText(icon, "Ref (0,0,0)", (x0 + 6, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.addWeighted(icon, 0.75, frame, 0.25, 0, frame)
    return frame


def visualize_head_pose(frame: np.ndarray, landmarks_2d: np.ndarray,
                       rotation_vector: np.ndarray, translation_vector: np.ndarray,
                       euler_angles: np.ndarray, output_path: Path):
    """
    Visualize head pose with 3D coordinate axes and angle/position text.

    Args:
        frame: Input frame
        landmarks_2d: 2D facial landmarks
        rotation_vector: Rotation vector from PnP
        translation_vector: Translation vector from PnP
        euler_angles: Euler angles (pitch, yaw, roll) in degrees
        output_path: Path to save visualization
    """
    vis_frame = frame.copy()

    # Get camera matrix
    image_shape = frame.shape[:2]
    camera_matrix, dist_coeffs = get_camera_matrix(image_shape)

    # Draw 3D coordinate axes
    # Define 3D axis points (in mm)
    axis_length = 100
    axis_points_3d = np.array([
        (0, 0, 0),              # Origin
        (axis_length, 0, 0),    # X-axis (Red)
        (0, axis_length, 0),    # Y-axis (Green)
        (0, 0, axis_length)     # Z-axis (Blue)
    ], dtype=np.float64)

    # Project 3D points to 2D
    axis_points_2d, _ = cv2.projectPoints(
        axis_points_3d,
        rotation_vector,
        translation_vector,
        camera_matrix,
        dist_coeffs
    )

    # Convert to integer coordinates
    origin = tuple(axis_points_2d[0].ravel().astype(int))
    x_axis = tuple(axis_points_2d[1].ravel().astype(int))
    y_axis = tuple(axis_points_2d[2].ravel().astype(int))
    z_axis = tuple(axis_points_2d[3].ravel().astype(int))

    # Draw axes
    # Draw a semi-transparent origin marker and directional line showing the
    # forward (Z) direction so it's easier to identify the yaw origin.
    overlay = vis_frame.copy()

    # Origin marker (semi-transparent filled circle)
    origin_radius = max(8, int(min(image_shape) * 0.02))
    origin_color = (50, 50, 50)  # dark gray
    cv2.circle(overlay, origin, origin_radius, origin_color, -1)

    # Directional line from origin toward positive Z (projected z_axis)
    dir_color = (200, 200, 200)
    dir_thickness = max(4, int(origin_radius / 2))
    cv2.line(overlay, origin, z_axis, dir_color, dir_thickness)

    # Blend overlay with original frame for translucency
    alpha = 0.45
    cv2.addWeighted(overlay, alpha, vis_frame, 1 - alpha, 0, vis_frame)

    # Draw axes on top for clarity
    line_thickness = 3
    cv2.line(vis_frame, origin, x_axis, (0, 0, 255), line_thickness)  # X-axis: Red
    cv2.line(vis_frame, origin, y_axis, (0, 255, 0), line_thickness)  # Y-axis: Green
    cv2.line(vis_frame, origin, z_axis, (255, 0, 0), line_thickness)  # Z-axis: Blue

    # Draw origin border to make it pop
    cv2.circle(vis_frame, origin, origin_radius, (255, 255, 255), 1)

    # Add axis labels
    cv2.putText(vis_frame, 'X', x_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(vis_frame, 'Y', y_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis_frame, 'Z', z_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display angles and position as text
    pitch, yaw, roll = euler_angles
    tx, ty, tz = translation_vector.ravel()

    # Text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (255, 255, 255)  # White
    thickness = 2
    line_spacing = 30
    x_pos = 10
    y_pos = 30

    # Draw black background for text
    bg_color = (0, 0, 0)
    overlay = vis_frame.copy()
    cv2.rectangle(overlay, (0, 0), (400, 180), bg_color, -1)
    cv2.addWeighted(overlay, 0.6, vis_frame, 0.4, 0, vis_frame)

    # Draw text
    cv2.putText(vis_frame, f"Pitch: {pitch:.2f} deg", (x_pos, y_pos), font, font_scale, color, thickness)
    cv2.putText(vis_frame, f"Yaw:   {yaw:.2f} deg", (x_pos, y_pos + line_spacing), font, font_scale, color, thickness)
    cv2.putText(vis_frame, f"Roll:  {roll:.2f} deg", (x_pos, y_pos + line_spacing * 2), font, font_scale, color, thickness)
    cv2.putText(vis_frame, f"X: {tx:.2f}", (x_pos, y_pos + line_spacing * 3), font, font_scale, color, thickness)
    cv2.putText(vis_frame, f"Y: {ty:.2f}", (x_pos, y_pos + line_spacing * 4), font, font_scale, color, thickness)
    cv2.putText(vis_frame, f"Z: {tz:.2f}", (x_pos, y_pos + line_spacing * 5), font, font_scale, color, thickness)

    # Add reference axes icon (roll/pitch/yaw = 0) on bottom-left
    vis_frame = add_reference_axes_icon(vis_frame)

    # Save visualization
    cv2.imwrite(str(output_path), vis_frame)
    print(f"  [CHECK] Head pose visualization saved to: {output_path}")


def process_video_with_head_pose(video_path: Path, landmarks_array: np.ndarray,
                                 timestamps: List[float], csv_file,
                                 should_rotate: bool = False,
                                 check_mode: bool = False,
                                 check_output_path: Optional[Path] = None,
                                 video_out_path: Optional[Path] = None,
                                 valid_mask: Optional[List[int]] = None):
    """
    Process landmarks and estimate head pose for each frame.
    Video is only used for visualization when check_mode is enabled or video_out_path is provided.

    Args:
        video_path: Path to video file (used only for visualization)
        landmarks_array: Array of 2D landmarks (num_frames, num_landmarks, 2)
        timestamps: List of timestamps
        csv_file: Open CSV file to write results
        should_rotate: Whether to rotate frames
        check_mode: Whether to save first frame visualization
        check_output_path: Path to save check visualization
        video_out_path: Optional path to save visualization video (AVI format)
    """
    print(f"Processing landmarks from: {video_path.name}")
    if should_rotate:
        print(f"  [INFO] Frames will be rotated 90 degrees clockwise for visualization")

    # Get image shape from video for camera matrix calculation
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Could not open video file")
        return

    ret, first_frame = cap.read()
    if not ret:
        print(f"  [ERROR] Could not read first frame")
        cap.release()
        return

    if should_rotate:
        first_frame = cv2.rotate(first_frame, cv2.ROTATE_90_CLOCKWISE)

    image_shape = first_frame.shape[:2]

    # Initialize VideoWriter if video output is requested
    video_writer = None
    if video_out_path is not None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = image_shape[1]
        frame_height = image_shape[0]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (frame_width, frame_height))
        print(f"  [INFO] Creating output video: {video_out_path.name} ({frame_width}x{frame_height} @ {fps:.2f} fps)")
        # Reset capture for frame reading
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        cap.release()

    num_frames = len(landmarks_array)
    csv_writer = csv.writer(csv_file)

    # Inference time tracking
    inference_times = []

    # Batch processing time tracking
    batch_start_time = time.time()

    # Temporal consistency: track previous frame's pose
    prev_rvec = None
    prev_tvec = None

    # Process each frame's landmarks
    for frame_idx in range(num_frames):
        # Get landmarks for this frame
        landmarks_2d = landmarks_array[frame_idx]
        frame_valid = 1
        if valid_mask is not None:
            frame_valid = 1 if valid_mask[frame_idx] == 1 else 0

        # Read video frame if video output is requested
        current_frame = None
        if video_writer is not None:
            ret, current_frame = cap.read()
            if ret:
                if should_rotate:
                    current_frame = cv2.rotate(current_frame, cv2.ROTATE_90_CLOCKWISE)
            else:
                print(f"  [WARNING] Could not read frame {frame_idx}")

        # Estimate head pose with timing (using previous frame for stability)
        start_time = time.time()
        if frame_valid == 1:
            rotation_vec, translation_vec, euler_angles = estimate_head_pose(
                landmarks_2d, image_shape, prev_rvec, prev_tvec
            )
        else:
            rotation_vec = translation_vec = euler_angles = None
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        # Write to CSV
        row = [timestamps[frame_idx]]

        # Validate pose (filter out implausible depth)
        z_valid = translation_vec is not None and float(translation_vec[2]) <= 1500

        if rotation_vec is not None and translation_vec is not None and z_valid:
            # Add rotation vector
            row.extend(rotation_vec.ravel().tolist())
            # Add translation vector
            row.extend(translation_vec.ravel().tolist())
            # Add Euler angles
            row.extend(euler_angles.tolist())

            # Visualize first frame if check mode
            if check_mode and frame_idx == 0 and check_output_path is not None:
                visualize_head_pose(first_frame, landmarks_2d, rotation_vec, translation_vec, euler_angles, check_output_path)

            # Write visualization frame to video
            if video_writer is not None and current_frame is not None:
                vis_frame = draw_head_pose_axes(current_frame, landmarks_2d, rotation_vec, translation_vec, image_shape)

                # Add text overlays for angles/position with background box
                pitch, yaw, roll = euler_angles
                tx, ty, tz = translation_vec.ravel()

                overlay = vis_frame.copy()
                box_w, box_h = 260, 170
                cv2.rectangle(overlay, (5, 5), (5 + box_w, 5 + box_h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, vis_frame, 0.4, 0, vis_frame)

                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.7
                thick = 2
                line_gap = 28
                x0, y0 = 15, 35

                def draw_line(label, value, y, color_value):
                    cv2.putText(vis_frame, f"{label}: ", (x0, y), font, scale, (255, 255, 255), thick)
                    label_width = cv2.getTextSize(f"{label}: ", font, scale, thick)[0][0]
                    cv2.putText(vis_frame, value, (x0 + label_width, y), font, scale, color_value, thick)

                draw_line("Pitch", f"{pitch:.1f} deg", y0, (0, 0, 255))
                draw_line("Yaw", f"{yaw:.1f} deg", y0 + line_gap, (0, 0, 255))
                draw_line("Roll", f"{roll:.1f} deg", y0 + line_gap * 2, (0, 0, 255))
                draw_line("X", f"{tx:.0f} mm", y0 + line_gap * 3, (0, 255, 0))
                draw_line("Y", f"{ty:.0f} mm", y0 + line_gap * 4, (0, 255, 0))
                draw_line("Z", f"{tz:.0f} mm", y0 + line_gap * 5, (0, 255, 0))

                video_writer.write(vis_frame)

            # Update previous vectors for next frame (temporal consistency)
            prev_rvec = rotation_vec.copy()
            prev_tvec = translation_vec.copy()
            valid_out = 1
        else:
            # Write NaN for failed estimation
            row.extend([np.nan] * 9)  # 3 rotation + 3 translation + 3 euler angles
            valid_out = 0

            # Write blank frame to video if output is requested
            if video_writer is not None and current_frame is not None:
                video_writer.write(current_frame)

        row.append(valid_out)
        csv_writer.writerow(row)
        csv_file.flush()

        if (frame_idx + 1) % 100 == 0:
            batch_elapsed = time.time() - batch_start_time
            print(f"  Processed {frame_idx + 1}/{num_frames} frames ({batch_elapsed*1000:.2f}ms for 100 frames)")
            batch_start_time = time.time()

    print(f"  [SUCCESS] Processed all {num_frames} frames")

    # Release video resources
    if video_writer is not None:
        video_writer.release()
        cap.release()
        print(f"  [SUCCESS] Video output saved to: {video_out_path}")

    # Print inference time statistics
    if inference_times:
        avg_time = np.mean(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        print(f"  Inference time - Avg: {avg_time*1000:.2f}ms, Min: {min_time*1000:.2f}ms, Max: {max_time*1000:.2f}ms")
        print(f"  Average FPS: {fps:.2f}")

    print()


def process_single_file(face_kps_csv: Path, output_path: Path,
                       video_path: Optional[Path] = None,
                       should_rotate: bool = False,
                       check_mode: bool = False,
                       video_out_path: Optional[Path] = None) -> None:
    """
    Process a single face landmarks CSV file and estimate head pose.

    Args:
        face_kps_csv: Path to face_kps CSV file
        output_path: Output CSV file path
        video_path: Optional path to video file (for visualization and image size)
        should_rotate: Whether to rotate frames
        check_mode: Whether to save first frame visualization
        video_out_path: Optional path to save visualization video (AVI format)
    """
    print(f"Processing single file: {face_kps_csv.name}")
    print(f"Output: {output_path}")
    print(f"{'=' * 80}\n")

    # Read face landmarks
    print(f"Reading face landmarks from {face_kps_csv.name}...")
    timestamps, landmarks_array, valid_mask = read_face_landmarks_csv(face_kps_csv)
    if landmarks_array.ndim < 2 or landmarks_array.size == 0:
        print(f"Error: Landmarks array empty or invalid for {face_kps_csv}")
        return
    print(f"Loaded {len(timestamps)} frames with {landmarks_array.shape[1]} landmarks\n")

    # Get image shape from video or use default
    image_shape = (480, 640)  # Default shape
    first_frame = None
    cap = None
    video_writer = None

    if video_path is not None and video_path.exists():
        print(f"Reading video for image dimensions: {video_path.name}")
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            ret, first_frame = cap.read()
            if ret:
                if should_rotate:
                    first_frame = cv2.rotate(first_frame, cv2.ROTATE_90_CLOCKWISE)
                image_shape = first_frame.shape[:2]
                print(f"  Video dimensions: {image_shape[1]}x{image_shape[0]}")

                # Initialize VideoWriter if video output is requested
                if video_out_path is not None:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_width = image_shape[1]
                    frame_height = image_shape[0]
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (frame_width, frame_height))
                    print(f"  [INFO] Creating output video: {video_out_path.name} ({frame_width}x{frame_height} @ {fps:.2f} fps)")
                    # Reset capture for frame reading
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    cap.release()
                    cap = None
            else:
                cap.release()
                cap = None
        else:
            print(f"  [WARNING] Could not open video file, using default dimensions")
    else:
        print(f"  No video file provided, using default dimensions: {image_shape[1]}x{image_shape[0]}")

    # Prepare check output path
    check_output_path = None
    if check_mode:
        check_output_path = output_path.parent / f"check_headpose_{output_path.stem.replace('head_pose_', '')}.jpg"

    # Create CSV file and write header
    print(f"\nCreating output file: {output_path}")
    header = generate_csv_header()

    with open(output_path, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)
        csv_file.flush()
        print(f"  [SUCCESS] CSV file created with header ({len(header)} columns)\n")

        # Inference time tracking
        inference_times = []

        # Batch processing time tracking
        batch_start_time = time.time()

        # Temporal consistency: track previous frame's pose
        prev_rvec = None
        prev_tvec = None

        # Process each frame's landmarks
        num_frames = len(landmarks_array)
        for frame_idx in range(num_frames):
            # Get landmarks for this frame
            landmarks_2d = landmarks_array[frame_idx]

            # Read video frame if video output is requested
            current_frame = None
            if video_writer is not None and cap is not None:
                ret, current_frame = cap.read()
                if ret:
                    if should_rotate:
                        current_frame = cv2.rotate(current_frame, cv2.ROTATE_90_CLOCKWISE)
                else:
                    print(f"  [WARNING] Could not read frame {frame_idx}")

            # Estimate head pose with timing (using previous frame for stability)
            start_time = time.time()
            if valid_mask is not None and valid_mask[frame_idx] != 1:
                rotation_vec = translation_vec = euler_angles = None
            else:
                rotation_vec, translation_vec, euler_angles = estimate_head_pose(
                    landmarks_2d, image_shape, prev_rvec, prev_tvec
                )
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # Write to CSV
            row = [timestamps[frame_idx]]

            if rotation_vec is not None and translation_vec is not None:
                # Add rotation vector
                row.extend(rotation_vec.ravel().tolist())
                # Add translation vector
                row.extend(translation_vec.ravel().tolist())
                # Add Euler angles
                row.extend(euler_angles.tolist())

                # Visualize first frame if check mode
                if check_mode and frame_idx == 0 and check_output_path is not None and first_frame is not None:
                    visualize_head_pose(first_frame, landmarks_2d, rotation_vec, translation_vec, euler_angles, check_output_path)

                # Write visualization frame to video
                if video_writer is not None and current_frame is not None:
                    vis_frame = draw_head_pose_axes(current_frame, landmarks_2d, rotation_vec, translation_vec, image_shape)

                    # Add text overlays for angles and position
                    pitch, yaw, roll = euler_angles
                    tx, ty, tz = translation_vec.ravel()

                    cv2.putText(vis_frame, f"Pitch: {pitch:.1f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(vis_frame, f"Yaw: {yaw:.1f} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(vis_frame, f"Roll: {roll:.1f} deg", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(vis_frame, f"Pos: ({tx:.0f}, {ty:.0f}, {tz:.0f}) mm", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    video_writer.write(vis_frame)

                # Update previous vectors for next frame (temporal consistency)
                prev_rvec = rotation_vec.copy()
                prev_tvec = translation_vec.copy()
                valid_out = 1
            else:
                # Write NaN for failed estimation
                row.extend([np.nan] * 9)  # 3 rotation + 3 translation + 3 euler angles
                valid_out = 0

                # Write blank frame to video if output is requested
                if video_writer is not None and current_frame is not None:
                    video_writer.write(current_frame)

            row.append(valid_out)
            csv_writer.writerow(row)
            csv_file.flush()

            if (frame_idx + 1) % 100 == 0:
                batch_elapsed = time.time() - batch_start_time
                print(f"  Processed {frame_idx + 1}/{num_frames} frames ({batch_elapsed*1000:.2f}ms for 100 frames)")
                batch_start_time = time.time()

        print(f"  [SUCCESS] Processed all {num_frames} frames")

        # Release video resources
        if video_writer is not None:
            video_writer.release()
            if cap is not None:
                cap.release()
            print(f"  [SUCCESS] Video output saved to: {video_out_path}")

        # Print inference time statistics
        if inference_times:
            avg_time = np.mean(inference_times)
            min_time = np.min(inference_times)
            max_time = np.max(inference_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            print(f"  Inference time - Avg: {avg_time*1000:.2f}ms, Min: {min_time*1000:.2f}ms, Max: {max_time*1000:.2f}ms")
            print(f"  Average FPS: {fps:.2f}")

        print()

    print(f"[SUCCESS] Output saved to: {output_path}")


def generate_csv_header() -> List[str]:
    """
    Generate CSV header for head pose data.

    Returns:
        List of column names
    """
    header = [
        'timestamp',
        'head_rotation_x', 'head_rotation_y', 'head_rotation_z',
        'head_translation_x', 'head_translation_y', 'head_translation_z',
        'head_pitch', 'head_yaw', 'head_roll',
        'valid_mask'
    ]
    return header


def main():
    parser = argparse.ArgumentParser(
        description='Generate 3D head pose estimation from face landmarks and video files'
    )
    parser.add_argument(
        '--path',
        required=True,
        help='Video/image file path (single mode) or directory path (batch mode)'
    )
    parser.add_argument(
        '--no-batch',
        action='store_true',
        help='Single file mode: process a single video/image file'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Treat --path as a parent directory containing multiple case folders (batch only)'
    )
    parser.add_argument(
        '--face-landmark',
        type=str,
        default=None,
        help='Face landmarks CSV file path (single mode) or filename to use per case (batch/recursive)'
    )
    parser.add_argument(
        '--rotate',
        type=int,
        nargs='*',
        default=None,
        help='IDs to rotate 90 degrees clockwise (batch mode) or use --rotate for single file'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Save first frame with head pose visualization as JPG'
    )
    parser.add_argument(
        '--select',
        type=str,
        default=None,
        help='Comma-separated list of camera IDs to process (e.g., "0,2,4,6")'
    )
    parser.add_argument(
        '--video-out',
        action='store_true',
        help='Save visualization as video file (AVI format with same fps and resolution as input)'
    )

    args = parser.parse_args()

    print(f"\n{'=' * 80}")
    print(f"Head Pose Generator")
    print(f"Mode: {'Single File' if args.no_batch else 'Batch'}")

    # Single file mode
    if args.no_batch:
        file_path = Path(args.path)
        if not file_path.exists():
            print(f"Error: File does not exist: {args.path}")
            return 1

        if not file_path.is_file():
            print(f"Error: Path is not a file: {args.path}")
            return 1

        # Get face landmarks CSV file path (required)
        if not args.face_landmark:
            print(f"Error: --face-landmark option is required for single file mode")
            return 1

        face_kps_csv = Path(args.face_landmark)
        if not face_kps_csv.exists():
            print(f"Error: Face landmarks CSV file does not exist: {args.face_landmark}")
            return 1

        # Determine output path
        output_dir = file_path.parent
        output_filename = f"head_pose_{file_path.stem}.csv"
        output_path = output_dir / output_filename

        # Check if rotation is requested (--rotate flag present means rotate)
        should_rotate = args.rotate is not None

        # Determine video output path if requested
        video_out_path = None
        if args.video_out:
            file_id = extract_id_from_filename(file_path.name)
            if file_id is not None:
                video_out_path = output_dir / f"headpose_{file_id}_visualization.avi"
            else:
                video_out_path = output_dir / f"headpose_{file_path.stem}_visualization.avi"

        try:
            process_single_file(
                face_kps_csv=face_kps_csv,
                output_path=output_path,
                video_path=file_path,
                should_rotate=should_rotate,
                check_mode=args.check,
                video_out_path=video_out_path
            )
        except Exception as e:
            print(f"\nError processing file: {e}")
            import traceback
            traceback.print_exc()
            return 1

        return 0

    def process_batch(base_directory: Path) -> int:
        if not base_directory.exists():
            print(f"Error: Directory does not exist: {base_directory}")
            return 1
        if not base_directory.is_dir():
            print(f"Error: Path is not a directory: {base_directory}")
            return 1

        # Output files will be saved in base directory
        output_directory = base_directory

        # camera directory must exist
        input_directory = base_directory / "camera"
        if not input_directory.exists():
            print(f"Error: Camera directory does not exist: {input_directory}")
            return 1
        if not input_directory.is_dir():
            print(f"Error: Camera path is not a directory: {input_directory}")
            return 1

        print(f"Input Directory: {input_directory}")
        print(f"Output Directory: {output_directory}")
        print(f"Rotate IDs: {args.rotate if args.rotate is not None else 'None'}")
        print(f"Check mode: {args.check}")
        print(f"Video output: {args.video_out}\n")

        # Parse selected IDs if provided
        selected_ids = None
        if args.select:
            try:
                selected_ids = [int(x.strip()) for x in args.select.split(',')]
                print(f"Selected IDs: {selected_ids}\n")
            except ValueError:
                print(f"Error: Invalid format for --select. Expected comma-separated integers (e.g., '0,2,4,6')")
                return 1

        def resolve_face_landmark_path(case_dir: Path, face_landmark_arg: str) -> Path:
            candidate = Path(face_landmark_arg)
            if candidate.is_absolute():
                return candidate
            if candidate.parent != Path('.'):
                return case_dir / candidate
            return case_dir / face_landmark_arg

        def find_avi_for_id(camera_dir: Path, file_id: int) -> Optional[Path]:
            for avi_file in camera_dir.glob('*.avi'):
                if avi_file.name.startswith('._'):
                    continue
                if extract_id_from_filename(avi_file.name) == file_id:
                    return avi_file
            return None

        # Find file pairs
        if args.face_landmark:
            face_kps_csv = resolve_face_landmark_path(base_directory, args.face_landmark)
            if not face_kps_csv.exists():
                print(f"Error: Face landmarks CSV not found: {face_kps_csv}")
                return 1
            file_id = extract_id_from_filename(face_kps_csv.name)
            if file_id is None:
                print(f"Error: Could not extract ID from face landmarks filename: {face_kps_csv.name}")
                return 1
            avi_path = find_avi_for_id(input_directory, file_id)
            if avi_path is None:
                print(f"Error: No matching AVI found for ID {file_id} in {input_directory}")
                return 1
            pairs = {file_id: {'avi': avi_path, 'csv': face_kps_csv}}
        else:
            pairs = find_file_pairs(base_directory, selected_ids=selected_ids)

        if not pairs:
            print("No file pairs found. Exiting.")
            return 1

        # Process each pair
        print(f"{'=' * 80}")
        print(f"Processing video files")
        print(f"{'=' * 80}\n")

        for file_id, files in pairs.items():
            print(f"{'=' * 80}")
            print(f"Processing ID: {file_id}")
            print(f"{'=' * 80}\n")

            # Read face landmarks
            print(f"Reading face landmarks from {files['csv'].name}...")
            timestamps, landmarks_array, valid_mask = read_face_landmarks_csv(files['csv'])
            if landmarks_array.ndim < 2 or landmarks_array.size == 0:
                print(f"[ERROR] Landmarks array empty or invalid for {files['csv'].name}. Skipping.")
                continue
            print(f"Loaded {len(timestamps)} frames with {landmarks_array.shape[1]} landmarks\n")

            # Check if this ID should be rotated
            should_rotate = args.rotate is not None and file_id in args.rotate
            if should_rotate:
                print(f"  [INFO] ID {file_id} will be rotated 90 degrees clockwise\n")

            # Prepare check visualization path
            check_output_path = None
            if args.check:
                check_output_path = output_directory / f"check_headpose_{file_id}.jpg"

            # Prepare video output path if requested
            video_out_path = None
            if args.video_out:
                video_out_path = output_directory / f"headpose_{file_id}_visualization.avi"

            # Generate output filename
            output_path = output_directory / f"head_pose_{file_id}.csv"

            # Create CSV file and write header
            print(f"Creating output file: {output_path}")
            header = generate_csv_header()
            with open(output_path, 'w', newline='', encoding='utf-8') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(header)
                csv_file.flush()
                print(f"  [SUCCESS] CSV file created with header ({len(header)} columns)\n")

                # Process video and estimate head pose
                process_video_with_head_pose(
                    files['avi'], landmarks_array, timestamps, csv_file,
                    should_rotate=should_rotate,
                    check_mode=args.check,
                    check_output_path=check_output_path,
                    video_out_path=video_out_path,
                    valid_mask=valid_mask
                )

            print(f"  [SUCCESS] Output saved to: {output_path}\n")

        print(f"{'=' * 80}")
        print(f"[SUCCESS] All files processed")
        print(f"Total pairs processed: {len(pairs)}")
        print(f"{'=' * 80}")

        return 0

    # Batch mode
    if args.recursive:
        base_directory = Path(args.path)
        if not base_directory.exists() or not base_directory.is_dir():
            print(f"Error: Directory does not exist or is not a directory: {args.path}")
            return 1
        case_dirs = sorted({p.parent for p in base_directory.rglob('camera') if p.is_dir()})
        if not case_dirs:
            print(f"No case directories with a camera folder found under: {base_directory}")
            return 1
        print(f"\n[INFO] Recursive mode: {len(case_dirs)} case(s) found")
        for idx, case_dir in enumerate(case_dirs, start=1):
            print(f"  [{idx}] {case_dir}")
    else:
        case_dirs = [Path(args.path)]

    had_error = False
    for case_dir in case_dirs:
        if args.recursive:
            print(f"\n{'=' * 80}")
            print(f"[CASE] {case_dir}")
            print(f"{'=' * 80}")
        if process_batch(case_dir) != 0:
            had_error = True

    return 0 if not had_error else 1

    return 0


if __name__ == '__main__':
    exit(main())
