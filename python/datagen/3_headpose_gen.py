#!/usr/bin/env python3
"""
Head Pose Generator
Processes face landmarks CSV and generates 3D head pose estimation (position and orientation).
Uses PnP algorithm to estimate head pose from 2D facial landmarks.
"""

import argparse
import csv
import re
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


def find_file_pairs(directory: Path) -> Dict[int, Dict[str, Path]]:
    """
    Find and pair AVI and face landmarks CSV files by their ID suffix.

    Args:
        directory: Directory to search for files

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


def read_face_landmarks_csv(csv_path: Path) -> Tuple[List[float], np.ndarray]:
    """
    Read face landmarks from CSV file.
    Extracts only 2D landmarks (ignores 3D if present).

    Args:
        csv_path: Path to face_kps CSV file

    Returns:
        Tuple of (timestamps, landmarks_array)
        landmarks_array shape: (num_frames, num_landmarks, 2)
    """
    encoding = detect_csv_encoding(csv_path)
    timestamps = []
    landmarks_list = []

    with open(csv_path, 'r', encoding=encoding) as f:
        reader = csv.reader(f)
        header = next(reader)

        # Find 2D landmark column indices
        # Header format: timestamp, landmark_0_x_2d, landmark_0_y_2d, ..., [landmark_0_x_3d, ...]
        landmark_2d_indices = []
        col_idx = 1  # Start after timestamp

        while col_idx < len(header):
            col_name = header[col_idx]
            # Check if this is a 2D x coordinate
            if '_x_2d' in col_name or (col_idx == 1 and '_x' in col_name and '_3d' not in col_name):
                # Found x_2d, next should be y_2d
                landmark_2d_indices.append((col_idx, col_idx + 1))
                col_idx += 2
            elif '_x_3d' in col_name:
                # Reached 3D section, stop
                break
            else:
                col_idx += 1

        num_landmarks = len(landmark_2d_indices)

        if num_landmarks == 0:
            # Fallback: assume all columns after timestamp are 2D landmarks (x, y pairs)
            num_cols = len(header) - 1
            num_landmarks = num_cols // 2
            landmark_2d_indices = [(1 + i * 2, 1 + i * 2 + 1) for i in range(num_landmarks)]

        for row in reader:
            if not row:
                continue

            timestamp = float(row[0])
            timestamps.append(timestamp)

            # Parse 2D landmarks only
            landmarks = []
            for x_idx, y_idx in landmark_2d_indices:
                x = float(row[x_idx])
                y = float(row[y_idx])
                landmarks.append([x, y])

            landmarks_list.append(landmarks)

    landmarks_array = np.array(landmarks_list)
    return timestamps, landmarks_array


def get_3d_model_points():
    """
    Get 3D model points for standard facial landmarks.
    Using a generic 3D face model.

    Returns:
        np.ndarray: 3D coordinates of facial landmarks
    """
    # 3D model points (generic face model)
    # Indices based on 68-point model
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip (30)
        (0.0, -330.0, -65.0),        # Chin (8)
        (-225.0, 170.0, -135.0),     # Left eye left corner (36)
        (225.0, 170.0, -135.0),      # Right eye right corner (45)
        (-150.0, -150.0, -125.0),    # Left Mouth corner (48)
        (150.0, -150.0, -125.0)      # Right mouth corner (54)
    ], dtype=np.float64)

    return model_points


def get_camera_matrix(image_shape):
    """
    Estimate camera matrix from image size.

    Args:
        image_shape: Tuple of (height, width)

    Returns:
        Camera matrix
    """
    size = image_shape
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype=np.float64
    )
    return camera_matrix


def estimate_head_pose(landmarks_2d: np.ndarray, image_shape: Tuple[int, int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estimate head pose from 2D facial landmarks.

    Args:
        landmarks_2d: 2D facial landmarks (68 points expected)
        image_shape: Image shape (height, width)

    Returns:
        Tuple of (rotation_vector, translation_vector, euler_angles)
        euler_angles: (pitch, yaw, roll) in degrees
    """
    if len(landmarks_2d) < 68:
        return None, None, None

    # Get 3D model points
    model_points = get_3d_model_points()

    # Get corresponding 2D image points (specific landmark indices)
    # Based on 68-point model: nose_tip=30, chin=8, left_eye_left=36, right_eye_right=45, left_mouth=48, right_mouth=54
    landmark_indices = [30, 8, 36, 45, 48, 54]
    image_points = np.array([landmarks_2d[i] for i in landmark_indices], dtype=np.float64)

    # Check for NaN values
    if np.any(np.isnan(image_points)):
        return None, None, None

    # Camera internals
    camera_matrix = get_camera_matrix(image_shape)
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
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
    camera_matrix = get_camera_matrix(frame.shape[:2])
    dist_coeffs = np.zeros((4, 1))

    # Draw 3D coordinate axes
    # Define 3D axis points (in mm)
    axis_length = 300
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
    line_thickness = 3
    cv2.line(vis_frame, origin, x_axis, (0, 0, 255), line_thickness)  # X-axis: Red
    cv2.line(vis_frame, origin, y_axis, (0, 255, 0), line_thickness)  # Y-axis: Green
    cv2.line(vis_frame, origin, z_axis, (255, 0, 0), line_thickness)  # Z-axis: Blue

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

    # Save visualization
    cv2.imwrite(str(output_path), vis_frame)
    print(f"  [CHECK] Head pose visualization saved to: {output_path}")


def process_video_with_head_pose(video_path: Path, landmarks_array: np.ndarray,
                                 timestamps: List[float], csv_file,
                                 should_rotate: bool = False,
                                 check_mode: bool = False,
                                 check_output_path: Optional[Path] = None):
    """
    Process landmarks and estimate head pose for each frame.
    Video is only used for visualization when check_mode is enabled.

    Args:
        video_path: Path to video file (used only for visualization)
        landmarks_array: Array of 2D landmarks (num_frames, num_landmarks, 2)
        timestamps: List of timestamps
        csv_file: Open CSV file to write results
        should_rotate: Whether to rotate frames
        check_mode: Whether to save first frame visualization
        check_output_path: Path to save check visualization
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
    cap.release()

    num_frames = len(landmarks_array)
    csv_writer = csv.writer(csv_file)

    # Process each frame's landmarks
    for frame_idx in range(num_frames):
        # Get landmarks for this frame
        landmarks_2d = landmarks_array[frame_idx]

        # Estimate head pose
        rotation_vec, translation_vec, euler_angles = estimate_head_pose(landmarks_2d, image_shape)

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
            if check_mode and frame_idx == 0 and check_output_path is not None:
                visualize_head_pose(first_frame, landmarks_2d, rotation_vec, translation_vec, euler_angles, check_output_path)
        else:
            # Write NaN for failed estimation
            row.extend([np.nan] * 9)  # 3 rotation + 3 translation + 3 euler angles

        csv_writer.writerow(row)
        csv_file.flush()

        if (frame_idx + 1) % 100 == 0:
            print(f"  Processed {frame_idx + 1}/{num_frames} frames")

    print(f"  [SUCCESS] Processed all {num_frames} frames\n")


def generate_csv_header() -> List[str]:
    """
    Generate CSV header for head pose data.

    Returns:
        List of column names
    """
    header = [
        'timestamp',
        'rotation_x', 'rotation_y', 'rotation_z',
        'translation_x', 'translation_y', 'translation_z',
        'pitch', 'yaw', 'roll'
    ]
    return header


def main():
    parser = argparse.ArgumentParser(
        description='Generate 3D head pose estimation from face landmarks and video files'
    )
    parser.add_argument(
        '--path',
        required=True,
        help='Directory containing camera subdirectory and face_kps CSV files'
    )
    parser.add_argument(
        '--rotate',
        type=int,
        nargs='+',
        default=[],
        help='IDs to rotate 90 degrees clockwise (e.g., --rotate 0 1 2)'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Save first frame with head pose visualization as JPG'
    )

    args = parser.parse_args()

    # Validate path
    base_directory = Path(args.path)
    if not base_directory.exists():
        print(f"Error: Directory does not exist: {args.path}")
        return 1

    if not base_directory.is_dir():
        print(f"Error: Path is not a directory: {args.path}")
        return 1

    # Output files will be saved in base directory
    output_directory = base_directory

    print(f"\n{'=' * 80}")
    print(f"Head Pose Generator")
    print(f"Base Directory: {base_directory}")
    print(f"Output Directory: {output_directory}")
    print(f"Rotate IDs: {args.rotate if args.rotate else 'None'}")
    print(f"Check mode: {args.check}")
    print(f"{'=' * 80}\n")

    # Find file pairs
    pairs = find_file_pairs(base_directory)

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
        timestamps, landmarks_array = read_face_landmarks_csv(files['csv'])
        print(f"Loaded {len(timestamps)} frames with {landmarks_array.shape[1]} landmarks\n")

        # Check if this ID should be rotated
        should_rotate = file_id in args.rotate
        if should_rotate:
            print(f"  [INFO] ID {file_id} will be rotated 90 degrees clockwise\n")

        # Prepare check visualization path
        check_output_path = None
        if args.check:
            check_output_path = output_directory / f"check_headpose_{file_id}.jpg"

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
                check_output_path=check_output_path
            )

        print(f"  [SUCCESS] Output saved to: {output_path}\n")

    print(f"{'=' * 80}")
    print(f"[SUCCESS] All files processed")
    print(f"Total pairs processed: {len(pairs)}")
    print(f"{'=' * 80}")

    return 0


if __name__ == '__main__':
    exit(main())
