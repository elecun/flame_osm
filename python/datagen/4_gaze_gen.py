#!/usr/bin/env python3
"""
Gaze Estimation Generator
Processes video files and generates gaze estimation data using MPIIGaze-based models.
Supports three model modes: mpiigaze, mpiifacegaze, eth-xgaze.
Estimates eye gaze direction (pitch and yaw angles).
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
try:
    from ptgaze import GazeEstimator
    PTGAZE_AVAILABLE = True
except ImportError:
    PTGAZE_AVAILABLE = False
    print("Warning: ptgaze library not available. Will use simplified gaze estimation.")
    print("Install with: pip install ptgaze")


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
        filename: Filename to extract ID from

    Returns:
        Numeric ID or None if not found
    """
    match = re.search(r'_(\d+)\.[^.]+$', filename)
    if match:
        return int(match.group(1))
    return None


def find_file_pairs(directory: Path) -> Dict[int, Dict[str, Path]]:
    """
    Find and pair AVI and timestamp CSV files by their ID suffix.

    Args:
        directory: Directory to search for files

    Returns:
        Dictionary mapping ID to {'avi': Path, 'csv': Path}
    """
    print(f"{'=' * 80}")
    print(f"Searching for file pairs in: {directory}")
    print(f"{'=' * 80}\n")

    # Find all AVI files
    print("Searching for AVI files...")
    avi_files = {}
    all_avi = list(directory.glob('*.avi'))
    print(f"Found {len(all_avi)} AVI file(s)\n")

    for avi_file in all_avi:
        if avi_file.name.startswith('._'):
            continue
        file_id = extract_id_from_filename(avi_file.name)
        if file_id is not None:
            avi_files[file_id] = avi_file
            print(f"  Found AVI file: {avi_file.name} (ID: {file_id})")

    # Find all CSV files
    print(f"\nSearching for CSV files...")
    csv_files = {}
    all_csv = list(directory.glob('*.csv'))
    print(f"Found {len(all_csv)} CSV file(s)\n")

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
            print(f"  [WARNING] ID {file_id}: AVI file has no matching CSV")

    for file_id in csv_files.keys():
        if file_id not in avi_files:
            print(f"  [WARNING] ID {file_id}: CSV file has no matching AVI")

    print(f"\n{'=' * 80}")
    print(f"Total pairs found: {len(pairs)}")
    print(f"{'=' * 80}\n")

    return pairs


def read_timestamps(csv_path: Path) -> List[float]:
    """
    Read timestamps from CSV file.

    Args:
        csv_path: Path to timestamp CSV file

    Returns:
        List of timestamp values
    """
    encoding = detect_csv_encoding(csv_path)
    timestamps = []

    with open(csv_path, 'r', encoding=encoding) as f:
        reader = csv.reader(f)
        rows = list(reader)

        if not rows:
            return timestamps

        # Check if first row is a header
        first_value = rows[0][0].strip()
        has_header = False

        try:
            float(first_value)
            has_header = False
        except ValueError:
            has_header = True

        # Read timestamps
        start_idx = 1 if has_header else 0
        for row in rows[start_idx:]:
            if row:
                timestamp = float(row[0])
                timestamps.append(timestamp)

    return timestamps


def detect_face_and_eyes(frame: np.ndarray) -> Tuple[Optional[Tuple], Optional[Tuple], Optional[Tuple]]:
    """
    Detect face and eyes using Haar cascades.

    Args:
        frame: Input frame

    Returns:
        Tuple of (face_rect, left_eye_rect, right_eye_rect)
    """
    # Load Haar cascade classifiers
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None, None, None

    # Get largest face
    face = max(faces, key=lambda rect: rect[2] * rect[3])
    fx, fy, fw, fh = face

    # Detect eyes in face region
    face_roi = gray[fy:fy+fh, fx:fx+fw]
    eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 5)

    if len(eyes) < 2:
        return face, None, None

    # Sort eyes by x coordinate (left to right)
    eyes = sorted(eyes, key=lambda e: e[0])

    # Get left and right eyes (relative to face region)
    left_eye = (fx + eyes[0][0], fy + eyes[0][1], eyes[0][2], eyes[0][3])
    right_eye = (fx + eyes[1][0], fy + eyes[1][1], eyes[1][2], eyes[1][3])

    return face, left_eye, right_eye


def estimate_gaze_direction_simple(frame: np.ndarray, left_eye: Tuple, right_eye: Tuple) -> Tuple[float, float]:
    """
    Estimate gaze direction from eye regions using simplified method.
    This is a fallback method when ptgaze is not available.

    Args:
        frame: Input frame
        left_eye: Left eye rectangle (x, y, w, h)
        right_eye: Right eye rectangle (x, y, w, h)

    Returns:
        Tuple of (pitch, yaw) in degrees
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Extract eye regions
    lex, ley, lew, leh = left_eye
    rex, rey, rew, reh = right_eye

    left_eye_roi = gray[ley:ley+leh, lex:lex+lew]
    right_eye_roi = gray[rey:rey+reh, rex:rex+rew]

    # Find eye centers using moments (simplified method)
    def find_eye_center(eye_roi):
        # Apply threshold to find dark regions (pupil/iris)
        _, threshold = cv2.threshold(eye_roi, 70, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            # Return center of ROI if no contours found
            return eye_roi.shape[1] / 2, eye_roi.shape[0] / 2

        # Get largest contour (likely the pupil/iris)
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate moments
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return eye_roi.shape[1] / 2, eye_roi.shape[0] / 2

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        return cx, cy

    # Get eye centers
    left_center_x, left_center_y = find_eye_center(left_eye_roi)
    right_center_x, right_center_y = find_eye_center(right_eye_roi)

    # Calculate gaze direction relative to eye region
    # Normalized to [-1, 1] range
    left_gaze_x = (left_center_x / lew - 0.5) * 2
    left_gaze_y = (left_center_y / leh - 0.5) * 2

    right_gaze_x = (right_center_x / rew - 0.5) * 2
    right_gaze_y = (right_center_y / reh - 0.5) * 2

    # Average gaze from both eyes
    avg_gaze_x = (left_gaze_x + right_gaze_x) / 2
    avg_gaze_y = (left_gaze_y + right_gaze_y) / 2

    # Convert to pitch and yaw angles (in degrees)
    # This is a simplified approximation
    yaw = avg_gaze_x * 30  # +/- 30 degrees horizontal
    pitch = avg_gaze_y * 20  # +/- 20 degrees vertical

    return pitch, yaw


def estimate_gaze_direction(frame: np.ndarray, gaze_estimator, model_type: str,
                           face: Optional[Tuple] = None,
                           left_eye: Optional[Tuple] = None,
                           right_eye: Optional[Tuple] = None) -> Tuple[float, float]:
    """
    Estimate gaze direction using the selected model.

    Args:
        frame: Input frame
        gaze_estimator: GazeEstimator instance (or None for simplified method)
        model_type: Model type ('mpiigaze', 'mpiifacegaze', 'eth-xgaze', or 'simple')
        face: Face rectangle (x, y, w, h) - used for simplified method
        left_eye: Left eye rectangle - used for simplified method
        right_eye: Right eye rectangle - used for simplified method

    Returns:
        Tuple of (pitch, yaw) in degrees
    """
    if gaze_estimator is None or model_type == 'simple':
        # Use simplified method
        if left_eye is None or right_eye is None:
            return np.nan, np.nan
        return estimate_gaze_direction_simple(frame, left_eye, right_eye)

    # Use ptgaze model
    result = gaze_estimator.predict(frame)

    if result is None or len(result) == 0:
        return np.nan, np.nan

    # Get first detected face
    face_result = result[0]

    # Extract gaze vector
    gaze_vector = face_result.get('gaze_vector', None)
    if gaze_vector is None:
        return np.nan, np.nan

    # Convert gaze vector to pitch and yaw
    # Gaze vector is normalized 3D vector (x, y, z)
    x, y, z = gaze_vector

    # Calculate pitch and yaw from gaze vector
    # Pitch: angle in vertical plane (up/down)
    # Yaw: angle in horizontal plane (left/right)
    pitch = np.degrees(np.arcsin(-y))
    yaw = np.degrees(np.arctan2(x, -z))

    return pitch, yaw


def visualize_gaze(frame: np.ndarray, face: Tuple, left_eye: Tuple, right_eye: Tuple,
                  pitch: float, yaw: float, output_path: Path):
    """
    Visualize gaze direction on frame.

    Args:
        frame: Input frame
        face: Face rectangle
        left_eye: Left eye rectangle
        right_eye: Right eye rectangle
        pitch: Pitch angle in degrees
        yaw: Yaw angle in degrees
        output_path: Path to save visualization
    """
    vis_frame = frame.copy()

    # Draw face rectangle
    fx, fy, fw, fh = face
    cv2.rectangle(vis_frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)

    # Draw eye rectangles
    if left_eye is not None:
        lex, ley, lew, leh = left_eye
        cv2.rectangle(vis_frame, (lex, ley), (lex+lew, ley+leh), (255, 0, 0), 2)

    if right_eye is not None:
        rex, rey, rew, reh = right_eye
        cv2.rectangle(vis_frame, (rex, rey), (rex+rew, rey+reh), (255, 0, 0), 2)

    # Calculate center point between eyes for gaze arrow
    if left_eye is not None and right_eye is not None:
        center_x = (left_eye[0] + left_eye[2]//2 + right_eye[0] + right_eye[2]//2) // 2
        center_y = (left_eye[1] + left_eye[3]//2 + right_eye[1] + right_eye[3]//2) // 2

        # Draw gaze direction arrow
        arrow_length = 100
        # Convert pitch/yaw to arrow endpoint
        end_x = int(center_x + arrow_length * np.sin(np.radians(yaw)))
        end_y = int(center_y + arrow_length * np.sin(np.radians(pitch)))

        cv2.arrowedLine(vis_frame, (center_x, center_y), (end_x, end_y), (0, 0, 255), 3, tipLength=0.3)

    # Add text with angles
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (255, 255, 255)
    thickness = 2

    # Draw background for text
    overlay = vis_frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, vis_frame, 0.4, 0, vis_frame)

    # Draw text
    cv2.putText(vis_frame, f"Pitch: {pitch:.2f} deg", (10, 30), font, font_scale, color, thickness)
    cv2.putText(vis_frame, f"Yaw:   {yaw:.2f} deg", (10, 60), font, font_scale, color, thickness)

    # Save visualization
    cv2.imwrite(str(output_path), vis_frame)
    print(f"  [CHECK] Gaze visualization saved to: {output_path}")


def process_video_with_gaze(video_path: Path, timestamps: List[float], csv_file,
                            gaze_estimator,
                            model_type: str = 'simple',
                            should_rotate: bool = False,
                            check_mode: bool = False,
                            check_output_path: Optional[Path] = None):
    """
    Process video and estimate gaze for each frame.

    Args:
        video_path: Path to video file
        timestamps: List of timestamps
        csv_file: Open CSV file to write results
        gaze_estimator: GazeEstimator instance (or None for simplified method)
        model_type: Model type ('mpiigaze', 'mpiifacegaze', 'eth-xgaze', or 'simple')
        should_rotate: Whether to rotate frames
        check_mode: Whether to save first frame visualization
        check_output_path: Path to save check visualization
    """
    print(f"Processing video: {video_path.name}")
    print(f"  [INFO] Using model: {model_type}")
    if should_rotate:
        print(f"  [INFO] Video will be rotated 90 degrees clockwise")

    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    csv_writer = csv.writer(csv_file)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if should_rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # For simplified method, detect face and eyes first
        face, left_eye, right_eye = None, None, None
        if model_type == 'simple':
            face, left_eye, right_eye = detect_face_and_eyes(frame)

        # Estimate gaze direction
        pitch, yaw = estimate_gaze_direction(
            frame, gaze_estimator, model_type,
            face=face, left_eye=left_eye, right_eye=right_eye
        )

        # Write to CSV
        row = [timestamps[frame_idx]]
        row.extend([pitch, yaw])

        csv_writer.writerow(row)
        csv_file.flush()

        # Visualize first frame if check mode
        if check_mode and frame_idx == 0 and check_output_path is not None:
            # For visualization, we need face and eyes
            if face is None or left_eye is None or right_eye is None:
                face, left_eye, right_eye = detect_face_and_eyes(frame)

            if face is not None and left_eye is not None and right_eye is not None:
                visualize_gaze(frame, face, left_eye, right_eye, pitch, yaw, check_output_path)

        if (frame_idx + 1) % 100 == 0:
            print(f"  Processed {frame_idx + 1}/{frame_count} frames")

        frame_idx += 1

    cap.release()
    print(f"  [SUCCESS] Processed all {frame_count} frames\n")


def generate_csv_header() -> List[str]:
    """
    Generate CSV header for gaze data.

    Returns:
        List of column names
    """
    return ['timestamp', 'pitch', 'yaw']


def main():
    parser = argparse.ArgumentParser(
        description='Generate gaze estimation data from video files using MPIIGaze models'
    )
    parser.add_argument(
        '--path',
        required=True,
        help='Directory containing camera subdirectory with AVI and timestamp CSV files'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['mpiigaze', 'mpiifacegaze', 'eth-xgaze', 'simple'],
        default='simple',
        help='Gaze estimation model to use (default: simple)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default='cuda',
        help='Device to use for gaze estimation (default: cuda)'
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
        help='Save first frame with gaze visualization as JPG'
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

    # Input files are in 'camera' subdirectory
    input_directory = base_directory / "camera"
    if not input_directory.exists():
        print(f"Error: Camera directory does not exist: {input_directory}")
        return 1

    if not input_directory.is_dir():
        print(f"Error: Camera path is not a directory: {input_directory}")
        return 1

    # Output files will be saved in base directory
    output_directory = base_directory

    print(f"\n{'=' * 80}")
    print(f"Gaze Estimation Generator")
    print(f"Input Directory: {input_directory}")
    print(f"Output Directory: {output_directory}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Rotate IDs: {args.rotate if args.rotate else 'None'}")
    print(f"Check mode: {args.check}")
    print(f"{'=' * 80}\n")

    # Initialize gaze estimator
    gaze_estimator = None
    if args.model != 'simple':
        if not PTGAZE_AVAILABLE:
            print("Error: ptgaze library is required for advanced models (mpiigaze, mpiifacegaze, eth-xgaze)")
            print("Install with: pip install ptgaze")
            print("Falling back to simplified method...\n")
            args.model = 'simple'
        else:
            print(f"Initializing {args.model} model on {args.device}...")
            try:
                # Map model names to ptgaze model names
                model_map = {
                    'mpiigaze': 'MPIIGaze',
                    'mpiifacegaze': 'MPIIFaceGaze',
                    'eth-xgaze': 'ETH-XGaze'
                }
                gaze_estimator = GazeEstimator(model=model_map[args.model], device=args.device)
                print(f"  [SUCCESS] {args.model} model initialized on {args.device}\n")
            except Exception as e:
                print(f"  [ERROR] Failed to initialize {args.model}: {e}")
                print("  Falling back to simplified method...\n")
                args.model = 'simple'
                gaze_estimator = None

    # Find file pairs in camera directory
    pairs = find_file_pairs(input_directory)

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

        # Read timestamps
        timestamps = read_timestamps(files['csv'])
        print(f"Loaded {len(timestamps)} timestamps from {files['csv'].name}\n")

        # Check if this ID should be rotated
        should_rotate = file_id in args.rotate
        if should_rotate:
            print(f"  [INFO] ID {file_id} will be rotated 90 degrees clockwise\n")

        # Prepare check visualization path
        check_output_path = None
        if args.check:
            check_output_path = output_directory / f"check_gaze_{file_id}.jpg"

        # Generate output filename in base directory
        output_path = output_directory / f"gaze_{file_id}.csv"

        # Create CSV file and write header
        print(f"Creating output file: {output_path}")
        header = generate_csv_header()
        with open(output_path, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(header)
            csv_file.flush()
            print(f"  [SUCCESS] CSV file created with header ({len(header)} columns)\n")

            # Process video and estimate gaze
            process_video_with_gaze(
                files['avi'], timestamps, csv_file,
                gaze_estimator=gaze_estimator,
                model_type=args.model,
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
