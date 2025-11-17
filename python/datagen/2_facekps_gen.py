#!/usr/bin/env python3
"""
Face Keypoints Generator
Processes video files and generates face landmark CSV files using Face Alignment Network.
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import face_alignment


# Face landmark names (68 points for 2D landmarks)
def generate_landmark_names(num_landmarks: int) -> List[str]:
    """Generate landmark names based on number of landmarks."""
    return [f'landmark_{i}' for i in range(num_landmarks)]


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

    # Default to utf-8 if all fail
    return 'utf-8'


def extract_id_from_filename(filename: str) -> Optional[int]:
    """
    Extract numeric ID from the end of filename.

    Args:
        filename: Filename to extract ID from (e.g., 'cam_0.avi', 'timestamp_0.csv')

    Returns:
        Numeric ID or None if not found
    """
    # Match pattern: underscore followed by digits before file extension
    match = re.search(r'_(\d+)\.[^.]+$', filename)
    if match:
        return int(match.group(1))
    return None


def find_file_pairs(directory: Path) -> Dict[int, Dict[str, Path]]:
    """
    Find and pair AVI and CSV files by their ID suffix.

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
    print(f"Found {len(all_avi)} AVI file(s) in directory\n")

    for avi_file in all_avi:
        # Skip macOS hidden files (._*)
        if avi_file.name.startswith('._'):
            print(f"  Skipping macOS hidden file: {avi_file.name}")
            continue

        file_id = extract_id_from_filename(avi_file.name)
        if file_id is not None:
            avi_files[file_id] = avi_file
            print(f"  Found AVI file: {avi_file.name} (ID: {file_id})")
        else:
            print(f"  Skipping AVI file (no ID found): {avi_file.name}")

    # Find all CSV files with ID suffix
    print(f"\nSearching for CSV files...")
    csv_files = {}
    all_csv = list(directory.glob('*.csv'))
    print(f"Found {len(all_csv)} CSV file(s) in directory\n")

    for csv_file in all_csv:
        # Skip macOS hidden files (._*)
        if csv_file.name.startswith('._'):
            print(f"  Skipping macOS hidden file: {csv_file.name}")
            continue

        file_id = extract_id_from_filename(csv_file.name)
        if file_id is not None:
            csv_files[file_id] = csv_file
            print(f"  Found CSV file: {csv_file.name} (ID: {file_id})")
        else:
            print(f"  Skipping CSV file (no ID found): {csv_file.name}")

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
            print(f"  [WARNING] ID {file_id}: AVI file '{avi_files[file_id].name}' has no matching CSV")

    # Check for CSV files without matching AVI
    for file_id in csv_files.keys():
        if file_id not in avi_files:
            print(f"  [WARNING] ID {file_id}: CSV file '{csv_files[file_id].name}' has no matching AVI")

    print(f"\n{'=' * 80}")
    print(f"Total pairs found: {len(pairs)}")
    print(f"{'=' * 80}\n")

    return pairs


def validate_frame_counts(pairs: Dict[int, Dict[str, Path]]) -> Dict[int, Tuple[int, int, bool]]:
    """
    Validate that frame count matches row count for each pair.

    Args:
        pairs: Dictionary of file pairs

    Returns:
        Dictionary mapping ID to (frame_count, row_count, is_valid)
    """
    print(f"{'=' * 80}")
    print(f"Validating frame counts")
    print(f"{'=' * 80}\n")

    validation_results = {}

    for file_id, files in pairs.items():
        avi_path = files['avi']
        csv_path = files['csv']

        print(f"[ID {file_id}] Validating {avi_path.name}")

        # Get frame count from AVI
        cap = cv2.VideoCapture(str(avi_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"  AVI frame count: {frame_count}")

        # Get row count from CSV (detect header automatically)
        encoding = detect_csv_encoding(csv_path)
        print(f"  Detected CSV encoding: {encoding}")
        with open(csv_path, 'r', encoding=encoding) as f:
            reader = csv.reader(f)
            rows = list(reader)

            # Check if first row is a header
            has_header = False
            if rows:
                try:
                    float(rows[0][0].strip())
                    has_header = False
                except ValueError:
                    has_header = True

            row_count = len(rows) - (1 if has_header else 0)
        print(f"  CSV row count: {row_count} (header detected: {has_header})")

        is_valid = frame_count == row_count
        if is_valid:
            print(f"  [SUCCESS] Frame count matches row count\n")
        else:
            print(f"  [ERROR] Frame count mismatch! Difference: {abs(frame_count - row_count)}\n")

        validation_results[file_id] = (frame_count, row_count, is_valid)

    return validation_results


def read_timestamps(csv_path: Path) -> List[float]:
    """
    Read timestamps from CSV file.
    Automatically detects if CSV has header or not.

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

        # Check if first row is a header (try to convert to float)
        first_value = rows[0][0].strip()
        has_header = False

        try:
            float(first_value)
            # First row is a number, no header
            has_header = False
        except ValueError:
            # First row is not a number, it's a header
            has_header = True

        # Read timestamps starting from appropriate row
        start_idx = 1 if has_header else 0
        for row in rows[start_idx:]:
            if row:  # Skip empty rows
                timestamp = float(row[0])
                timestamps.append(timestamp)

    return timestamps


def autofill_missing_values(landmarks_sequence: np.ndarray) -> np.ndarray:
    """
    Fill missing values (NaN) with average of neighboring 5 values.

    Args:
        landmarks_sequence: Array of shape (num_frames, num_landmarks, 2 or 3)

    Returns:
        Array with filled values
    """
    filled = landmarks_sequence.copy()
    num_frames, num_landmarks, num_coords = filled.shape

    for lm_idx in range(num_landmarks):
        for coord_idx in range(num_coords):  # x, y or x, y, z
            for frame_idx in range(num_frames):
                if np.isnan(filled[frame_idx, lm_idx, coord_idx]):
                    # Get neighboring values within window
                    window_start = max(0, frame_idx - 5)
                    window_end = min(num_frames, frame_idx + 6)

                    neighbor_values = []
                    for i in range(window_start, window_end):
                        if i != frame_idx and not np.isnan(filled[i, lm_idx, coord_idx]):
                            neighbor_values.append(filled[i, lm_idx, coord_idx])

                    # Fill with average of neighbors
                    if neighbor_values:
                        filled[frame_idx, lm_idx, coord_idx] = np.mean(neighbor_values)
                    else:
                        # If no valid neighbors, keep as 0
                        filled[frame_idx, lm_idx, coord_idx] = 0.0

    return filled


def visualize_first_frame(frame: np.ndarray, landmarks: np.ndarray, output_path: Path):
    """
    Visualize face landmarks on first frame and save as image.

    Args:
        frame: First frame of video
        landmarks: Array of shape (num_landmarks, 2) with landmark coordinates
        output_path: Path to save visualization image
    """
    # Create a copy of the frame
    vis_frame = frame.copy()

    num_landmarks = len(landmarks)

    # Define landmark connections for 68-point model
    if num_landmarks >= 68:
        # Face landmark connections (based on 68-point model)
        connections = {
            'jaw': list(range(0, 17)),  # Jawline
            'left_eyebrow': list(range(17, 22)),  # Left eyebrow
            'right_eyebrow': list(range(22, 27)),  # Right eyebrow
            'nose_bridge': list(range(27, 31)),  # Nose bridge
            'nose_bottom': list(range(31, 36)),  # Nose bottom
            'left_eye': list(range(36, 42)) + [36],  # Left eye (closed loop)
            'right_eye': list(range(42, 48)) + [42],  # Right eye (closed loop)
            'outer_lip': list(range(48, 60)) + [48],  # Outer lip (closed loop)
            'inner_lip': list(range(60, 68)) + [60],  # Inner lip (closed loop)
        }
    else:
        # For other landmark counts, just connect sequentially
        connections = {
            'all': list(range(num_landmarks))
        }

    # Draw connections with white lines
    line_color = (255, 255, 255)  # White
    line_thickness = 1

    for region, indices in connections.items():
        for i in range(len(indices) - 1):
            idx1 = indices[i]
            idx2 = indices[i + 1]

            if idx1 < num_landmarks and idx2 < num_landmarks:
                pt1 = landmarks[idx1]
                pt2 = landmarks[idx2]

                # Only draw if both points are valid
                if not (np.isnan(pt1[0]) or np.isnan(pt2[0])):
                    start_point = (int(pt1[0]), int(pt1[1]))
                    end_point = (int(pt2[0]), int(pt2[1]))
                    cv2.line(vis_frame, start_point, end_point, line_color, line_thickness)

    # Draw landmark points on top of lines
    for idx, (x, y) in enumerate(landmarks):
        if not np.isnan(x):
            point = (int(x), int(y))
            # Different colors for different face regions
            if idx < 17:  # Jawline
                color = (255, 0, 0)  # Blue
            elif idx < 27:  # Eyebrows
                color = (0, 255, 0)  # Green
            elif idx < 36:  # Nose
                color = (0, 255, 255)  # Yellow
            elif idx < 48:  # Eyes
                color = (255, 0, 255)  # Magenta
            else:  # Mouth
                color = (0, 165, 255)  # Orange

            cv2.circle(vis_frame, point, 2, color, -1)

    # Save visualization
    cv2.imwrite(str(output_path), vis_frame)
    print(f"  [CHECK] First frame visualization saved to: {output_path}")


def process_video_with_face_alignment(video_path: Path, fa_model_2d: Optional[face_alignment.FaceAlignment],
                                     fa_model_3d: Optional[face_alignment.FaceAlignment],
                                     csv_file, timestamps: List[float], landmark_type: str,
                                     should_rotate: bool = False, check_mode: bool = False,
                                     check_output_path: Optional[Path] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, int]:
    """
    Process video and extract face landmarks using Face Alignment Network.
    Writes results to CSV file in real-time for each frame.

    Args:
        video_path: Path to video file
        fa_model_2d: Face Alignment model for 2D landmarks (or None)
        fa_model_3d: Face Alignment model for 3D landmarks (or None)
        csv_file: Open CSV file object to write results
        timestamps: List of timestamps for each frame
        landmark_type: Type of landmarks ('2d', '3d', or 'both')
        should_rotate: Whether to rotate frames 90 degrees clockwise
        check_mode: Whether to save first frame with landmarks visualization
        check_output_path: Path to save check visualization image

    Returns:
        Tuple of (landmarks_2d, landmarks_3d, num_landmarks_2d, num_landmarks_3d)
    """
    print(f"Processing video: {video_path.name}")
    if should_rotate:
        print(f"  [INFO] Video will be rotated 90 degrees clockwise")

    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Process first frame to determine number of landmarks
    ret, first_frame = cap.read()
    if not ret:
        print(f"  [ERROR] Could not read first frame")
        cap.release()
        return None, None, 0, 0

    if should_rotate:
        first_frame = cv2.rotate(first_frame, cv2.ROTATE_90_CLOCKWISE)

    # Detect 2D landmarks if requested
    num_landmarks_2d = 0
    all_landmarks_2d = None
    if fa_model_2d is not None:
        first_landmarks_2d = fa_model_2d.get_landmarks(first_frame)
        if first_landmarks_2d is None or len(first_landmarks_2d) == 0:
            print(f"  [ERROR] No face detected in first frame for 2D landmarks")
            cap.release()
            return None, None, 0, 0
        num_landmarks_2d = len(first_landmarks_2d[0])
        print(f"  Detected {num_landmarks_2d} 2D landmarks per face")
        all_landmarks_2d = np.full((frame_count, num_landmarks_2d, 2), np.nan)

    # Detect 3D landmarks if requested
    num_landmarks_3d = 0
    all_landmarks_3d = None
    if fa_model_3d is not None:
        first_landmarks_3d = fa_model_3d.get_landmarks(first_frame)
        if first_landmarks_3d is None or len(first_landmarks_3d) == 0:
            print(f"  [ERROR] No face detected in first frame for 3D landmarks")
            cap.release()
            return None, None, 0, 0
        num_landmarks_3d = len(first_landmarks_3d[0])
        print(f"  Detected {num_landmarks_3d} 3D landmarks per face")
        all_landmarks_3d = np.full((frame_count, num_landmarks_3d, 3), np.nan)

    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    csv_writer = csv.writer(csv_file)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Rotate frame if requested (90 degrees clockwise)
        if should_rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Detect 2D landmarks
        lms_2d = None
        if fa_model_2d is not None:
            landmarks_2d = fa_model_2d.get_landmarks(frame)
            lms_2d = np.full((num_landmarks_2d, 2), np.nan)
            if landmarks_2d is not None and len(landmarks_2d) > 0:
                lms_2d = landmarks_2d[0][:num_landmarks_2d]
            all_landmarks_2d[frame_idx] = lms_2d

        # Detect 3D landmarks
        lms_3d = None
        if fa_model_3d is not None:
            landmarks_3d = fa_model_3d.get_landmarks(frame)
            lms_3d = np.full((num_landmarks_3d, 3), np.nan)
            if landmarks_3d is not None and len(landmarks_3d) > 0:
                lms_3d = landmarks_3d[0][:num_landmarks_3d]
            all_landmarks_3d[frame_idx] = lms_3d

        # Save first frame with landmarks visualization if check mode is enabled
        if check_mode and frame_idx == 0 and check_output_path is not None:
            if lms_2d is not None:
                visualize_first_frame(frame, lms_2d, check_output_path)
            elif lms_3d is not None:
                visualize_first_frame(frame, lms_3d[:, :2], check_output_path)  # Use x, y only

        # Write to CSV immediately
        row = [timestamps[frame_idx]]

        # Add 2D landmarks first
        if lms_2d is not None:
            for lm_idx in range(num_landmarks_2d):
                x = lms_2d[lm_idx, 0]
                y = lms_2d[lm_idx, 1]
                row.append(x)
                row.append(y)

        # Add 3D landmarks after 2D
        if lms_3d is not None:
            for lm_idx in range(num_landmarks_3d):
                x = lms_3d[lm_idx, 0]
                y = lms_3d[lm_idx, 1]
                z = lms_3d[lm_idx, 2]
                row.append(x)
                row.append(y)
                row.append(z)

        csv_writer.writerow(row)
        csv_file.flush()  # Flush to disk immediately

        if (frame_idx + 1) % 100 == 0:
            print(f"  Processed {frame_idx + 1}/{frame_count} frames")

        frame_idx += 1

    cap.release()
    print(f"  [SUCCESS] Processed all {frame_count} frames\n")

    return all_landmarks_2d, all_landmarks_3d, num_landmarks_2d, num_landmarks_3d


def generate_csv_header(num_landmarks_2d: int, num_landmarks_3d: int, landmark_type: str) -> List[str]:
    """
    Generate CSV header with timestamp and landmark names.

    Args:
        num_landmarks_2d: Number of 2D face landmarks
        num_landmarks_3d: Number of 3D face landmarks
        landmark_type: Type of landmarks ('2d', '3d', or 'both')

    Returns:
        List of column names
    """
    header = ['timestamp']

    # Add 2D landmarks first
    if landmark_type in ['2d', 'both'] and num_landmarks_2d > 0:
        landmark_names_2d = generate_landmark_names(num_landmarks_2d)
        for lm_name in landmark_names_2d:
            header.append(f'{lm_name}_x_2d')
            header.append(f'{lm_name}_y_2d')

    # Add 3D landmarks after 2D
    if landmark_type in ['3d', 'both'] and num_landmarks_3d > 0:
        landmark_names_3d = generate_landmark_names(num_landmarks_3d)
        for lm_name in landmark_names_3d:
            header.append(f'{lm_name}_x_3d')
            header.append(f'{lm_name}_y_3d')
            header.append(f'{lm_name}_z_3d')

    return header


def main():
    parser = argparse.ArgumentParser(
        description='Generate face landmarks CSV from video files using Face Alignment Network'
    )
    parser.add_argument(
        '--path',
        required=True,
        help='Directory containing AVI and timestamp CSV files'
    )
    parser.add_argument(
        '--autofill',
        action='store_true',
        help='Autofill missing values with average of neighboring 5 values'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for face alignment (default: cuda)'
    )
    parser.add_argument(
        '--type',
        default='both',
        choices=['2d', '3d', 'both'],
        help='Type of landmarks to extract: 2d, 3d, or both (default: both)'
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
        help='Save first frame with landmarks visualization as JPG'
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
    print(f"Face Keypoints Generator")
    print(f"Input Directory: {input_directory}")
    print(f"Output Directory: {output_directory}")
    print(f"Autofill: {args.autofill}")
    print(f"Device: {args.device}")
    print(f"Landmark Type: {args.type}")
    print(f"Rotate IDs: {args.rotate if args.rotate else 'None'}")
    print(f"Check mode: {args.check}")
    print(f"{'=' * 80}\n")

    # Find file pairs in camera directory
    pairs = find_file_pairs(input_directory)

    if not pairs:
        print("No file pairs found. Exiting.")
        return 1

    # Validate frame counts
    validation_results = validate_frame_counts(pairs)

    # Check for any mismatches
    mismatches = [file_id for file_id, (_, _, is_valid) in validation_results.items() if not is_valid]
    if mismatches:
        print(f"[WARNING] Found {len(mismatches)} file pair(s) with frame count mismatch")
        print(f"IDs with mismatch: {mismatches}")
        response = input("Continue processing? (y/n): ")
        if response.lower() != 'y':
            print("Exiting.")
            return 1
        print()

    # Load Face Alignment model(s)
    print(f"{'=' * 80}")
    print(f"Loading Face Alignment model(s) on {args.device}")
    print(f"{'=' * 80}\n")

    fa_model_2d = None
    fa_model_3d = None

    if args.type in ['2d', 'both']:
        print(f"Loading 2D Face Alignment model with BlazeFace detector...")
        fa_model_2d = face_alignment.FaceAlignment(
            landmarks_type=face_alignment.LandmarksType.TWO_D,
            device=args.device,
            flip_input=False,
            face_detector='blazeface',
            face_detector_kwargs={'back_model': True}
        )
        print(f"[SUCCESS] 2D model loaded\n")

    if args.type in ['3d', 'both']:
        print(f"Loading 3D Face Alignment model with BlazeFace detector...")
        fa_model_3d = face_alignment.FaceAlignment(
            landmarks_type=face_alignment.LandmarksType.THREE_D,
            device=args.device,
            flip_input=False,
            face_detector='blazeface',
            face_detector_kwargs={'back_model': True}
        )
        print(f"[SUCCESS] 3D model loaded\n")

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

        # Prepare check visualization path in base directory
        check_output_path = None
        if args.check:
            check_output_path = output_directory / f"check_face_{file_id}.jpg"

        # Detect number of landmarks from first frame
        print(f"Detecting number of landmarks...")
        cap = cv2.VideoCapture(str(files['avi']))
        ret, first_frame = cap.read()
        cap.release()

        if not ret:
            print(f"  [ERROR] Could not read video file\n")
            continue

        if should_rotate:
            first_frame = cv2.rotate(first_frame, cv2.ROTATE_90_CLOCKWISE)

        num_landmarks_2d = 0
        num_landmarks_3d = 0

        if fa_model_2d is not None:
            first_landmarks_2d = fa_model_2d.get_landmarks(first_frame)
            if first_landmarks_2d is None or len(first_landmarks_2d) == 0:
                print(f"  [ERROR] No face detected in first frame for 2D, skipping ID {file_id}\n")
                continue
            num_landmarks_2d = len(first_landmarks_2d[0])
            print(f"  Detected {num_landmarks_2d} 2D landmarks")

        if fa_model_3d is not None:
            first_landmarks_3d = fa_model_3d.get_landmarks(first_frame)
            if first_landmarks_3d is None or len(first_landmarks_3d) == 0:
                print(f"  [ERROR] No face detected in first frame for 3D, skipping ID {file_id}\n")
                continue
            num_landmarks_3d = len(first_landmarks_3d[0])
            print(f"  Detected {num_landmarks_3d} 3D landmarks")

        print()

        # Generate output filename in base directory
        output_path = output_directory / f"face_kps_{file_id}.csv"

        # Create CSV file and write header immediately
        print(f"Creating output file: {output_path}")
        header = generate_csv_header(num_landmarks_2d, num_landmarks_3d, args.type)
        with open(output_path, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(header)
            csv_file.flush()
            print(f"  [SUCCESS] CSV file created with header ({len(header)} columns)\n")

            # Process video and write results in real-time
            landmarks_2d, landmarks_3d, _, _ = process_video_with_face_alignment(
                files['avi'], fa_model_2d, fa_model_3d, csv_file, timestamps, args.type,
                should_rotate=should_rotate,
                check_mode=args.check,
                check_output_path=check_output_path
            )

        if (fa_model_2d is not None and landmarks_2d is None) or (fa_model_3d is not None and landmarks_3d is None):
            print(f"  [ERROR] Failed to process video\n")
            continue

        print(f"  [SUCCESS] Output saved to: {output_path}\n")

        # Autofill missing values if requested
        if args.autofill:
            print(f"Applying autofill to missing values...")

            total_missing = 0
            if landmarks_2d is not None:
                missing_2d = np.isnan(landmarks_2d).sum()
                total_missing += missing_2d
            if landmarks_3d is not None:
                missing_3d = np.isnan(landmarks_3d).sum()
                total_missing += missing_3d

            if total_missing > 0:
                print(f"  Found {total_missing} missing values")

                if landmarks_2d is not None:
                    landmarks_2d = autofill_missing_values(landmarks_2d)
                if landmarks_3d is not None:
                    landmarks_3d = autofill_missing_values(landmarks_3d)

                # Rewrite CSV with filled values
                print(f"  Rewriting CSV with autofilled values...")
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)

                    for frame_idx, timestamp in enumerate(timestamps):
                        row = [timestamp]

                        # Add 2D landmarks
                        if landmarks_2d is not None:
                            for lm_idx in range(num_landmarks_2d):
                                x = landmarks_2d[frame_idx, lm_idx, 0]
                                y = landmarks_2d[frame_idx, lm_idx, 1]
                                row.append(x)
                                row.append(y)

                        # Add 3D landmarks
                        if landmarks_3d is not None:
                            for lm_idx in range(num_landmarks_3d):
                                x = landmarks_3d[frame_idx, lm_idx, 0]
                                y = landmarks_3d[frame_idx, lm_idx, 1]
                                z = landmarks_3d[frame_idx, lm_idx, 2]
                                row.append(x)
                                row.append(y)
                                row.append(z)

                        writer.writerow(row)

                print(f"  [SUCCESS] Autofill completed and CSV updated\n")
            else:
                print(f"  No missing values found\n")

    print(f"{'=' * 80}")
    print(f"[SUCCESS] All files processed")
    print(f"Total pairs processed: {len(pairs)}")
    print(f"{'=' * 80}")

    return 0


if __name__ == '__main__':
    exit(main())
