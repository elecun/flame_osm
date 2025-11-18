#!/usr/bin/env python3
"""
Body Keypoints Generator
Processes video files and generates body keypoints CSV files using YOLO pose estimation.
"""

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from ultralytics import YOLO


# COCO keypoint names for YOLO pose model (17 keypoints)
KEYPOINT_NAMES = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]


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


def autofill_missing_values(keypoints_sequence: np.ndarray) -> np.ndarray:
    """
    Fill missing values (NaN) with average of neighboring 5 values.

    Args:
        keypoints_sequence: Array of shape (num_frames, num_keypoints, 2)

    Returns:
        Array with filled values
    """
    filled = keypoints_sequence.copy()
    num_frames, num_keypoints, _ = filled.shape

    for kp_idx in range(num_keypoints):
        for coord_idx in range(2):  # x, y
            for frame_idx in range(num_frames):
                if np.isnan(filled[frame_idx, kp_idx, coord_idx]):
                    # Get neighboring values within window
                    window_start = max(0, frame_idx - 5)
                    window_end = min(num_frames, frame_idx + 6)

                    neighbor_values = []
                    for i in range(window_start, window_end):
                        if i != frame_idx and not np.isnan(filled[i, kp_idx, coord_idx]):
                            neighbor_values.append(filled[i, kp_idx, coord_idx])

                    # Fill with average of neighbors
                    if neighbor_values:
                        filled[frame_idx, kp_idx, coord_idx] = np.mean(neighbor_values)
                    else:
                        # If no valid neighbors, keep as 0
                        filled[frame_idx, kp_idx, coord_idx] = 0.0

    return filled


def visualize_first_frame(frame: np.ndarray, keypoints: np.ndarray, output_path: Path):
    """
    Visualize keypoints on first frame and save as image.

    Args:
        frame: First frame of video
        keypoints: Array of shape (17, 2) with keypoint coordinates
        output_path: Path to save visualization image
    """
    # Create a copy of the frame
    vis_frame = frame.copy()

    # Define skeleton connections (COCO format)
    skeleton = [
        (0, 1), (0, 2),  # nose to eyes
        (1, 3), (2, 4),  # eyes to ears
        (0, 5), (0, 6),  # nose to shoulders
        (5, 7), (7, 9),  # left arm
        (6, 8), (8, 10), # right arm
        (5, 11), (6, 12), # shoulders to hips
        (11, 13), (13, 15), # left leg
        (12, 14), (14, 16)  # right leg
    ]

    # Draw skeleton connections
    for start_idx, end_idx in skeleton:
        start_point = keypoints[start_idx]
        end_point = keypoints[end_idx]

        # Only draw if both points are valid (not NaN)
        if not (np.isnan(start_point[0]) or np.isnan(end_point[0])):
            start_pt = (int(start_point[0]), int(start_point[1]))
            end_pt = (int(end_point[0]), int(end_point[1]))
            cv2.line(vis_frame, start_pt, end_pt, (0, 255, 0), 2)

    # Draw keypoints
    for idx, (x, y) in enumerate(keypoints):
        if not np.isnan(x):
            point = (int(x), int(y))
            # Different colors for different keypoint types
            if idx == 0:  # nose
                color = (255, 0, 0)  # Blue
            elif idx in [1, 2, 3, 4]:  # face
                color = (0, 255, 255)  # Yellow
            elif idx in [5, 6, 7, 8, 9, 10]:  # upper body
                color = (0, 255, 0)  # Green
            else:  # lower body
                color = (255, 0, 255)  # Magenta

            cv2.circle(vis_frame, point, 5, color, -1)
            # Add keypoint name
            cv2.putText(vis_frame, KEYPOINT_NAMES[idx],
                       (point[0] + 5, point[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    # Save visualization
    cv2.imwrite(str(output_path), vis_frame)
    print(f"  [CHECK] First frame visualization saved to: {output_path}")


def process_video_with_pose(video_path: Path, model: YOLO, csv_file, timestamps: List[float],
                           should_rotate: bool = False, check_mode: bool = False,
                           check_output_path: Optional[Path] = None) -> np.ndarray:
    """
    Process video and extract pose keypoints using YOLO.
    Writes results to CSV file in real-time for each frame.

    Args:
        video_path: Path to video file
        model: YOLO model
        csv_file: Open CSV file object to write results
        timestamps: List of timestamps for each frame
        should_rotate: Whether to rotate frames 90 degrees clockwise
        check_mode: Whether to save first frame with keypoints visualization
        check_output_path: Path to save check visualization image

    Returns:
        Array of shape (num_frames, 17, 2) containing keypoint coordinates
    """
    print(f"Processing video: {video_path.name}")
    if should_rotate:
        print(f"  [INFO] Video will be rotated 90 degrees clockwise")

    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize array to store keypoints for autofill later: (num_frames, 17 keypoints, 2 coords)
    all_keypoints = np.full((frame_count, 17, 2), np.nan)

    csv_writer = csv.writer(csv_file)

    # Batch processing time tracking
    batch_start_time = time.time()

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Rotate frame if requested (90 degrees clockwise)
        if should_rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Run YOLO pose estimation
        results = model(frame, verbose=False)

        # Extract keypoints from first detected person
        kps = np.full((17, 2), np.nan)  # Default to NaN
        if len(results) > 0 and results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy  # Get xy coordinates
            if len(keypoints) > 0:
                # Take first person's keypoints
                kps = keypoints[0].cpu().numpy()  # Shape: (17, 2)

        all_keypoints[frame_idx] = kps

        # Save first frame with keypoints visualization if check mode is enabled
        if check_mode and frame_idx == 0 and check_output_path is not None:
            visualize_first_frame(frame, kps, check_output_path)

        # Write to CSV immediately
        row = [timestamps[frame_idx]]
        for kp_idx in range(17):
            x = kps[kp_idx, 0]
            y = kps[kp_idx, 1]
            row.append(x)
            row.append(y)
        csv_writer.writerow(row)
        csv_file.flush()  # Flush to disk immediately

        if (frame_idx + 1) % 100 == 0:
            batch_elapsed = time.time() - batch_start_time
            print(f"  Processed {frame_idx + 1}/{frame_count} frames ({batch_elapsed*1000:.2f}ms for 100 frames)")
            batch_start_time = time.time()

        frame_idx += 1

    cap.release()
    print(f"  [SUCCESS] Processed all {frame_count} frames\n")

    return all_keypoints


def process_single_file(file_path: Path, model: YOLO, output_path: Path,
                       should_rotate: bool = False, check_mode: bool = False,
                       autofill: bool = False) -> None:
    """
    Process a single image or video file.

    Args:
        file_path: Path to image or video file
        model: YOLO model
        output_path: Output CSV file path
        should_rotate: Whether to rotate frames 90 degrees clockwise
        check_mode: Whether to save first frame with keypoints visualization
        autofill: Whether to autofill missing values
    """
    print(f"Processing single file: {file_path.name}")
    print(f"Output: {output_path}")
    print(f"{'=' * 80}\n")

    # Check if file is image or video
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    video_extensions = ['.avi', '.mp4', '.mov', '.mkv']
    file_ext = file_path.suffix.lower()

    is_image = file_ext in image_extensions
    is_video = file_ext in video_extensions

    if not is_image and not is_video:
        raise ValueError(f"Unsupported file type: {file_ext}. Must be image or video.")

    # Generate timestamps based on frame count
    if is_image:
        print("Detected: Image file")
        frame_count = 1
        timestamps = [0.0]
        fps = 30.0  # Default FPS for single image
    else:
        print("Detected: Video file")
        cap = cv2.VideoCapture(str(file_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30.0  # Default if FPS cannot be determined
        cap.release()

        # Generate timestamps based on FPS
        timestamps = [i / fps for i in range(frame_count)]
        print(f"  Frame count: {frame_count}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Duration: {frame_count / fps:.2f} seconds\n")

    # Create CSV file and write header
    header = generate_csv_header()
    check_output_path = None
    if check_mode:
        check_output_path = output_path.parent / f"check_{file_path.stem}.jpg"

    with open(output_path, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)
        csv_file.flush()
        print(f"CSV file created with header ({len(header)} columns)\n")

        if is_image:
            # Process single image
            img = cv2.imread(str(file_path))
            if img is None:
                raise ValueError(f"Cannot read image: {file_path}")

            if should_rotate:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                print("  [INFO] Image rotated 90 degrees clockwise")

            # Run YOLO pose estimation with timing
            start_time = time.time()
            results = model(img, verbose=False)
            inference_time = time.time() - start_time
            print(f"  Inference time: {inference_time*1000:.2f}ms\n")

            # Extract keypoints
            kps = np.full((17, 2), np.nan)
            if len(results) > 0 and results[0].keypoints is not None:
                keypoints = results[0].keypoints.xy
                if len(keypoints) > 0:
                    kps = keypoints[0].cpu().numpy()

            # Save visualization if check mode
            if check_mode and check_output_path is not None:
                visualize_first_frame(img, kps, check_output_path)

            # Write to CSV
            row = [timestamps[0]]
            for kp_idx in range(17):
                row.append(kps[kp_idx, 0])
                row.append(kps[kp_idx, 1])
            csv_writer.writerow(row)

            print(f"[SUCCESS] Processed image")
            

        else:
            # Process video
            all_keypoints = np.full((frame_count, 17, 2), np.nan)
            cap = cv2.VideoCapture(str(file_path))

            # Inference time tracking
            inference_times = []

            # Batch processing time tracking
            batch_start_time = time.time()

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if should_rotate:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                # Run YOLO pose estimation with timing
                start_time = time.time()
                results = model(frame, verbose=False)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                # Extract keypoints
                kps = np.full((17, 2), np.nan)
                if len(results) > 0 and results[0].keypoints is not None:
                    keypoints = results[0].keypoints.xy
                    if len(keypoints) > 0:
                        kps = keypoints[0].cpu().numpy()

                all_keypoints[frame_idx] = kps

                # Save first frame visualization if check mode
                if check_mode and frame_idx == 0 and check_output_path is not None:
                    visualize_first_frame(frame, kps, check_output_path)

                # Write to CSV immediately
                row = [timestamps[frame_idx]]
                for kp_idx in range(17):
                    row.append(kps[kp_idx, 0])
                    row.append(kps[kp_idx, 1])
                csv_writer.writerow(row)
                csv_file.flush()

                if (frame_idx + 1) % 100 == 0:
                    batch_elapsed = time.time() - batch_start_time
                    print(f"  Processed {frame_idx + 1}/{frame_count} frames ({batch_elapsed*1000:.2f}ms for 100 frames)")
                    batch_start_time = time.time()

                frame_idx += 1

            cap.release()

            # Print inference time statistics
            if inference_times:
                avg_time = np.mean(inference_times)
                min_time = np.min(inference_times)
                max_time = np.max(inference_times)
                fps = 1.0 / avg_time if avg_time > 0 else 0
                print(f"[SUCCESS] Processed all {frame_count} frames")
                print(f"  Inference time - Avg: {avg_time*1000:.2f}ms, Min: {min_time*1000:.2f}ms, Max: {max_time*1000:.2f}ms")
                print(f"  Average FPS: {fps:.2f}\n")
            else:
                print(f"[SUCCESS] Processed all {frame_count} frames\n")

            # Autofill if requested
            if autofill:
                print(f"Applying autofill to missing values...")
                missing_count = np.isnan(all_keypoints).sum()
                if missing_count > 0:
                    print(f"  Found {missing_count} missing values")
                    all_keypoints = autofill_missing_values(all_keypoints)

                    # Rewrite CSV with filled values
                    print(f"  Rewriting CSV with autofilled values...")
                    with open(output_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(header)

                        for frame_idx, timestamp in enumerate(timestamps):
                            row = [timestamp]
                            for kp_idx in range(17):
                                row.append(all_keypoints[frame_idx, kp_idx, 0])
                                row.append(all_keypoints[frame_idx, kp_idx, 1])
                            writer.writerow(row)

                    print(f"  [SUCCESS] Autofill completed and CSV updated\n")
                else:
                    print(f"  No missing values found\n")

    print(f"[SUCCESS] Output saved to: {output_path}")


def generate_csv_header() -> List[str]:
    """
    Generate CSV header with timestamp and keypoint names.

    Returns:
        List of column names
    """
    header = ['timestamp']
    for kp_name in KEYPOINT_NAMES:
        header.append(f'{kp_name}_x')
        header.append(f'{kp_name}_y')
    return header


def main():
    parser = argparse.ArgumentParser(
        description='Generate body keypoints CSV from video files using YOLO pose estimation'
    )
    parser.add_argument(
        '--path',
        required=True,
        help='Directory or file path (directory for batch mode, file for single mode)'
    )
    parser.add_argument(
        '--no-batch',
        action='store_true',
        help='Single file mode: process a single image or video file'
    )
    parser.add_argument(
        '--autofill',
        action='store_true',
        help='Autofill missing values with average of neighboring 5 values'
    )
    parser.add_argument(
        '--model',
        default='yolov8n-pose.pt',
        help='YOLO pose model to use (default: yolov8n-pose.pt)'
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
        help='Save first frame with keypoints visualization as JPG'
    )

    args = parser.parse_args()

    # Load YOLO model first
    print(f"\n{'=' * 80}")
    print(f"Body Keypoints Generator")
    print(f"Mode: {'Single File' if args.no_batch else 'Batch'}")
    print(f"Model: {args.model}")

    # Model load
    model = YOLO(args.model)

    # Single file mode
    if args.no_batch:
        file_path = Path(args.path)
        if not file_path.exists():
            print(f"Error: File does not exist: {args.path}")
            return 1

        if not file_path.is_file():
            print(f"Error: Path is not a file: {args.path}")
            return 1

        # Determine output path
        output_dir = file_path.parent
        output_filename = f"body_kps_{file_path.stem}.csv"
        output_path = output_dir / output_filename

        # Check if rotation is requested (--rotate flag present means rotate)
        should_rotate = args.rotate is not None

        try:
            process_single_file(
                file_path=file_path,
                model=model,
                output_path=output_path,
                should_rotate=should_rotate,
                check_mode=args.check,
                autofill=args.autofill
            )
        except Exception as e:
            print(f"\nError processing file: {e}")
            import traceback
            traceback.print_exc()
            return 1

        return 0

    # Batch mode (original logic)
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

    print(f"Input Directory: {input_directory}")
    print(f"Output Directory: {output_directory}")
    print(f"Autofill: {args.autofill}")
    print(f"Rotate IDs: {args.rotate if args.rotate is not None else 'None'}")
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

        # Generate output filename in base directory
        output_path = output_directory / f"body_kps_{file_id}.csv"

        # Check if this ID should be rotated
        should_rotate = file_id in args.rotate
        if should_rotate:
            print(f"  [INFO] ID {file_id} will be rotated 90 degrees clockwise\n")

        # Prepare check visualization path in base directory
        check_output_path = None
        if args.check:
            check_output_path = output_directory / f"check_frame_{file_id}.jpg"

        # Create CSV file and write header immediately
        print(f"Creating output file: {output_path}")
        header = generate_csv_header()
        with open(output_path, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(header)
            csv_file.flush()
            print(f"  [SUCCESS] CSV file created with header ({len(header)} columns)\n")

            # Process video and write results in real-time
            keypoints = process_video_with_pose(
                files['avi'], model, csv_file, timestamps,
                should_rotate=should_rotate,
                check_mode=args.check,
                check_output_path=check_output_path
            )

        print(f"  [SUCCESS] Output saved to: {output_path}\n")

        # Autofill missing values if requested
        if args.autofill:
            print(f"Applying autofill to missing values...")
            missing_count = np.isnan(keypoints).sum()
            if missing_count > 0:
                print(f"  Found {missing_count} missing values")
                keypoints = autofill_missing_values(keypoints)

                # Rewrite CSV with filled values
                print(f"  Rewriting CSV with autofilled values...")
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)

                    for frame_idx, timestamp in enumerate(timestamps):
                        row = [timestamp]
                        for kp_idx in range(17):
                            x = keypoints[frame_idx, kp_idx, 0]
                            y = keypoints[frame_idx, kp_idx, 1]
                            row.append(x)
                            row.append(y)
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
