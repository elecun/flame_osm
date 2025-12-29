#!/usr/bin/env python3
"""
Dense Face Keypoints Generator using MediaPipe
Processes video files and generates face landmark CSV files using MediaPipe FaceMesh (468 landmarks).
"""

import argparse
import csv
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO


# Face landmark names (468 points for MediaPipe FaceMesh)
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
    # Quick BOM sniff
    try:
        with open(file_path, 'rb') as f:
            start_bytes = f.read(4)
        if start_bytes.startswith(b'\xff\xfe') or start_bytes.startswith(b'\xfe\xff'):
            return 'utf-16'
        if start_bytes.startswith(b'\xef\xbb\xbf'):
            return 'utf-8-sig'
    except Exception:
        pass

    encodings = ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'cp949', 'euc-kr', 'latin-1']

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


def find_file_pairs(directory: Path, selected_ids: Optional[List[int]] = None) -> Dict[int, Dict[str, Path]]:
    """
    Find and pair AVI and CSV files by their ID suffix.

    Args:
        directory: Directory to search for files
        selected_ids: Optional list of IDs to filter (only return pairs with these IDs)

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
        # Skip macOS hidden files
        if csv_file.name.startswith('._'):
            print(f"  Skipping macOS hidden file: {csv_file.name}")
            continue

        file_id = extract_id_from_filename(csv_file.name)
        if file_id is not None:
            csv_files[file_id] = csv_file
            print(f"  Found CSV file: {csv_file.name} (ID: {file_id})")
        else:
            print(f"  Skipping CSV file (no ID found): {csv_file.name}")

    # Pair AVI and CSV files
    print(f"\nPairing files...")
    pairs = {}
    for file_id in avi_files:
        # Skip if selected_ids is provided and this ID is not in the list
        if selected_ids is not None and file_id not in selected_ids:
            print(f"  [SKIPPED] ID {file_id}: Not in selected IDs")
            continue

        if file_id in csv_files:
            pairs[file_id] = {
                'avi': avi_files[file_id],
                'csv': csv_files[file_id]
            }
            print(f"  [ID {file_id}] Paired: {avi_files[file_id].name} + {csv_files[file_id].name}")
        else:
            print(f"  [ID {file_id}] WARNING: No matching CSV file for {avi_files[file_id].name}")

    # Check for unmatched CSV files
    for file_id in csv_files:
        if file_id not in avi_files:
            print(f"  [ID {file_id}] WARNING: No matching AVI file for {csv_files[file_id].name}")

    print(f"\n{'=' * 80}")
    print(f"Found {len(pairs)} valid pair(s)")
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

    with open(csv_path, 'r', encoding=encoding, errors='replace', newline='') as f:
        raw_lines = f.readlines()

    rows = []
    for idx, line in enumerate(raw_lines, start=1):
        if '\x00' in line:
            raise csv.Error(f"NUL found in timestamp file at line {idx}")
        rows.append(next(csv.reader([line])))

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


def visualize_first_frame(frame: np.ndarray, landmarks: np.ndarray, output_path: Path, bbox=None):
    """
    Visualize face landmarks on first frame and save as image.

    Args:
        frame: First frame of video
        landmarks: Array of shape (num_landmarks, 3) with landmark coordinates (x, y, z)
        output_path: Path to save visualization image
        bbox: Optional bounding box tuple (x1, y1, x2, y2) to draw in red
    """
    vis_frame = frame.copy()
    num_landmarks = len(landmarks)

    # Draw bounding box if provided (in red)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color, thickness 2

    # Draw landmark points
    for idx, (x, y, z) in enumerate(landmarks):
        if x != -1.0:
            point = (int(x), int(y))
            # Use single color for all landmarks
            color = (0, 255, 0)  # Green
            cv2.circle(vis_frame, point, 2, color, -1)

    # Save visualization
    cv2.imwrite(str(output_path), vis_frame)
    print(f"  Saved first frame visualization to: {output_path}")


def process_video_with_mediapipe(video_path: Path, face_mesh, yolo_model, csv_file, timestamps: List[float],
                                  should_rotate: bool = False, check_mode: bool = False,
                                  check_output_path: Optional[Path] = None, video_out_path: Optional[Path] = None,
                                  wider: float = 0.0) -> Tuple[np.ndarray, int, List[int]]:
    """
    Process video with YOLO face detection + MediaPipe FaceMesh and write results to CSV immediately.

    Args:
        video_path: Path to video file
        face_mesh: MediaPipe FaceMesh object
        yolo_model: YOLO face detection model (or None to use full frame)
        csv_file: Open CSV file object to write results
        timestamps: List of timestamps for each frame
        should_rotate: Whether to rotate frames 90 degrees clockwise
        check_mode: Whether to save first frame with landmarks visualization
        check_output_path: Path to save check visualization image
        video_out_path: Optional path to save visualization video (AVI format)
        wider: Percentage to widen the face bounding box (e.g., 10 for 10%)

    Returns:
        Tuple of (landmarks, num_landmarks, valid_mask)
    """
    print(f"Processing video: {video_path.name}")
    if should_rotate:
        print(f"  [INFO] Video will be rotated 90 degrees clockwise")
    if yolo_model is not None:
        print(f"  [INFO] Using YOLO face detector for face cropping")
        if wider > 0:
            print(f"  [INFO] Bounding box will be widened by {wider}%")

    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0

    # MediaPipe FaceMesh has 468 landmarks
    num_landmarks = 468
    all_landmarks = np.zeros((frame_count, num_landmarks, 3), dtype=np.float32)
    valid_mask = [0] * frame_count

    csv_writer = csv.writer(csv_file, lineterminator="\n")

    # Initialize VideoWriter if video output is requested
    video_writer = None
    if video_out_path is not None:
        # Get first frame to determine dimensions after rotation
        ret, first_frame = cap.read()
        if ret:
            if should_rotate:
                first_frame = cv2.rotate(first_frame, cv2.ROTATE_90_CLOCKWISE)
            frame_height, frame_width = first_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (frame_width, frame_height))
            print(f"  [INFO] Video output will be saved to: {video_out_path}")
            print(f"  [INFO] Video properties - FPS: {fps:.2f}, Resolution: {frame_width}x{frame_height}")
        # Reset to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Inference time tracking
    yolo_times = []
    mediapipe_times = []

    # Batch processing time tracking
    batch_start_time = time.time()

    # Track if we've saved check image
    check_image_saved = False

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Rotate frame if requested (90 degrees clockwise)
        if should_rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Get frame dimensions
        height, width, _ = frame.shape

        # Extract landmarks
        lms = np.zeros((num_landmarks, 3), dtype=np.float32)
        detected = False

        # Step 1: Detect face with YOLO (if provided)
        bbox = None
        if yolo_model is not None:
            yolo_start = time.time()
            results = yolo_model(frame, verbose=False)
            yolo_times.append(time.time() - yolo_start)

            # Get the first detected face
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Get box with highest confidence
                boxes = results[0].boxes
                confidences = boxes.conf.cpu().numpy()
                best_idx = np.argmax(confidences)

                # Get bounding box coordinates (xyxy format)
                box = boxes.xyxy[best_idx].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)

                # Widen the bounding box if requested
                if wider > 0:
                    box_width = x2 - x1
                    box_height = y2 - y1
                    increase_w = box_width * (wider / 100.0) / 2.0
                    increase_h = box_height * (wider / 100.0) / 2.0
                    x1 = int(x1 - increase_w)
                    y1 = int(y1 - increase_h)
                    x2 = int(x2 + increase_w)
                    y2 = int(y2 + increase_h)

                # Ensure bbox is within frame boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)

                bbox = (x1, y1, x2, y2)

        # Step 2: Process face region with MediaPipe
        if bbox is not None or yolo_model is None:
            # Crop face region if YOLO detected a face
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                face_crop = frame[y1:y2, x1:x2]
                crop_height, crop_width = face_crop.shape[:2]
            else:
                # Use full frame if no YOLO model
                face_crop = frame
                crop_height, crop_width = height, width
                x1, y1 = 0, 0

            # Convert BGR to RGB for MediaPipe
            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

            # Detect landmarks with timing
            mp_start = time.time()
            mp_results = face_mesh.process(face_crop_rgb)
            mediapipe_times.append(time.time() - mp_start)

            if mp_results.multi_face_landmarks:
                # Take the first detected face
                face_landmarks = mp_results.multi_face_landmarks[0]
                detected = True

                # Convert normalized coordinates to pixel coordinates
                # Only process up to num_landmarks (468) to avoid index errors
                for idx, landmark in enumerate(face_landmarks.landmark[:num_landmarks]):
                    # Convert from cropped image coordinates to original image coordinates
                    lms[idx, 0] = (landmark.x * crop_width) + x1
                    lms[idx, 1] = (landmark.y * crop_height) + y1
                    lms[idx, 2] = landmark.z * crop_width  # z is relative to crop width

        all_landmarks[frame_idx] = lms
        valid_mask[frame_idx] = 1 if detected else 0

        # Save first frame with face detection if check mode is enabled
        if check_mode and not check_image_saved and check_output_path is not None:
            if valid_mask[frame_idx]:
                print(f"  Saving check image from frame {frame_idx}...")
                visualize_first_frame(frame, lms, check_output_path, bbox=bbox)
                check_image_saved = True

        # Write visualization frame to video if requested
        if video_writer is not None:
            vis_frame = frame.copy()

            # Draw bounding box if available
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Draw landmark points
            for idx, (x, y, z) in enumerate(lms):
                if valid_mask[frame_idx]:
                    cv2.circle(vis_frame, (int(x), int(y)), 2, (0, 255, 0), -1)

            video_writer.write(vis_frame)

        # Write to CSV immediately
        row = [timestamps[frame_idx]]

        # Add landmarks
        for lm_idx in range(num_landmarks):
            x = lms[lm_idx, 0]
            y = lms[lm_idx, 1]
            z = lms[lm_idx, 2]
            row.append(x)
            row.append(y)
            row.append(z)

        row.append(valid_mask[frame_idx])
        csv_writer.writerow(row)
        csv_file.flush()  # Flush to disk immediately

        if (frame_idx + 1) % 100 == 0:
            batch_elapsed = time.time() - batch_start_time
            print(f"  Processed {frame_idx + 1}/{frame_count} frames ({batch_elapsed*1000:.2f}ms for 100 frames)")
            batch_start_time = time.time()

        frame_idx += 1

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"  [SUCCESS] Video output saved to: {video_out_path}")

    # Check if check image was saved
    if check_mode and not check_image_saved and check_output_path is not None:
        print(f"\n  [WARNING] No face detected in any frame - check image was not generated")

    # Print inference statistics
    if yolo_model is not None and yolo_times:
        avg_yolo = np.mean(yolo_times) * 1000
        print(f"\n  YOLO inference time: {avg_yolo:.2f}ms (avg)")

    if mediapipe_times:
        avg_mp = np.mean(mediapipe_times) * 1000
        min_mp = np.min(mediapipe_times) * 1000
        max_mp = np.max(mediapipe_times) * 1000
        print(f"  MediaPipe inference time statistics:")
        print(f"    Average: {avg_mp:.2f}ms")
        print(f"    Min: {min_mp:.2f}ms")
        print(f"    Max: {max_mp:.2f}ms")

    return all_landmarks, num_landmarks, valid_mask


def generate_csv_header(num_landmarks: int) -> List[str]:
    """
    Generate CSV header with timestamp and landmark names.

    Args:
        num_landmarks: Number of face landmarks (468 for MediaPipe)

    Returns:
        List of column names
    """
    header = ['timestamp']

    # Add landmarks
    landmark_names = generate_landmark_names(num_landmarks)
    for lm_name in landmark_names:
        header.append(f'{lm_name}_x')
        header.append(f'{lm_name}_y')
        header.append(f'{lm_name}_z')

    header.append('valid_mask')

    return header


def process_single_file(file_path: Path, face_mesh, yolo_model, output_path: Path,
                        should_rotate: bool = False, check_mode: bool = False,
                        video_out_path: Optional[Path] = None,
                        wider: float = 0.0) -> None:
    """
    Process a single image or video file for face landmarks.

    Args:
        file_path: Path to image or video file
        face_mesh: MediaPipe FaceMesh object
        yolo_model: YOLO face detection model (or None to use full frame)
        output_path: Output CSV file path
        should_rotate: Whether to rotate frames 90 degrees clockwise
        check_mode: Whether to save first frame with landmarks visualization
        video_out_path: Optional path to save visualization video (AVI format)
        wider: Percentage to widen the face bounding box (e.g., 10 for 10%)
    """
    print(f"Processing single file: {file_path.name}")
    print(f"Output: {output_path}")
    if yolo_model and wider > 0:
        print(f"Bounding box will be widened by {wider}%")
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
        fps = 30.0
    else:
        print("Detected: Video file")
        cap = cv2.VideoCapture(str(file_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30.0
        cap.release()
        timestamps = [i / fps for i in range(frame_count)]
        print(f"  Frame count: {frame_count}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Duration: {frame_count / fps:.2f} seconds\n")

    check_output_path = None
    if check_mode:
        check_output_path = output_path.parent / f"check_face_{file_path.stem}.jpg"

    # MediaPipe FaceMesh has 468 landmarks
    num_landmarks = 468

    # Create CSV header
    header = generate_csv_header(num_landmarks)

    # For single file, process with the video processing function
    with open(output_path, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)

        if is_video:
            landmarks, num_landmarks, valid_mask = process_video_with_mediapipe(
                file_path, face_mesh, yolo_model, csv_file, timestamps,
                should_rotate=should_rotate, check_mode=check_mode, check_output_path=check_output_path,
                video_out_path=video_out_path, wider=wider
            )
        else:
            # Process single image
            img = cv2.imread(str(file_path))
            if img is None:
                raise ValueError(f"Cannot read image: {file_path}")

            if should_rotate:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                print("  [INFO] Image rotated 90 degrees clockwise")

            # Get image dimensions
            height, width, _ = img.shape

            # Extract landmarks
            lms = np.zeros((num_landmarks, 3), dtype=np.float32)
            detected = False

            # Step 1: Detect face with YOLO (if provided)
            bbox = None
            if yolo_model is not None:
                print("  Detecting face with YOLO...")
                results = yolo_model(img, verbose=False)

                # Get the first detected face
                if len(results) > 0 and len(results[0].boxes) > 0:
                    # Get box with highest confidence
                    boxes = results[0].boxes
                    confidences = boxes.conf.cpu().numpy()
                    best_idx = np.argmax(confidences)

                    # Get bounding box coordinates (xyxy format)
                    box = boxes.xyxy[best_idx].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box)

                    # Widen the bounding box if requested
                    if wider > 0:
                        box_width = x2 - x1
                        box_height = y2 - y1
                        increase_w = box_width * (wider / 100.0) / 2.0
                        increase_h = box_height * (wider / 100.0) / 2.0
                        x1 = int(x1 - increase_w)
                        y1 = int(y1 - increase_h)
                        x2 = int(x2 + increase_w)
                        y2 = int(y2 + increase_h)

                    # Ensure bbox is within frame boundaries
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)

                    bbox = (x1, y1, x2, y2)
                    print(f"  Face detected at [{x1}, {y1}, {x2}, {y2}]")
                else:
                    print("  No face detected by YOLO")

            # Step 2: Process face region with MediaPipe
            if bbox is not None or yolo_model is None:
                # Crop face region if YOLO detected a face
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    face_crop = img[y1:y2, x1:x2]
                    crop_height, crop_width = face_crop.shape[:2]
                else:
                    # Use full frame if no YOLO model
                    face_crop = img
                    crop_height, crop_width = height, width
                    x1, y1 = 0, 0

                # Convert BGR to RGB for MediaPipe
                face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

                # Detect landmarks
                print("  Processing with MediaPipe FaceMesh...")
                mp_results = face_mesh.process(face_crop_rgb)

                if mp_results.multi_face_landmarks:
                    face_landmarks = mp_results.multi_face_landmarks[0]
                    detected = True
                    # Only process up to num_landmarks (468) to avoid index errors
                    for idx, landmark in enumerate(face_landmarks.landmark[:num_landmarks]):
                        # Convert from cropped image coordinates to original image coordinates
                        lms[idx, 0] = (landmark.x * crop_width) + x1
                        lms[idx, 1] = (landmark.y * crop_height) + y1
                        lms[idx, 2] = landmark.z * crop_width  # z is relative to crop width
                    print(f"  Detected {num_landmarks} face landmarks")
                else:
                    print("  No face landmarks detected by MediaPipe")

            # Save visualization if check mode
            if check_mode and check_output_path is not None:
                if detected:
                    print(f"  Saving check image...")
                    visualize_first_frame(img, lms, check_output_path, bbox=bbox)
                else:
                    print(f"  [WARNING] No face landmarks detected - saving original image for debugging")
                    cv2.imwrite(str(check_output_path), img)
                    print(f"  Saved original image to: {check_output_path}")

            # Write to CSV
            valid_mask = 1 if detected else 0
            row = [timestamps[0]]
            for lm_idx in range(num_landmarks):
                x = lms[lm_idx, 0]
                y = lms[lm_idx, 1]
                z = lms[lm_idx, 2]
                row.append(x)
                row.append(y)
                row.append(z)
            row.append(valid_mask)
            csv_writer.writerow(row)

    print(f"\n[SUCCESS] Processing completed!")
    print(f"  Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate dense face keypoints (468 landmarks) using MediaPipe FaceMesh'
    )

    # Input/output arguments
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
        '--recursive',
        action='store_true',
        help='Treat --path as a parent directory containing multiple case folders (batch only)'
    )

    # Processing options
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
        help='Save first frame with landmarks visualization as JPG'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate frame counts without processing'
    )

    # MediaPipe options
    parser.add_argument('--max-num-faces', type=int, default=1,
                        help='Maximum number of faces to detect (default: 1)')
    parser.add_argument('--min-detection-confidence', type=float, default=0.5,
                        help='Minimum confidence for face detection (default: 0.5)')
    parser.add_argument('--min-tracking-confidence', type=float, default=0.5,
                        help='Minimum confidence for face tracking (default: 0.5)')

    # YOLO face detector option
    parser.add_argument('--face-detector', type=str, default=None,
                        help='Path to YOLO face detector model (e.g., yolo11n-face.pt). If not provided, uses full frame.')
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
    parser.add_argument(
        '--wider',
        type=float,
        default=0.0,
        help='Percentage to widen the face bounding box (e.g., 10 for 10%%). Default: 0.'
    )

    args = parser.parse_args()

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
        output_filename = f"face_kps_dense_{file_path.stem}.csv"
        output_path = output_dir / output_filename

        # Check if rotation is requested (--rotate flag present means rotate)
        should_rotate = args.rotate is not None

        # Prepare video output path if requested
        video_out_path = None
        if args.video_out:
            video_out_path = output_dir / f"face_kps_dense_{file_path.stem}_visualization.avi"

        print(f"\n{'=' * 80}")
        print(f"Dense Face Keypoints Generator (MediaPipe)")
        print(f"Mode: Single File")
        print(f"Input file: {file_path}")
        print(f"Output file: {output_path}")
        print(f"{'=' * 80}\n")

        # Load YOLO face detector if provided
        yolo_model = None
        if args.face_detector:
            print(f"Loading YOLO face detector: {args.face_detector}")
            yolo_model = YOLO(args.face_detector)
            print(f"  YOLO model loaded successfully\n")

        # Initialize MediaPipe FaceMesh
        print(f"Initializing MediaPipe FaceMesh...")
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=args.max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence
        )
        print(f"  MediaPipe FaceMesh initialized with 468 landmarks\n")

        try:
            process_single_file(
                file_path=file_path,
                face_mesh=face_mesh,
                yolo_model=yolo_model,
                output_path=output_path,
                should_rotate=should_rotate,
                check_mode=args.check,
                video_out_path=video_out_path,
                wider=args.wider
            )
        except Exception as e:
            print(f"\nError processing file: {e}")
            import traceback
            traceback.print_exc()
            return 1
        finally:
            face_mesh.close()

        return 0

    def process_batch(base_directory: Path, face_mesh, yolo_model) -> int:
        if not base_directory.exists():
            print(f"Error: Directory does not exist: {base_directory}")
            return 1
        if not base_directory.is_dir():
            print(f"Error: Path is not a directory: {base_directory}")
            return 1

        # Input files are in 'camera' subdirectory
        input_directory = base_directory / "camera"
        if not input_directory.exists():
            print(f"Error: Camera directory does not exist: {input_directory}")
            return 1
        if not input_directory.is_dir():
            print(f"Error: Camera path is not a directory: {input_directory}")
            return 1

        # Output directory is the base directory
        output_directory = base_directory

        print(f"\n{'=' * 80}")
        print(f"Dense Face Keypoints Generator (MediaPipe)")
        print(f"Mode: Batch")
        print(f"Input directory: {input_directory}")
        print(f"Output directory: {output_directory}")
        print(f"Rotate IDs: {args.rotate if args.rotate is not None else 'None'}")
        print(f"Check mode: {args.check}")
        print(f"Video output: {args.video_out}")
        if args.wider > 0:
            print(f"Widen bbox: {args.wider}%")

        # Parse selected IDs if provided
        selected_ids = None
        if args.select:
            try:
                selected_ids = [int(x.strip()) for x in args.select.split(',')]
                print(f"Selected IDs: {selected_ids}")
            except ValueError:
                print(f"Error: Invalid format for --select. Use comma-separated integers (e.g., '0,2,4,6')")
                return 1

        print(f"{'=' * 80}\n")

        # Find file pairs
        pairs = find_file_pairs(input_directory, selected_ids=selected_ids)

        if not pairs:
            print("[ERROR] No valid file pairs found!")
            return 1

        # Validate frame counts if requested
        if args.validate_only:
            validate_frame_counts(pairs)
            return 0

        # Determine which IDs to rotate
        rotate_ids = set(args.rotate) if args.rotate else set()

        # Process each pair
        for file_id, files in sorted(pairs.items()):
            print(f"{'=' * 80}")
            print(f"Processing ID: {file_id}")
            print(f"  AVI: {files['avi'].name}")
            print(f"  CSV: {files['csv'].name}")

            should_rotate = file_id in rotate_ids
            if should_rotate:
                print(f"  [INFO] ID {file_id} will be rotated 90 degrees clockwise\n")

            # Prepare check visualization path in base directory
            check_output_path = None
            if args.check:
                check_output_path = output_directory / f"check_face_dense_{file_id}.jpg"

            # Prepare video output path if requested
            video_out_path = None
            if args.video_out:
                video_out_path = output_directory / f"face_kps_dense_{file_id}_visualization.avi"

            # Generate output filename in base directory
            output_path = output_directory / f"face_kps_dense_{file_id}.csv"

            # Read timestamps
            timestamps = read_timestamps(files['csv'])

            # Create CSV file and write header immediately
            print(f"Creating output file: {output_path}")
            num_landmarks = 468
            header = generate_csv_header(num_landmarks)

            with open(output_path, 'w', newline='', encoding='utf-8', errors='strict') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(header)
                csv_file.flush()
                print(f"  [SUCCESS] CSV file created with header ({len(header)} columns)\n")

                # Process video
                landmarks, num_landmarks, valid_mask = process_video_with_mediapipe(
                    files['avi'], face_mesh, yolo_model, csv_file, timestamps,
                    should_rotate=should_rotate, check_mode=args.check, check_output_path=check_output_path,
                    video_out_path=video_out_path, wider=args.wider
                )

            if landmarks is None:
                print(f"  [ERROR] Failed to process video\n")
                continue

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

    # Load YOLO face detector if provided (shared)
    yolo_model = None
    if args.face_detector:
        print(f"Loading YOLO face detector: {args.face_detector}")
        yolo_model = YOLO(args.face_detector)
        print(f"  YOLO model loaded successfully\n")

    # Initialize MediaPipe FaceMesh (shared)
    print(f"Initializing MediaPipe FaceMesh...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=args.max_num_faces,
        refine_landmarks=True,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    )
    print(f"  MediaPipe FaceMesh initialized with 468 landmarks")
    print()

    had_error = False
    for case_dir in case_dirs:
        if args.recursive:
            print(f"\n{'=' * 80}")
            print(f"[CASE] {case_dir}")
            print(f"{'=' * 80}")
        if process_batch(case_dir, face_mesh, yolo_model) != 0:
            had_error = True

    face_mesh.close()

    return 0 if not had_error else 1


if __name__ == '__main__':
    exit(main())
