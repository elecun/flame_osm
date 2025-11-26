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
import tempfile
import time
import warnings
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
try:
    from omegaconf import OmegaConf
    from ptgaze.gaze_estimator import GazeEstimator
    from ptgaze.utils import (
        download_dlib_pretrained_model,
        download_mpiigaze_model,
        download_mpiifacegaze_model,
        download_ethxgaze_model,
        expanduser_all,
        generate_dummy_camera_params
    )
    import torch
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


def find_file_pairs(directory: Path, selected_ids: Optional[List[int]] = None) -> Dict[int, Dict[str, Path]]:
    """
    Find and pair AVI and timestamp CSV files by their ID suffix.

    Args:
        directory: Directory to search for files
        selected_ids: Optional list of camera IDs to filter (e.g., [0, 2, 4, 6])

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
            # Filter by selected IDs if provided
            if selected_ids is not None and file_id not in selected_ids:
                continue
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
        frame: Input frame (RGB format)

    Returns:
        Tuple of (face_rect, left_eye_rect, right_eye_rect)
    """
    # Load Haar cascade classifiers
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

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
        frame: Input frame (RGB format)
        left_eye: Left eye rectangle (x, y, w, h)
        right_eye: Right eye rectangle (x, y, w, h)

    Returns:
        Tuple of (pitch, yaw) in degrees
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

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
        frame: Input frame (RGB format, will be converted to BGR for ptgaze)
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

    # Use ptgaze model (ptgaze expects BGR format)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    try:
        # Detect faces
        faces = gaze_estimator.detect_faces(frame_bgr)

        if not faces or len(faces) == 0:
            return np.nan, np.nan

        # Use first detected face
        detected_face = faces[0]

        # Estimate gaze
        gaze_estimator.estimate_gaze(frame_bgr, detected_face)

        # Extract gaze vector from face
        if hasattr(detected_face, 'gaze_vector'):
            gaze_vector = detected_face.gaze_vector
        elif hasattr(detected_face, 'normalized_gaze_vector'):
            gaze_vector = detected_face.normalized_gaze_vector
        else:
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
    except Exception as e:
        print(f"  [WARNING] Gaze estimation failed: {e}")
        return np.nan, np.nan


def visualize_gaze(frame: np.ndarray, face: Tuple, left_eye: Tuple, right_eye: Tuple,
                  pitch: float, yaw: float, output_path: Path):
    """
    Visualize gaze direction on frame.

    Args:
        frame: Input frame (RGB format)
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

    # Save visualization (convert RGB to BGR for cv2.imwrite)
    vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), vis_frame_bgr)
    print(f"  [CHECK] Gaze visualization saved to: {output_path}")


def process_video_with_gaze(video_path: Path, timestamps: List[float], csv_file,
                            gaze_estimator,
                            model_type: str = 'simple',
                            should_rotate: bool = False,
                            check_mode: bool = False,
                            check_output_path: Optional[Path] = None,
                            video_out_path: Optional[Path] = None):
    """
    Process video and estimate gaze for each frame.
    Frames are converted from BGR to RGB before processing.

    Args:
        video_path: Path to video file
        timestamps: List of timestamps
        csv_file: Open CSV file to write results
        gaze_estimator: GazeEstimator instance (or None for simplified method)
        model_type: Model type ('mpiigaze', 'mpiifacegaze', 'eth-xgaze', or 'simple')
        should_rotate: Whether to rotate frames
        check_mode: Whether to save first frame visualization
        check_output_path: Path to save check visualization
        video_out_path: Optional path to save visualization video (AVI format)
    """
    print(f"Processing video: {video_path.name}")
    print(f"  [INFO] Using model: {model_type}")
    if should_rotate:
        print(f"  [INFO] Video will be rotated 90 degrees clockwise")

    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize VideoWriter if video output is requested
    video_writer = None
    if video_out_path is not None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Get frame dimensions
        ret, first_frame = cap.read()
        if ret:
            if should_rotate:
                first_frame = cv2.rotate(first_frame, cv2.ROTATE_90_CLOCKWISE)
            frame_height, frame_width = first_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (frame_width, frame_height))
            print(f"  [INFO] Creating output video: {video_out_path.name} ({frame_width}x{frame_height} @ {fps:.2f} fps)")
            # Reset capture for frame reading
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    csv_writer = csv.writer(csv_file)

    # Inference time tracking
    inference_times = []

    frame_idx = 0
    batch_start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if should_rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # For simplified method, detect face and eyes first
        face, left_eye, right_eye = None, None, None
        if model_type == 'simple':
            face, left_eye, right_eye = detect_face_and_eyes(frame)

        # Estimate gaze direction with timing
        start_time = time.time()
        pitch, yaw = estimate_gaze_direction(
            frame, gaze_estimator, model_type,
            face=face, left_eye=left_eye, right_eye=right_eye
        )
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

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

        # Write visualization frame to video if requested
        if video_writer is not None:
            # For visualization, we need face and eyes
            if face is None or left_eye is None or right_eye is None:
                face, left_eye, right_eye = detect_face_and_eyes(frame)

            # Create visualization frame
            vis_frame = frame.copy()
            if face is not None and left_eye is not None and right_eye is not None:
                # Draw face rectangle
                x, y, w, h = face
                cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Draw eye rectangles
                ex, ey, ew, eh = left_eye
                cv2.rectangle(vis_frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
                ex, ey, ew, eh = right_eye
                cv2.rectangle(vis_frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

                # Draw gaze direction indicators
                face_center_x = x + w // 2
                face_center_y = y + h // 2

                # Convert pitch and yaw to arrow
                arrow_length = 100
                # Yaw: left-right, Pitch: up-down
                end_x = int(face_center_x + arrow_length * np.sin(np.radians(yaw)))
                end_y = int(face_center_y - arrow_length * np.sin(np.radians(pitch)))
                cv2.arrowedLine(vis_frame, (face_center_x, face_center_y), (end_x, end_y), (0, 0, 255), 3, tipLength=0.3)

            # Add text overlay for gaze angles
            cv2.putText(vis_frame, f"Pitch: {pitch:.1f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Yaw: {yaw:.1f} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Convert RGB back to BGR for video writing
            vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
            video_writer.write(vis_frame_bgr)

        if (frame_idx + 1) % 100 == 0:
            batch_time = (time.time() - batch_start_time) * 1000
            print(f"  Processed {frame_idx + 1}/{frame_count} frames ({batch_time:.2f}ms)")
            batch_start_time = time.time()

        frame_idx += 1

    cap.release()

    # Release video writer if used
    if video_writer is not None:
        video_writer.release()
        print(f"  [SUCCESS] Video output saved to: {video_out_path}")

    # Print inference time statistics
    if inference_times:
        avg_time = np.mean(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        print(f"  [SUCCESS] Processed all {frame_count} frames")
        print(f"  Inference time - Avg: {avg_time*1000:.2f}ms, Min: {min_time*1000:.2f}ms, Max: {max_time*1000:.2f}ms")
        print(f"  Average FPS: {fps:.2f}\n")
    else:
        print(f"  [SUCCESS] Processed all {frame_count} frames\n")


def process_single_file(video_path: Path, output_path: Path,
                       gaze_estimator,
                       model_type: str = 'simple',
                       should_rotate: bool = False,
                       check_mode: bool = False,
                       video_out_path: Optional[Path] = None) -> None:
    """
    Process a single video or image file and estimate gaze.
    Images/frames are converted from BGR to RGB before processing.

    Args:
        video_path: Path to video or image file
        output_path: Output CSV file path
        gaze_estimator: GazeEstimator instance (or None for simplified method)
        model_type: Model type ('mpiigaze', 'mpiifacegaze', 'eth-xgaze', or 'simple')
        should_rotate: Whether to rotate frames
        check_mode: Whether to save first frame visualization
        video_out_path: Optional path to save visualization video (AVI format)
    """
    print(f"\n{'=' * 80}")
    print(f"Processing single file: {video_path.name}")
    print(f"Output: {output_path}")
    print(f"Model: {model_type}")
    print(f"{'=' * 80}\n")

    # Check if file is image or video
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    video_extensions = ['.avi', '.mp4', '.mov', '.mkv']
    file_ext = video_path.suffix.lower()

    is_image = file_ext in image_extensions
    is_video = file_ext in video_extensions

    if not is_image and not is_video:
        raise ValueError(f"Unsupported file type: {file_ext}. Must be image or video.")

    # Generate timestamps
    if is_image:
        print("Detected: Image file")
        frame_count = 1
        timestamps = [0.0]
        fps = 30.0
    else:
        # Generate timestamps from video FPS
        print("Detected: Video file")
        print("Generating timestamps from video FPS...")
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30.0
        cap.release()
        timestamps = [i / fps for i in range(frame_count)]
        print(f"  Frame count: {frame_count}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Duration: {frame_count / fps:.2f} seconds\n")

    # Prepare check output path
    check_output_path = None
    if check_mode:
        check_output_path = output_path.parent / f"check_gaze_{output_path.stem.replace('gaze_', '')}.jpg"

    # Create CSV file and write header
    print(f"Creating output file: {output_path}")
    header = generate_csv_header()

    with open(output_path, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)
        csv_file.flush()
        print(f"  [SUCCESS] CSV file created with header ({len(header)} columns)\n")

        # Process image or video
        if is_image:
            # Process single image
            print(f"Processing image...")
            img = cv2.imread(str(video_path))
            if img is None:
                raise ValueError(f"Cannot read image: {video_path}")

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if should_rotate:
                print(f"  [INFO] Image will be rotated 90 degrees clockwise")
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

            # For simplified method, detect face and eyes first
            face, left_eye, right_eye = None, None, None
            if model_type == 'simple':
                face, left_eye, right_eye = detect_face_and_eyes(img)

            # Estimate gaze direction with timing
            start_time = time.time()
            pitch, yaw = estimate_gaze_direction(
                img, gaze_estimator, model_type,
                face=face, left_eye=left_eye, right_eye=right_eye
            )
            inference_time = time.time() - start_time

            # Save visualization if check mode
            if check_mode and check_output_path is not None:
                # For visualization, we need face and eyes
                if face is None or left_eye is None or right_eye is None:
                    face, left_eye, right_eye = detect_face_and_eyes(img)

                if face is not None and left_eye is not None and right_eye is not None:
                    visualize_gaze(img, face, left_eye, right_eye, pitch, yaw, check_output_path)
                else:
                    print(f"  [WARNING] Cannot visualize: face or eyes not detected")

            # Write to CSV
            row = [timestamps[0]]
            row.extend([pitch, yaw])
            csv_writer.writerow(row)
            csv_file.flush()

            print(f"  [SUCCESS] Processed image")
            print(f"    Pitch: {pitch:.2f} deg, Yaw: {yaw:.2f} deg")
            print(f"    Inference time: {inference_time*1000:.2f}ms\n")

        else:
            # Process video
            print(f"Processing video...")
            if should_rotate:
                print(f"  [INFO] Video will be rotated 90 degrees clockwise")

            cap = cv2.VideoCapture(str(video_path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Initialize VideoWriter if video output is requested
            video_writer = None
            if video_out_path is not None:
                fps = cap.get(cv2.CAP_PROP_FPS)
                # Get frame dimensions
                ret, first_frame = cap.read()
                if ret:
                    if should_rotate:
                        first_frame = cv2.rotate(first_frame, cv2.ROTATE_90_CLOCKWISE)
                    frame_height, frame_width = first_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (frame_width, frame_height))
                    print(f"  [INFO] Creating output video: {video_out_path.name} ({frame_width}x{frame_height} @ {fps:.2f} fps)")
                    # Reset capture for frame reading
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Inference time tracking
            inference_times = []

            frame_idx = 0
            batch_start_time = time.time()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if should_rotate:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                # For simplified method, detect face and eyes first
                face, left_eye, right_eye = None, None, None
                if model_type == 'simple':
                    face, left_eye, right_eye = detect_face_and_eyes(frame)

                # Estimate gaze direction with timing
                start_time = time.time()
                pitch, yaw = estimate_gaze_direction(
                    frame, gaze_estimator, model_type,
                    face=face, left_eye=left_eye, right_eye=right_eye
                )
                inference_time = time.time() - start_time
                inference_times.append(inference_time)

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

                # Write visualization frame to video if requested
                if video_writer is not None:
                    # For visualization, we need face and eyes
                    if face is None or left_eye is None or right_eye is None:
                        face, left_eye, right_eye = detect_face_and_eyes(frame)

                    # Create visualization frame
                    vis_frame = frame.copy()
                    if face is not None and left_eye is not None and right_eye is not None:
                        # Draw face rectangle
                        x, y, w, h = face
                        cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Draw eye rectangles
                        ex, ey, ew, eh = left_eye
                        cv2.rectangle(vis_frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
                        ex, ey, ew, eh = right_eye
                        cv2.rectangle(vis_frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

                        # Draw gaze direction indicators
                        face_center_x = x + w // 2
                        face_center_y = y + h // 2

                        # Convert pitch and yaw to arrow
                        arrow_length = 100
                        # Yaw: left-right, Pitch: up-down
                        end_x = int(face_center_x + arrow_length * np.sin(np.radians(yaw)))
                        end_y = int(face_center_y - arrow_length * np.sin(np.radians(pitch)))
                        cv2.arrowedLine(vis_frame, (face_center_x, face_center_y), (end_x, end_y), (0, 0, 255), 3, tipLength=0.3)

                    # Add text overlay for gaze angles
                    cv2.putText(vis_frame, f"Pitch: {pitch:.1f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(vis_frame, f"Yaw: {yaw:.1f} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Convert RGB back to BGR for video writing
                    vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
                    video_writer.write(vis_frame_bgr)

                if (frame_idx + 1) % 100 == 0:
                    batch_time = (time.time() - batch_start_time) * 1000
                    print(f"  Processed {frame_idx + 1}/{frame_count} frames ({batch_time:.2f}ms)")
                    batch_start_time = time.time()

                frame_idx += 1

            cap.release()

            # Release video writer if used
            if video_writer is not None:
                video_writer.release()
                print(f"  [SUCCESS] Video output saved to: {video_out_path}")

            # Print inference time statistics
            if inference_times:
                avg_time = np.mean(inference_times)
                min_time = np.min(inference_times)
                max_time = np.max(inference_times)
                fps = 1.0 / avg_time if avg_time > 0 else 0
                print(f"  [SUCCESS] Processed all {frame_count} frames")
                print(f"  Inference time - Avg: {avg_time*1000:.2f}ms, Min: {min_time*1000:.2f}ms, Max: {max_time*1000:.2f}ms")
                print(f"  Average FPS: {fps:.2f}\n")
            else:
                print(f"  [SUCCESS] Processed all {frame_count} frames\n")

    print(f"[SUCCESS] Output saved to: {output_path}")


def generate_csv_header() -> List[str]:
    """
    Generate CSV header for gaze data.

    Returns:
        List of column names
    """
    return ['timestamp', 'pitch', 'yaw']


def create_gaze_estimator_config(model_type: str, device: str):
    """
    Create ptgaze config for GazeEstimator.

    Args:
        model_type: Model type ('mpiigaze', 'mpiifacegaze', 'eth-xgaze')
        device: Device to use ('cpu' or 'cuda')

    Returns:
        OmegaConf DictConfig object
    """
    if not PTGAZE_AVAILABLE:
        return None

    # Get ptgaze package root
    import ptgaze
    package_root = Path(ptgaze.__file__).parent.resolve()

    # Map model types to config files and mode names
    config_map = {
        'mpiigaze': ('mpiigaze.yaml', 'MPIIGaze'),
        'mpiifacegaze': ('mpiifacegaze.yaml', 'MPIIFaceGaze'),
        'eth-xgaze': ('eth-xgaze.yaml', 'ETH-XGaze')
    }

    if model_type not in config_map:
        raise ValueError(f"Unknown model type: {model_type}")

    config_file, mode_name = config_map[model_type]
    config_path = package_root / 'data' / 'configs' / config_file

    # Load config
    config = OmegaConf.load(config_path)
    config.PACKAGE_ROOT = package_root.as_posix()

    # Set device
    config.device = device
    if config.device == 'cuda' and not torch.cuda.is_available():
        config.device = 'cpu'
        warnings.warn('Run on CPU because CUDA is not available.')

    # Use dummy camera params (we'll handle camera separately)
    config.gaze_estimator.use_dummy_camera_params = True

    # Expand user paths
    expanduser_all(config)

    # Generate dummy camera params (required for initialization)
    # We'll use a temporary dummy file
    out_file = tempfile.NamedTemporaryFile(suffix='.yaml', delete=False, mode='w')
    dummy_params = {
        'image_width': 640,
        'image_height': 480,
        'camera_matrix': {
            'rows': 3,
            'cols': 3,
            'data': [640, 0., 320, 0., 640, 240, 0., 0., 1.]
        },
        'distortion_coefficients': {
            'rows': 1,
            'cols': 5,
            'data': [0., 0., 0., 0., 0.]
        }
    }
    yaml.safe_dump(dummy_params, out_file)
    out_file.close()
    config.gaze_estimator.camera_params = out_file.name

    # Download required models
    if config.face_detector.mode == 'dlib':
        download_dlib_pretrained_model()

    if mode_name == 'MPIIGaze':
        download_mpiigaze_model()
    elif mode_name == 'MPIIFaceGaze':
        download_mpiifacegaze_model()
    elif mode_name == 'ETH-XGaze':
        download_ethxgaze_model()

    return config


def main():
    parser = argparse.ArgumentParser(
        description='Generate gaze estimation data from video files using MPIIGaze models'
    )
    parser.add_argument(
        '--path',
        required=True,
        help='Directory or video file path (directory for batch mode, file for single mode)'
    )
    parser.add_argument(
        '--no-batch',
        action='store_true',
        help='Single file mode: process a single video/image file'
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
        nargs='*',
        default=None,
        help='IDs to rotate 90 degrees clockwise (batch mode) or use --rotate for single file'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Save first frame with gaze visualization as JPG'
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
    print(f"Gaze Estimation Generator")
    print(f"Mode: {'Single File' if args.no_batch else 'Batch'}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
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
                # Create config and initialize GazeEstimator
                config = create_gaze_estimator_config(args.model, args.device)
                gaze_estimator = GazeEstimator(config)
                print(f"  [SUCCESS] {args.model} model initialized on {args.device}\n")
            except Exception as e:
                print(f"  [ERROR] Failed to initialize {args.model}: {e}")
                print("  Falling back to simplified method...\n")
                import traceback
                traceback.print_exc()
                args.model = 'simple'
                gaze_estimator = None

    # Single file mode
    if args.no_batch:
        video_path = Path(args.path)
        if not video_path.exists():
            print(f"Error: File does not exist: {args.path}")
            return 1

        if not video_path.is_file():
            print(f"Error: Path is not a file: {args.path}")
            return 1

        # Determine output path
        output_dir = video_path.parent
        output_filename = f"gaze_{video_path.stem}.csv"
        output_path = output_dir / output_filename

        # Check if rotation is requested (--rotate flag present means rotate)
        should_rotate = args.rotate is not None

        # Determine video output path if requested
        video_out_path = None
        if args.video_out:
            file_id = extract_id_from_filename(video_path.name)
            if file_id is not None:
                video_out_path = output_dir / f"gaze_{file_id}_visualization.avi"
            else:
                video_out_path = output_dir / f"gaze_{video_path.stem}_visualization.avi"

        try:
            process_single_file(
                video_path=video_path,
                output_path=output_path,
                gaze_estimator=gaze_estimator,
                model_type=args.model,
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

    # Find file pairs in camera directory
    pairs = find_file_pairs(input_directory, selected_ids=selected_ids)

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

        # Prepare video output path if requested
        video_out_path = None
        if args.video_out:
            video_out_path = output_directory / f"gaze_{file_id}_visualization.avi"

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
                check_output_path=check_output_path,
                video_out_path=video_out_path
            )

        print(f"  [SUCCESS] Output saved to: {output_path}\n")

    print(f"{'=' * 80}")
    print(f"[SUCCESS] All files processed")
    print(f"Total pairs processed: {len(pairs)}")
    print(f"{'=' * 80}")

    return 0


if __name__ == '__main__':
    exit(main())
