#!/usr/bin/env python3
"""
CSV Merger
Merges multiple CSV files based on timestamp from the first file.
Handles different sampling rates using averaging and linear interpolation.
"""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple
import numpy as np


def read_csv_file(csv_path: Path) -> Tuple[List[str], np.ndarray, List[float]]:
    """
    Read CSV file and return headers, data, and timestamps.

    Args:
        csv_path: Path to CSV file

    Returns:
        Tuple of (headers, data_array, timestamps)
    """
    print(f"Reading: {csv_path.name}")

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty CSV file: {csv_path}")

    # First row is header
    headers = rows[0]
    data_rows = rows[1:]

    if not data_rows:
        raise ValueError(f"No data rows in CSV file: {csv_path}")

    # Convert to numpy array
    data = np.array([[float(val) for val in row] for row in data_rows])

    # Extract timestamps (first column)
    timestamps = data[:, 0].tolist()

    print(f"  Loaded {len(data)} rows, {len(headers)} columns")
    print(f"  Timestamp range: {timestamps[0]:.6f} to {timestamps[-1]:.6f}")

    return headers, data, timestamps


def find_closest_index(target_time: float, timestamps: List[float]) -> int:
    """
    Find index of closest timestamp to target time.

    Args:
        target_time: Target timestamp
        timestamps: List of timestamps

    Returns:
        Index of closest timestamp
    """
    min_diff = float('inf')
    closest_idx = 0

    for idx, ts in enumerate(timestamps):
        diff = abs(ts - target_time)
        if diff < min_diff:
            min_diff = diff
            closest_idx = idx

    return closest_idx


def interpolate_value(t: float, t1: float, t2: float, v1: float, v2: float) -> float:
    """
    Linear interpolation between two points.

    Args:
        t: Target time
        t1: Time of first point
        t2: Time of second point
        v1: Value at first point
        v2: Value at second point

    Returns:
        Interpolated value at time t
    """
    if t2 == t1:
        return v1

    # Linear interpolation formula
    return v1 + (v2 - v1) * (t - t1) / (t2 - t1)


def resample_data(reference_timestamps: List[float],
                  source_timestamps: List[float],
                  source_data: np.ndarray,
                  start_col: int = 1) -> np.ndarray:
    """
    Resample source data to match reference timestamps.
    Uses averaging when multiple source points fall in one interval.
    Uses linear interpolation when no source points are available.

    Args:
        reference_timestamps: Target timestamps to resample to
        source_timestamps: Source timestamps
        source_data: Source data array (including timestamp column)
        start_col: Column index to start resampling (default 1, skip timestamp)

    Returns:
        Resampled data array (without timestamp column)
    """
    print(f"  Resampling from {len(source_timestamps)} to {len(reference_timestamps)} samples")

    num_ref = len(reference_timestamps)
    num_cols = source_data.shape[1] - start_col  # Exclude timestamp column
    resampled = np.zeros((num_ref, num_cols))

    # Find starting point in source data (closest to first reference timestamp)
    start_idx = find_closest_index(reference_timestamps[0], source_timestamps)
    ref_start_time = reference_timestamps[0]
    src_start_time = source_timestamps[start_idx]
    print(f"  Reference start time: {ref_start_time:.6f}")
    print(f"  Source start time: {src_start_time:.6f} (index {start_idx})")
    print(f"  Time offset: {abs(src_start_time - ref_start_time):.6f} seconds")

    # Use only source data from start_idx onwards
    valid_source_timestamps = source_timestamps[start_idx:]
    valid_source_data = source_data[start_idx:, :]

    if len(valid_source_timestamps) == 0:
        print(f"  [WARNING] No valid source data after alignment")
        return resampled

    print(f"  Using {len(valid_source_timestamps)} source samples (from index {start_idx})")

    # Process each reference timestamp
    for ref_idx in range(num_ref):
        ref_time = reference_timestamps[ref_idx]

        # Calculate time interval for this reference point
        # This interval determines which source points belong to this reference point
        if ref_idx < num_ref - 1:
            next_ref_time = reference_timestamps[ref_idx + 1]
            time_delta = next_ref_time - ref_time
            interval_start = ref_time - time_delta / 2.0
            interval_end = ref_time + time_delta / 2.0
        else:
            # For last point, use same interval as previous
            if ref_idx > 0:
                prev_time_delta = ref_time - reference_timestamps[ref_idx - 1]
                interval_start = ref_time - prev_time_delta / 2.0
                interval_end = ref_time + prev_time_delta / 2.0
            else:
                # Single point case
                interval_start = ref_time - 0.001
                interval_end = ref_time + 0.001

        # Find all source points within this interval
        points_in_interval = []
        for src_idx, src_time in enumerate(valid_source_timestamps):
            if interval_start <= src_time <= interval_end:
                points_in_interval.append(src_idx)

        # Process each column
        for col_idx in range(num_cols):
            data_col = start_col + col_idx

            if len(points_in_interval) > 0:
                # Multiple or single point(s) in interval - use average
                values = [valid_source_data[idx, data_col] for idx in points_in_interval]
                resampled[ref_idx, col_idx] = np.mean(values)

            else:
                # No points in interval - use linear interpolation
                # Find nearest points before and after ref_time
                before_idx = None
                after_idx = None

                for src_idx in range(len(valid_source_timestamps)):
                    if valid_source_timestamps[src_idx] <= ref_time:
                        before_idx = src_idx
                    if valid_source_timestamps[src_idx] >= ref_time and after_idx is None:
                        after_idx = src_idx
                        break

                if before_idx is not None and after_idx is not None and before_idx != after_idx:
                    # Both points available - linear interpolation
                    t1 = valid_source_timestamps[before_idx]
                    t2 = valid_source_timestamps[after_idx]
                    v1 = valid_source_data[before_idx, data_col]
                    v2 = valid_source_data[after_idx, data_col]
                    resampled[ref_idx, col_idx] = interpolate_value(ref_time, t1, t2, v1, v2)

                elif before_idx is not None:
                    # Only before point available - use its value
                    resampled[ref_idx, col_idx] = valid_source_data[before_idx, data_col]

                elif after_idx is not None:
                    # Only after point available - use its value
                    resampled[ref_idx, col_idx] = valid_source_data[after_idx, data_col]

                else:
                    # No data available at all - use 0
                    resampled[ref_idx, col_idx] = 0.0

    return resampled


def merge_csv_files(csv_paths: List[Path], output_path: Path):
    """
    Merge multiple CSV files based on the first file's timestamps.

    Args:
        csv_paths: List of CSV file paths (first one is the reference)
        output_path: Output CSV file path
    """
    print(f"\n{'=' * 80}")
    print(f"CSV Merger")
    print(f"Input files: {len(csv_paths)}")
    print(f"Output: {output_path}")
    print(f"{'=' * 80}\n")

    if len(csv_paths) < 2:
        raise ValueError("At least 2 CSV files are required for merging")

    # Read the reference file (first file)
    print(f"[1/{len(csv_paths)}] Loading reference file...")
    ref_headers, ref_data, ref_timestamps = read_csv_file(csv_paths[0])
    print()

    # Start building merged data with reference data
    merged_data = ref_data.copy()
    merged_headers = ref_headers.copy()

    # Process remaining files
    for idx, csv_path in enumerate(csv_paths[1:], start=2):
        print(f"[{idx}/{len(csv_paths)}] Processing file for merge...")
        headers, data, timestamps = read_csv_file(csv_path)

        # Resample data to match reference timestamps
        resampled = resample_data(ref_timestamps, timestamps, data)

        # Add resampled columns to merged data (skip timestamp column from source)
        merged_data = np.concatenate([merged_data, resampled], axis=1)

        # Add headers (skip timestamp column from source)
        merged_headers.extend(headers[1:])
        print()

    # Write merged CSV
    print(f"{'=' * 80}")
    print(f"Writing merged CSV...")
    print(f"  Total columns: {len(merged_headers)}")
    print(f"  Total rows: {len(merged_data)}")

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(merged_headers)

        for row in merged_data:
            writer.writerow(row)

    print(f"[SUCCESS] Merged CSV saved to: {output_path}")
    print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(
        description='Merge multiple CSV files based on timestamp alignment'
    )
    parser.add_argument(
        'files',
        nargs='+',
        help='CSV files to merge (first file is the reference for timestamps)'
    )
    parser.add_argument(
        '--output',
        default='merged.csv',
        help='Output CSV filename (default: merged.csv)'
    )

    args = parser.parse_args()

    # Convert to Path objects
    csv_paths = [Path(f) for f in args.files]

    # Validate files
    for csv_path in csv_paths:
        if not csv_path.exists():
            print(f"Error: File does not exist: {csv_path}")
            return 1
        if not csv_path.is_file():
            print(f"Error: Not a file: {csv_path}")
            return 1

    output_path = Path(args.output)

    try:
        merge_csv_files(csv_paths, output_path)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
