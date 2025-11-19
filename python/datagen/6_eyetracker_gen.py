#!/usr/bin/env python3
"""
Eye Tracker Data K Coefficient Generator

This script processes eye tracker data and calculates K coefficients based on gaze stability.
"""

import argparse
import csv
import sys
from pathlib import Path
import numpy as np
from typing import Tuple


def load_time_reference(base_path: Path, time_reference_id: int) -> Tuple[float, float]:
    """
    Load time reference from camera/timestamp_{id}.csv

    Args:
        base_path: Base directory path
        time_reference_id: ID for the timestamp file (e.g., 0 for timestamp_0.csv)

    Returns:
        Tuple of (start_time_sec, end_time_sec)
    """
    timestamp_file = base_path / "camera" / f"timestamp_{time_reference_id}.csv"

    if not timestamp_file.exists():
        raise FileNotFoundError(f"Time reference file not found: {timestamp_file}")

    timestamps = []
    with open(timestamp_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  # Skip empty rows
                timestamps.append(float(row[0]))

    if not timestamps:
        raise ValueError(f"No timestamps found in timestamp_{time_reference_id}.csv")

    # Return start and end times in seconds
    return min(timestamps), max(timestamps)


def find_eyetracker_data_dir(base_path: Path) -> Path:
    """
    Find the eyetracker data directory under eyetracker/Timeseries Data/

    Args:
        base_path: Base directory path

    Returns:
        Path to the eyetracker data directory
    """
    timeseries_path = base_path / "eyetracker" / "Timeseries Data"

    if not timeseries_path.exists():
        raise FileNotFoundError(f"Timeseries Data directory not found: {timeseries_path}")

    # Find the single subdirectory
    subdirs = [d for d in timeseries_path.iterdir() if d.is_dir()]

    if len(subdirs) == 0:
        raise FileNotFoundError(f"No subdirectory found in: {timeseries_path}")
    elif len(subdirs) > 1:
        print(f"Warning: Multiple subdirectories found, using the first one: {subdirs[0].name}")

    return subdirs[0]


def load_gaze_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load gaze data from gaze.csv

    Args:
        data_dir: Directory containing eyetracker CSV files

    Returns:
        Tuple of (timestamps_ns, gaze_x, gaze_y)
    """
    gaze_file = data_dir / "gaze.csv"

    if not gaze_file.exists():
        raise FileNotFoundError(f"Gaze data file not found: {gaze_file}")

    timestamps = []
    gaze_x = []
    gaze_y = []

    with open(gaze_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                timestamps.append(int(row['timestamp [ns]']))
                gaze_x.append(float(row['gaze x [px]']))
                gaze_y.append(float(row['gaze y [px]']))
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping invalid row: {e}")
                continue

    return np.array(timestamps), np.array(gaze_x), np.array(gaze_y)


def filter_data_by_time_range(timestamps_ns: np.ndarray,
                               gaze_x: np.ndarray,
                               gaze_y: np.ndarray,
                               start_time_sec: float,
                               end_time_sec: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter gaze data to only include samples within the time reference range

    Args:
        timestamps_ns: Timestamps in nanoseconds
        gaze_x: Gaze X coordinates
        gaze_y: Gaze Y coordinates
        start_time_sec: Start time in seconds
        end_time_sec: End time in seconds

    Returns:
        Filtered (timestamps_ns, gaze_x, gaze_y)
    """
    # Convert nanoseconds to seconds for comparison
    timestamps_sec = timestamps_ns / 1e9

    # Create mask for timestamps within range
    mask = (timestamps_sec >= start_time_sec) & (timestamps_sec <= end_time_sec)

    return timestamps_ns[mask], gaze_x[mask], gaze_y[mask]


def calculate_k_coefficient(gaze_x: np.ndarray, gaze_y: np.ndarray) -> dict:
    """
    Calculate K coefficient and related metrics for gaze stability

    K coefficient measures the stability of gaze points:
    - Lower K coefficient indicates more stable gaze (fixation)
    - Higher K coefficient indicates more scattered gaze (saccades/pursuit)

    Args:
        gaze_x: Array of gaze X coordinates in pixels
        gaze_y: Array of gaze Y coordinates in pixels

    Returns:
        Dictionary containing K coefficient and related metrics
    """
    if len(gaze_x) == 0 or len(gaze_y) == 0:
        return {
            'k_coefficient': None,
            'std_x': None,
            'std_y': None,
            'mean_x': None,
            'mean_y': None,
            'dispersion': None,
            'n_samples': 0
        }

    # Calculate basic statistics
    mean_x = np.mean(gaze_x)
    mean_y = np.mean(gaze_y)
    std_x = np.std(gaze_x)
    std_y = np.std(gaze_y)

    # Calculate dispersion (spread of gaze points)
    dispersion = np.sqrt(std_x**2 + std_y**2)

    # K coefficient is typically defined as the normalized dispersion
    # We use the coefficient of variation (CV) as the K coefficient
    # CV = std / mean, but for 2D gaze we use dispersion / distance from origin
    distance_from_origin = np.sqrt(mean_x**2 + mean_y**2)

    if distance_from_origin > 0:
        k_coefficient = dispersion / distance_from_origin
    else:
        # If mean is at origin, use absolute dispersion
        k_coefficient = dispersion

    # Alternative K coefficient based on median absolute deviation
    median_x = np.median(gaze_x)
    median_y = np.median(gaze_y)
    mad_x = np.median(np.abs(gaze_x - median_x))
    mad_y = np.median(np.abs(gaze_y - median_y))
    mad_dispersion = np.sqrt(mad_x**2 + mad_y**2)

    return {
        'k_coefficient': k_coefficient,
        'k_coefficient_mad': mad_dispersion,
        'std_x': std_x,
        'std_y': std_y,
        'mean_x': mean_x,
        'mean_y': mean_y,
        'median_x': median_x,
        'median_y': median_y,
        'dispersion': dispersion,
        'mad_dispersion': mad_dispersion,
        'n_samples': len(gaze_x)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Calculate K coefficient from eye tracker data'
    )
    parser.add_argument(
        '--path',
        type=str,
        required=True,
        help='Base directory containing eyetracker data'
    )
    parser.add_argument(
        '--time-reference',
        type=int,
        default=0,
        help='Time reference ID: 0 = use camera/timestamp_0.csv, 2 = use camera/timestamp_2.csv, etc.'
    )

    args = parser.parse_args()

    base_path = Path(args.path)

    if not base_path.exists():
        print(f"Error: Base path does not exist: {base_path}", file=sys.stderr)
        sys.exit(1)

    try:
        # Load time reference
        print(f"Loading time reference from timestamp_{args.time_reference}.csv...")
        start_time_sec, end_time_sec = load_time_reference(base_path, args.time_reference)
        print(f"Time reference from camera/timestamp_{args.time_reference}.csv: {start_time_sec:.3f}s - {end_time_sec:.3f}s")

        # Find eyetracker data directory
        print(f"Finding eyetracker data directory...")
        data_dir = find_eyetracker_data_dir(base_path)
        print(f"Found eyetracker data in: {data_dir}")

        # Load gaze data
        print(f"Loading gaze data...")
        timestamps_ns, gaze_x, gaze_y = load_gaze_data(data_dir)
        print(f"Loaded {len(timestamps_ns)} gaze samples")

        # Filter data by time range
        print(f"Filtering data by time range...")
        timestamps_filtered, gaze_x_filtered, gaze_y_filtered = filter_data_by_time_range(
            timestamps_ns, gaze_x, gaze_y, start_time_sec, end_time_sec
        )
        print(f"Filtered to {len(timestamps_filtered)} samples within time range")

        # Calculate K coefficient
        print(f"\nCalculating K coefficient...")
        results = calculate_k_coefficient(gaze_x_filtered, gaze_y_filtered)

        # Print results
        print("\n" + "="*60)
        print("K COEFFICIENT RESULTS")
        print("="*60)
        print(f"Number of samples:        {results['n_samples']}")
        print(f"Mean gaze position:       ({results['mean_x']:.2f}, {results['mean_y']:.2f}) px")
        print(f"Median gaze position:     ({results['median_x']:.2f}, {results['median_y']:.2f}) px")
        print(f"Std deviation (X, Y):     ({results['std_x']:.2f}, {results['std_y']:.2f}) px")
        print(f"Dispersion:               {results['dispersion']:.2f} px")
        print(f"MAD Dispersion:           {results['mad_dispersion']:.2f} px")
        print(f"\nK Coefficient (CV):       {results['k_coefficient']:.4f}")
        print(f"K Coefficient (MAD):      {results['k_coefficient_mad']:.4f}")
        print("="*60)

        # Interpretation
        print("\nInterpretation:")
        if results['k_coefficient'] is not None:
            if results['k_coefficient'] < 0.1:
                print("  - Very stable gaze (likely fixation)")
            elif results['k_coefficient'] < 0.3:
                print("  - Moderately stable gaze")
            else:
                print("  - Unstable gaze (likely saccades or pursuit)")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
