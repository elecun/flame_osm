#!/usr/bin/env python3
"""
Eye Tracker Data Fixation and Saccade Mapper

This script maps camera timestamps to eye tracker fixation IDs and saccade amplitudes.
For each camera timestamp, it checks if the timestamp falls within any fixation or saccade period
and records the corresponding fixation ID and/or saccade amplitude.
"""

import argparse
import csv
import sys
from pathlib import Path
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
import warnings


@dataclass
class Fixation:
    """Fixation data from eye tracker"""
    fixation_id: int
    start_timestamp_ns: int
    end_timestamp_ns: int
    duration_ms: float
    fixation_x: float
    fixation_y: float


@dataclass
class Saccade:
    """Saccade data from eye tracker"""
    saccade_id: int
    start_timestamp_ns: int
    end_timestamp_ns: int
    duration_ms: float
    amplitude_deg: float


def load_time_reference(base_path: Path, time_reference_id: int) -> np.ndarray:
    """
    Load time reference from camera/timestamp_{id}.csv

    Args:
        base_path: Base directory path
        time_reference_id: ID for the timestamp file (e.g., 0 for timestamp_0.csv)

    Returns:
        Array of timestamps in seconds
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

    return np.array(timestamps)


def find_eyetracker_data_dir(base_path: Path) -> Path:
    """
    Find the eyetracker data directory under <path>/eyetracker.

    Args:
        base_path: Base directory path

    Returns:
        Path to the eyetracker data directory
    """
    data_dir = base_path / "eyetracker"

    if not data_dir.exists():
        raise FileNotFoundError(f"Eyetracker directory not found: {data_dir}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Eyetracker path is not a directory: {data_dir}")

    return data_dir


def find_first_existing_file(data_dir: Path, filenames) -> Path:
    """Return the first existing file from filenames inside data_dir."""
    for name in filenames:
        candidate = data_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"None of the files found in {data_dir}: {', '.join(filenames)}")


def load_fixations(data_dir: Path) -> List[Fixation]:
    """
    Load fixation data from fixations.csv

    Args:
        data_dir: Directory containing eyetracker CSV files

    Returns:
        List of Fixation objects
    """
    fixations_file = find_first_existing_file(data_dir, ["fixations.csv", "fixation.csv"])

    fixations = []

    with open(fixations_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fixation_id = int(row['fixation id'])
                start_timestamp_ns = int(row['start timestamp [ns]'])
                end_timestamp_ns = int(row['end timestamp [ns]'])
                duration_ms = float(row['duration [ms]'])
                fixation_x = float(row['fixation x [px]'])
                fixation_y = float(row['fixation y [px]'])

                fixations.append(Fixation(
                    fixation_id=fixation_id,
                    start_timestamp_ns=start_timestamp_ns,
                    end_timestamp_ns=end_timestamp_ns,
                    duration_ms=duration_ms,
                    fixation_x=fixation_x,
                    fixation_y=fixation_y
                ))
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping invalid fixation row: {e}")
                continue

    return fixations


def load_saccades(data_dir: Path) -> List[Saccade]:
    """
    Load saccade data from saccades.csv

    Args:
        data_dir: Directory containing eyetracker CSV files

    Returns:
        List of Saccade objects
    """
    saccades_file = find_first_existing_file(data_dir, ["saccades.csv", "saccade.csv"])

    saccades = []

    with open(saccades_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                saccade_id = int(row['saccade id'])
                start_timestamp_ns = int(row['start timestamp [ns]'])
                end_timestamp_ns = int(row['end timestamp [ns]'])
                duration_ms = float(row['duration [ms]'])
                amplitude_deg = float(row['amplitude [deg]'])

                saccades.append(Saccade(
                    saccade_id=saccade_id,
                    start_timestamp_ns=start_timestamp_ns,
                    end_timestamp_ns=end_timestamp_ns,
                    duration_ms=duration_ms,
                    amplitude_deg=amplitude_deg
                ))
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping invalid saccade row: {e}")
                continue

    return saccades


def classify_attention_level(k_coefficient: float, k1: float, k2: float, num_classes: int = 5) -> int:
    """
    Classify K coefficient into attention level

    Args:
        k_coefficient: K coefficient value (EWMA)
        k1: First threshold (neutral boundary)
        k2: Second threshold (strong attention boundary, only used for 5-class)
        num_classes: Number of classes (3 or 5, default: 5)

    Returns:
        For 5-class classification:
            1 = Strong Ambient Attention (k <= -k2)
            2 = Weak Ambient Attention (-k2 < k <= -k1)
            3 = Neutral (-k1 < k < +k1)
            4 = Weak Focal Attention (+k1 <= k < +k2)
            5 = Strong Focal Attention (k >= +k2)

        For 3-class classification (uses only k1, returns 1, 3, 5):
            1 = Strong Ambient Attention (k <= -k1)
            3 = Neutral (-k1 < k < +k1)
            5 = Strong Focal Attention (k >= +k1)
    """
    if num_classes == 3:
        # 3-class classification using only k1, returns 1, 3, 5
        if k_coefficient >= k1:
            return 5  # Strong Focal Attention
        elif k_coefficient > -k1:
            return 3  # Neutral
        else:
            return 1  # Strong Ambient Attention
    else:
        # 5-class classification using k1 and k2
        if k_coefficient >= k2:
            return 5  # Strong Focal Attention (>= +k2)
        elif k_coefficient >= k1:
            return 4  # Weak Focal Attention (+k1 ~ +k2)
        elif k_coefficient >= -k1:
            return 3  # Neutral (-k1 ~ +k1)
        elif k_coefficient >= -k2:
            return 2  # Weak Ambient Attention (-k2 ~ -k1)
        else:
            return 1  # Strong Ambient Attention (<= -k2)


def compute_transition_counts(results: List[Dict], window_sec: float = 60.0) -> List[int]:
    """
    Compute number of ambient<->focal transitions in the past window for each timestamp.
    Ambient: attention_level < 3
    Focal:   attention_level > 3
    Neutral (==3) does not count as a side; transitions are counted when sign changes across 3.
    """
    if window_sec <= 0:
        window_sec = 60.0

    event_times = []
    prev_state = None

    for res in results:
        level = res.get('attention_level', 3) or 3
        state = 0
        if level > 3:
            state = 1
        elif level < 3:
            state = -1

        if prev_state is not None and prev_state != 0 and state != 0 and prev_state != state:
            event_times.append(res['timestamp'])

        if state != 0:
            prev_state = state

    event_times = np.array(event_times, dtype=float)
    counts = [0] * len(results)

    left = 0
    right = 0
    n_events = len(event_times)

    for i, res in enumerate(results):
        t = res['timestamp']
        while left < n_events and event_times[left] <= t - window_sec:
            left += 1
        while right < n_events and event_times[right] <= t:
            right += 1
        counts[i] = right - left

    return counts


def visualize_attention_transition_frequency(
    results: List[Dict],
    output_path: Path,
    window_length_sec: float = 1.0,
    step_sec: float = 0.3
) -> None:
    """
    Visualize attention transition frequency (rising/falling around baseline 3) using FFT.
    Counts transitions per second using a sliding time window (default: 1s) sampled every step_sec (default: 0.3s).
    Saves a frequency vs magnitude plot to output_path.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not installed; cannot generate frequency visualization.")
        return

    if len(results) < 2:
        warnings.warn("Not enough data to compute transition frequency.")
        return

    window_length_sec = float(window_length_sec)
    step_sec = float(step_sec)
    if window_length_sec <= 0 or step_sec <= 0:
        warnings.warn("Window length and step must be positive for frequency visualization.")
        return

    timestamps = np.array([r['timestamp'] for r in results], dtype=float)
    levels = np.array([r.get('attention_level', 3) or 3 for r in results], dtype=float)

    duration = timestamps[-1] - timestamps[0]
    if duration <= 0:
        warnings.warn("Invalid timestamp range; cannot compute frequency.")
        return

    # Sample rate estimated from timestamps
    sample_rate = (len(timestamps) - 1) / duration

    # Build transition events: +1 for rising across 3, -1 for falling across 3
    transitions = np.zeros_like(levels)
    event_times = []
    for i in range(1, len(levels)):
        prev_level = levels[i - 1]
        curr_level = levels[i]
        if prev_level <= 3 < curr_level:
            transitions[i] = 1
            event_times.append(timestamps[i])
        elif prev_level >= 3 > curr_level:
            transitions[i] = -1
            event_times.append(timestamps[i])

    if len(event_times) == 0:
        warnings.warn("No transitions detected; skipping frequency visualization.")
        return

    event_times = np.array(event_times, dtype=float)

    # Build rate signal (transitions per second) using a sliding window
    start_t = timestamps[0]
    end_t = timestamps[-1]
    if end_t <= start_t:
        warnings.warn("Invalid timestamp range; cannot compute frequency.")
        return

    # Grid represents window end times; start from first full window
    time_grid = np.arange(start_t + window_length_sec, end_t + step_sec, step_sec)
    rates = np.zeros_like(time_grid)

    left = 0
    right = 0
    n_events = len(event_times)
    for idx, t in enumerate(time_grid):
        # Advance left/right pointers to maintain window (t - window_length_sec, t]
        while left < n_events and event_times[left] <= t - window_length_sec:
            left += 1
        while right < n_events and event_times[right] <= t:
            right += 1
        count = right - left
        rates[idx] = count / window_length_sec  # transitions per second over the window

    # Remove DC component
    signal = rates - np.mean(rates)

    # FFT on rate signal (uniform sampling at 1/step_sec Hz)
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=step_sec)
    magnitudes = np.abs(fft_vals)

    # Skip the zero-frequency component for visualization clarity
    freqs = freqs[1:]
    magnitudes = magnitudes[1:]

    plt.figure(figsize=(8, 4))
    plt.plot(freqs, magnitudes, label='Transition magnitude')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(f'Attention Transition Frequency (window={window_length_sec:.1f}s, step={step_sec:.1f}s, baseline=3)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved attention transition frequency visualization to {output_path}")


def apply_minimum_dwell_time(results: List[Dict], min_dwell_sec: float) -> None:
    """
    Apply minimum dwell time filtering for all attention level transitions.

    This prevents rapid transitions between attention levels by requiring that each
    attention level must persist for at least min_dwell_sec seconds. If a segment
    is shorter than min_dwell_sec, the attention level is reverted to the previous state.

    This applies to:
    - Transitions to focal attention (4, 5) from non-focal states (1, 2, 3)
    - Transitions between focal states (4 ↔ 5)
    - Any other attention level changes

    Args:
        results: List of result dictionaries with 'timestamp' and 'attention_level'
        min_dwell_sec: Minimum dwell time in seconds
    """
    if len(results) == 0:
        return

    # Create a copy of attention levels to modify
    attention_levels = [r['attention_level'] for r in results]
    timestamps = [r['timestamp'] for r in results]

    # Iteratively filter short segments until no more changes occur
    max_iterations = 10  # Prevent infinite loops
    for iteration in range(max_iterations):
        changed = False
        i = 0

        while i < len(attention_levels):
            current_level = attention_levels[i]

            # Get previous level (skip if at the beginning)
            if i == 0:
                i += 1
                continue

            prev_level = attention_levels[i - 1]

            # Check if there's a transition (level changed)
            if current_level != prev_level:
                # Find the end of this segment (same level)
                segment_start = i
                segment_end = i

                # Extend segment while attention level is the same
                while segment_end < len(attention_levels) and attention_levels[segment_end] == current_level:
                    segment_end += 1

                # Calculate segment duration
                if segment_end < len(timestamps):
                    segment_duration = timestamps[segment_end - 1] - timestamps[segment_start]
                else:
                    # Last segment goes to the end
                    segment_duration = timestamps[-1] - timestamps[segment_start]

                # If segment is shorter than min_dwell, revert to previous level
                if segment_duration < min_dwell_sec:
                    for j in range(segment_start, segment_end):
                        attention_levels[j] = prev_level
                    changed = True

                # Move to the end of this segment
                i = segment_end
            else:
                i += 1

        # If no changes were made, we're done
        if not changed:
            break

    # Update results with filtered attention levels
    for i, result in enumerate(results):
        result['attention_level'] = attention_levels[i]


def match_timestamps_to_fixations_and_saccades(
    time_references: np.ndarray,
    fixations: List[Fixation],
    saccades: List[Saccade],
    window_size: int = 10,
    k1: float = 0.5,
    k2: float = 1.5,
    min_dwell_sec: float = None,
    num_classes: int = 5
) -> List[Dict]:
    """
    Match camera timestamps to fixation IDs and saccade data, and calculate K coefficient
    using three different methods: Rolling Z-score, EWMA Z-score, and MAD.

    For each camera timestamp, find if it falls within any fixation or saccade period
    and record the fixation ID and/or saccade ID and amplitude.

    Calculate K coefficient using three methods:
    1. Rolling Z-score: Standard z-score using mean and std from rolling window
    2. EWMA Z-score: Z-score using exponentially weighted moving average and std
    3. MAD: Median Absolute Deviation based score

    Classify attention level based on EWMA K coefficient:
    - For 5-class: 1=Strong Ambient, 2=Weak Ambient, 3=Neutral, 4=Weak Focal, 5=Strong Focal
    - For 3-class: 1=Strong Ambient, 3=Neutral, 5=Strong Focal

    If min_dwell_sec is specified, applies minimum dwell time filtering to prevent rapid
    transitions between attention levels. Attention segments shorter than min_dwell_sec
    are reverted to the previous attention state.

    Args:
        time_references: Array of camera timestamps in seconds
        fixations: List of Fixation objects
        saccades: List of Saccade objects
        window_size: Number of past timestamps to consider for K calculation
        k1: First threshold for attention level classification (neutral boundary, default: 0.5)
        k2: Second threshold for attention level classification (strong boundary, default: 1.5)
        min_dwell_sec: Minimum dwell time in seconds for attention transitions (default: None)
        num_classes: Number of attention classes, 3 or 5 (default: 5)

    Returns:
        List of dictionaries with timestamp, fixation_id, saccade_id, saccade_amplitude_deg,
        k_coefficient_rolling, k_coefficient_ewma, k_coefficient_mad, attention_level
    """
    # Create fixation and saccade lookup dictionaries
    fixation_dict = {f.fixation_id: f for f in fixations}

    results = []

    for i, timestamp_sec in enumerate(time_references):
        # Convert timestamp from seconds to nanoseconds
        timestamp_ns = int(timestamp_sec * 1e9)

        # Find matching fixation
        matched_fixation_id = None
        for fixation in fixations:
            if fixation.start_timestamp_ns <= timestamp_ns <= fixation.end_timestamp_ns:
                matched_fixation_id = fixation.fixation_id
                break

        # Find matching saccade
        matched_saccade_id = None
        matched_saccade_amplitude = None
        for saccade in saccades:
            if saccade.start_timestamp_ns <= timestamp_ns <= saccade.end_timestamp_ns:
                matched_saccade_id = saccade.saccade_id
                matched_saccade_amplitude = saccade.amplitude_deg
                break

        results.append({
            'timestamp': timestamp_sec,
            'fixation_id': matched_fixation_id if matched_fixation_id is not None else '',
            'saccade_id': matched_saccade_id if matched_saccade_id is not None else '',
            'saccade_amplitude_deg': matched_saccade_amplitude if matched_saccade_amplitude is not None else ''
        })

    # Calculate fixation duration and saccade amplitude for each window
    fixation_durations_mean = []
    fixation_durations_std = []
    saccade_amplitudes_mean = []
    saccade_amplitudes_std = []

    for i in range(len(results)):
        # Get window indices (current timestamp and past window_size-1 timestamps)
        window_start = max(0, i - window_size + 1)
        window_end = i + 1
        window_results = results[window_start:window_end]

        if len(window_results) < window_size:
            # Not enough data, set to 0
            fixation_durations_mean.append(0.0)
            fixation_durations_std.append(0.0)
            saccade_amplitudes_mean.append(0.0)
            saccade_amplitudes_std.append(0.0)
        else:
            # Calculate fixation duration: total duration divided by number of segments
            unique_fixation_ids = set()
            for r in window_results:
                if r['fixation_id'] != '':
                    unique_fixation_ids.add(r['fixation_id'])

            if unique_fixation_ids:
                # Sum all fixation durations and divide by the number of segments
                total_duration = sum(fixation_dict[fid].duration_ms / 1000.0 for fid in unique_fixation_ids)
                num_segments = len(unique_fixation_ids)
                fixation_duration = total_duration / num_segments

                # Store the calculated fixation duration (total/segments)
                fixation_durations_mean.append(fixation_duration)

                # Calculate std deviation based on individual fixation durations
                fixation_durs = [fixation_dict[fid].duration_ms / 1000.0 for fid in unique_fixation_ids]
                fixation_durations_std.append(np.std(fixation_durs) if len(fixation_durs) > 1 else 0.0)
            else:
                fixation_durations_mean.append(0.0)
                fixation_durations_std.append(0.0)

            # Calculate saccade amplitude: mean and std of amplitudes in window
            saccade_amps = [
                r['saccade_amplitude_deg']
                for r in window_results
                if r['saccade_amplitude_deg'] != ''
            ]
            if saccade_amps:
                saccade_amplitudes_mean.append(np.mean(saccade_amps))
                # Calculate std deviation of saccade amplitudes
                saccade_amplitudes_std.append(np.std(saccade_amps) if len(saccade_amps) > 1 else 0.0)
            else:
                saccade_amplitudes_mean.append(0.0)
                saccade_amplitudes_std.append(0.0)

    # Convert to numpy arrays for calculations
    fixation_durations_arr = np.array(fixation_durations_mean)
    saccade_amplitudes_arr = np.array(saccade_amplitudes_mean)

    # Initialize K coefficient arrays for three methods
    k_coefficients_rolling = np.zeros(len(results))
    k_coefficients_ewma = np.zeros(len(results))
    k_coefficients_mad = np.zeros(len(results))

    # ============================================================
    # Method 1: Rolling Z-score
    # ============================================================
    fixation_zscores_rolling = np.zeros(len(results))
    saccade_zscores_rolling = np.zeros(len(results))

    for i in range(len(results)):
        if fixation_durations_arr[i] > 0:
            # Get non-zero values in the current window
            non_zero_fix = fixation_durations_arr[max(0, i - window_size + 1):i + 1]
            non_zero_fix = non_zero_fix[non_zero_fix > 0]

            non_zero_sac = saccade_amplitudes_arr[max(0, i - window_size + 1):i + 1]
            non_zero_sac = non_zero_sac[non_zero_sac > 0]

            if len(non_zero_fix) > 1:
                fix_mean = np.mean(non_zero_fix)
                fix_std = np.std(non_zero_fix)
                if fix_std > 0:
                    fixation_zscores_rolling[i] = (fixation_durations_arr[i] - fix_mean) / fix_std

            if len(non_zero_sac) > 1:
                sac_mean = np.mean(non_zero_sac)
                sac_std = np.std(non_zero_sac)
                if sac_std > 0:
                    saccade_zscores_rolling[i] = (saccade_amplitudes_arr[i] - sac_mean) / sac_std

    k_coefficients_rolling = fixation_zscores_rolling - saccade_zscores_rolling

    # ============================================================
    # Method 2: EWMA Z-score (Exponentially Weighted Moving Average)
    # ============================================================
    # Using span parameter for decay (span = 2/(alpha+1) - 1, so alpha = 2/(span+1))
    span = min(window_size, 10)  # Use smaller span for more responsiveness
    alpha = 2.0 / (span + 1)

    fixation_zscores_ewma = np.zeros(len(results))
    saccade_zscores_ewma = np.zeros(len(results))

    # Initialize EWMA values
    ewma_fix_mean = 0.0
    ewma_fix_var = 0.0
    ewma_sac_mean = 0.0
    ewma_sac_var = 0.0
    initialized_fix = False
    initialized_sac = False

    for i in range(len(results)):
        current_fix = fixation_durations_arr[i]
        current_sac = saccade_amplitudes_arr[i]

        # Update EWMA for fixation
        if current_fix > 0:
            if not initialized_fix:
                ewma_fix_mean = current_fix
                ewma_fix_var = 0.0
                initialized_fix = True
            else:
                delta = current_fix - ewma_fix_mean
                ewma_fix_mean = alpha * current_fix + (1 - alpha) * ewma_fix_mean
                ewma_fix_var = (1 - alpha) * (ewma_fix_var + alpha * delta * delta)

                ewma_fix_std = np.sqrt(ewma_fix_var)
                if ewma_fix_std > 0:
                    fixation_zscores_ewma[i] = (current_fix - ewma_fix_mean) / ewma_fix_std

        # Update EWMA for saccade
        if current_sac > 0:
            if not initialized_sac:
                ewma_sac_mean = current_sac
                ewma_sac_var = 0.0
                initialized_sac = True
            else:
                delta = current_sac - ewma_sac_mean
                ewma_sac_mean = alpha * current_sac + (1 - alpha) * ewma_sac_mean
                ewma_sac_var = (1 - alpha) * (ewma_sac_var + alpha * delta * delta)

                ewma_sac_std = np.sqrt(ewma_sac_var)
                if ewma_sac_std > 0:
                    saccade_zscores_ewma[i] = (current_sac - ewma_sac_mean) / ewma_sac_std

    k_coefficients_ewma = fixation_zscores_ewma - saccade_zscores_ewma

    # ============================================================
    # Apply additional EWMA smoothing for attention level classification
    # ============================================================
    alpha_k = 0.02  # Much slower smoothing for stable attention classification
    k_smooth = np.zeros(len(results))
    k_smooth_value = None

    for t in range(len(k_coefficients_ewma)):
        k_raw = k_coefficients_ewma[t]
        if k_smooth_value is None:
            k_smooth_value = k_raw
        else:
            k_smooth_value = alpha_k * k_raw + (1 - alpha_k) * k_smooth_value
        k_smooth[t] = k_smooth_value

    # ============================================================
    # Method 3: MAD (Median Absolute Deviation)
    # ============================================================
    fixation_zscores_mad = np.zeros(len(results))
    saccade_zscores_mad = np.zeros(len(results))

    # MAD constant for normal distribution
    MAD_CONSTANT = 1.4826

    for i in range(len(results)):
        if fixation_durations_arr[i] > 0:
            # Get non-zero values in the current window
            non_zero_fix = fixation_durations_arr[max(0, i - window_size + 1):i + 1]
            non_zero_fix = non_zero_fix[non_zero_fix > 0]

            non_zero_sac = saccade_amplitudes_arr[max(0, i - window_size + 1):i + 1]
            non_zero_sac = non_zero_sac[non_zero_sac > 0]

            if len(non_zero_fix) > 1:
                fix_median = np.median(non_zero_fix)
                fix_mad = np.median(np.abs(non_zero_fix - fix_median))
                if fix_mad > 0:
                    fixation_zscores_mad[i] = (fixation_durations_arr[i] - fix_median) / (fix_mad * MAD_CONSTANT)

            if len(non_zero_sac) > 1:
                sac_median = np.median(non_zero_sac)
                sac_mad = np.median(np.abs(non_zero_sac - sac_median))
                if sac_mad > 0:
                    saccade_zscores_mad[i] = (saccade_amplitudes_arr[i] - sac_median) / (sac_mad * MAD_CONSTANT)

    k_coefficients_mad = fixation_zscores_mad - saccade_zscores_mad

    # ============================================================
    # Add calculated values to results
    # ============================================================
    for i, result in enumerate(results):
        # Add K coefficients for all three methods
        # K coefficients are based on window data, so they can exist even without current fixation_id
        if fixation_durations_arr[i] > 0:
            result['k_coefficient_rolling'] = k_coefficients_rolling[i]
            result['k_coefficient_ewma'] = k_coefficients_ewma[i]
            result['k_coefficient_mad'] = k_coefficients_mad[i]

            # Classify attention level based on smoothed K coefficient (k_smooth)
            result['attention_level'] = classify_attention_level(k_smooth[i], k1, k2, num_classes)
        else:
            result['k_coefficient_rolling'] = ''
            result['k_coefficient_ewma'] = ''
            result['k_coefficient_mad'] = ''
            # Fill missing attention level with Neutral (always 3)
            result['attention_level'] = 3

    # Apply minimum dwell time filtering if specified
    if min_dwell_sec is not None and min_dwell_sec > 0:
        apply_minimum_dwell_time(results, min_dwell_sec)

    # Compute transition counts (ambient<->focal) over past 60s
    transition_counts = compute_transition_counts(results, window_sec=60.0)
    for i, res in enumerate(results):
        res['transitions_last60s'] = transition_counts[i]

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Map camera timestamps to eye tracker fixation IDs and saccade amplitudes'
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
    parser.add_argument(
        '--output',
        type=str,
        default='eyetrack_attention.csv',
        help='Output CSV filename (default: eyetrack_attention.csv)'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=10,
        help='Number of past timestamps to consider for K coefficient calculation (default: 10)'
    )
    parser.add_argument(
        '--k1',
        type=float,
        default=0.5,
        help='First threshold for attention level classification (weak attention boundary, default: 0.5)'
    )
    parser.add_argument(
        '--k2',
        type=float,
        default=1.5,
        help='Second threshold for attention level classification (strong attention boundary, default: 1.5)'
    )
    parser.add_argument(
        '--min-dwell',
        type=float,
        default=None,
        help='Minimum dwell time in seconds for focal attention transitions (default: None, no filtering)'
    )
    parser.add_argument(
        '--class',
        type=int,
        default=5,
        choices=[3, 5],
        dest='num_classes',
        help='Number of attention classes: 3 (levels 1,3,5: Strong Ambient/Neutral/Strong Focal) or 5 (levels 1,2,3,4,5: Strong Ambient/Weak Ambient/Neutral/Weak Focal/Strong Focal, default: 5)'
    )
    parser.add_argument(
        '--freq-vis',
        action='store_true',
        help='Generate FFT-based visualization of attention transitions (baseline=3) and save as image'
    )

    args = parser.parse_args()

    base_path = Path(args.path)

    if not base_path.exists():
        print(f"Error: Base path does not exist: {base_path}", file=sys.stderr)
        sys.exit(1)

    try:
        # Load time reference
        print(f"Loading time reference from timestamp_{args.time_reference}.csv...")
        time_references = load_time_reference(base_path, args.time_reference)
        print(f"Loaded {len(time_references)} timestamps from camera/timestamp_{args.time_reference}.csv")
        print(f"Time range: {time_references.min():.3f}s - {time_references.max():.3f}s")

        # Find eyetracker data directory
        print(f"Finding eyetracker data directory...")
        data_dir = find_eyetracker_data_dir(base_path)
        print(f"Found eyetracker data in: {data_dir}")

        # Load fixation data
        print(f"Loading fixation data...")
        fixations = load_fixations(data_dir)
        print(f"Loaded {len(fixations)} fixation segments")

        if len(fixations) == 0:
            print("Warning: No fixations found in fixations.csv")
            sys.exit(1)

        # Load saccade data
        print(f"Loading saccade data...")
        saccades = load_saccades(data_dir)
        print(f"Loaded {len(saccades)} saccade segments")

        if len(saccades) == 0:
            print("Warning: No saccades found in saccades.csv")
            sys.exit(1)

        # Match timestamps to fixations and saccades, and calculate K coefficient
        print(f"\nMatching camera timestamps to fixations and saccades...")
        print(f"Calculating K coefficients with window size: {args.window_size}")
        print(f"Using three methods: Rolling Z-score, EWMA Z-score, MAD")
        print(f"Number of attention classes: {args.num_classes}")
        if args.num_classes == 3:
            print(f"Attention level thresholds: k1={args.k1} (3-class: levels 1,3,5)")
            print(f"  1=Strong Ambient, 3=Neutral, 5=Strong Focal")
        else:
            print(f"Attention level thresholds: k1={args.k1}, k2={args.k2} (5-class)")
            print(f"  1=Strong Ambient, 2=Weak Ambient, 3=Neutral, 4=Weak Focal, 5=Strong Focal")
        if getattr(args, 'min_dwell', None) is not None:
            print(f"Applying minimum dwell time filter: {args.min_dwell} seconds")
        results = match_timestamps_to_fixations_and_saccades(
            time_references, fixations, saccades, args.window_size, args.k1, args.k2,
            min_dwell_sec=getattr(args, 'min_dwell', None),
            num_classes=args.num_classes
        )

        # Count matches
        fixation_matched_count = sum(1 for r in results if r['fixation_id'] != '')
        saccade_matched_count = sum(1 for r in results if r['saccade_amplitude_deg'] != '')
        k_calculated_count = sum(1 for r in results if r['k_coefficient_rolling'] != '')
        print(f"Matched {fixation_matched_count}/{len(results)} timestamps to fixations")
        print(f"Matched {saccade_matched_count}/{len(results)} timestamps to saccades")
        print(f"Calculated {k_calculated_count}/{len(results)} K coefficients")

        # Save results to CSV (append class info to filename)
        class_suffix = f"class_{args.num_classes}"
        output_name = args.output
        output_path = base_path / (
            f"{Path(output_name).stem}_{class_suffix}{Path(output_name).suffix}"
            if Path(output_name).suffix
            else f"{output_name}_{class_suffix}"
        )
        print(f"\nSaving results to {output_path}...")

        with open(output_path, 'w', newline='') as f:
            fieldnames = ['timestamp', 'fixation_id', 'saccade_id', 'saccade_amplitude_deg',
                         'k_coefficient_rolling', 'k_coefficient_ewma', 'k_coefficient_mad',
                         'attention_level', 'transitions_last60s']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for result in results:
                writer.writerow(result)

        print(f"Successfully saved results to {output_path}")

        # Optional frequency visualization
        if args.freq_vis:
            vis_path = output_path.with_name(f"{output_path.stem}_freq.png")
            visualize_attention_transition_frequency(results, vis_path, window_length_sec=1.0, step_sec=0.3)

        # Print summary statistics
        k_coeffs_rolling = [r['k_coefficient_rolling'] for r in results if r['k_coefficient_rolling'] != '']
        k_coeffs_ewma = [r['k_coefficient_ewma'] for r in results if r['k_coefficient_ewma'] != '']
        k_coeffs_mad = [r['k_coefficient_mad'] for r in results if r['k_coefficient_mad'] != '']
        attention_levels = [r['attention_level'] for r in results if r['attention_level'] != '']

        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        print(f"Total timestamps:              {len(results)}")
        print(f"Fixation matched timestamps:   {fixation_matched_count} ({fixation_matched_count/len(results)*100:.1f}%)")
        print(f"Saccade matched timestamps:    {saccade_matched_count} ({saccade_matched_count/len(results)*100:.1f}%)")
        print(f"K coefficients calculated:     {k_calculated_count} ({k_calculated_count/len(results)*100:.1f}%)")
        print(f"\nTotal fixations:               {len(fixations)}")
        print(f"Mean fixation duration:        {np.mean([f.duration_ms for f in fixations]):.2f}ms")
        print(f"\nTotal saccades:                {len(saccades)}")
        print(f"Mean saccade amplitude:        {np.mean([s.amplitude_deg for s in saccades]):.2f}°")

        # Attention level distribution
        if attention_levels:
            print(f"\nAttention Level Distribution (k1={args.k1}, k2={args.k2}):")
            level_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            for level in attention_levels:
                level_counts[level] = level_counts.get(level, 0) + 1

            total_with_level = len(attention_levels)
            print(f"  Level 1 (Strong Ambient):    {level_counts[1]:4d} ({level_counts[1]/total_with_level*100:5.1f}%)")
            print(f"  Level 2 (Weak Ambient):      {level_counts[2]:4d} ({level_counts[2]/total_with_level*100:5.1f}%)")
            print(f"  Level 3 (Neutral):           {level_counts[3]:4d} ({level_counts[3]/total_with_level*100:5.1f}%)")
            print(f"  Level 4 (Weak Focal):        {level_counts[4]:4d} ({level_counts[4]/total_with_level*100:5.1f}%)")
            print(f"  Level 5 (Strong Focal):      {level_counts[5]:4d} ({level_counts[5]/total_with_level*100:5.1f}%)")

        if k_coeffs_rolling:
            print(f"\nK coefficient statistics (Rolling Z-score):")
            print(f"  Mean:                        {np.mean(k_coeffs_rolling):.4f}")
            print(f"  Std:                         {np.std(k_coeffs_rolling):.4f}")
            print(f"  Min:                         {np.min(k_coeffs_rolling):.4f}")
            print(f"  Max:                         {np.max(k_coeffs_rolling):.4f}")

        if k_coeffs_ewma:
            print(f"\nK coefficient statistics (EWMA Z-score):")
            print(f"  Mean:                        {np.mean(k_coeffs_ewma):.4f}")
            print(f"  Std:                         {np.std(k_coeffs_ewma):.4f}")
            print(f"  Min:                         {np.min(k_coeffs_ewma):.4f}")
            print(f"  Max:                         {np.max(k_coeffs_ewma):.4f}")

        if k_coeffs_mad:
            print(f"\nK coefficient statistics (MAD):")
            print(f"  Mean:                        {np.mean(k_coeffs_mad):.4f}")
            print(f"  Std:                         {np.std(k_coeffs_mad):.4f}")
            print(f"  Min:                         {np.min(k_coeffs_mad):.4f}")
            print(f"  Max:                         {np.max(k_coeffs_mad):.4f}")
        print("="*60)

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
