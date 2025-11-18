import argparse
import os
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d


def find_eeg_file(eeg_csv_path):
    """
    Find CSV file containing 'md.mc.pm.bp' in the filename
    """
    eeg_csv_dir = Path(eeg_csv_path)
    if not eeg_csv_dir.exists():
        raise FileNotFoundError(f"EEG CSV directory not found: {eeg_csv_path}")

    matching_files = list(eeg_csv_dir.glob("*md.mc.pm.bp*.csv"))
    if not matching_files:
        raise FileNotFoundError(f"No file containing 'md.mc.pm.bp' found in {eeg_csv_path}")

    if len(matching_files) > 1:
        print(f"Warning: Multiple matching files found. Using: {matching_files[0]}")

    return matching_files[0]


def load_eeg_data(eeg_file_path):
    """
    Load EEG data from CSV file
    - Skip first row (metadata)
    - Second row is header
    - Extract 'Timestamp' and 'PM.Attention.Scaled' and 'PM.CognitiveStress.Scaled' columns
    """
    # Try different encodings
    encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1', 'iso-8859-1']
    df = None

    for encoding in encodings:
        try:
            df = pd.read_csv(eeg_file_path, skiprows=[0], encoding=encoding)
            print(f"Successfully loaded file with {encoding} encoding")
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

    if df is None:
        raise ValueError(f"Could not decode file with any of these encodings: {encodings}")

    # Check if required columns exist
    required_cols = ['Timestamp', 'PM.Attention.Scaled', 'PM.CognitiveStress.Scaled']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns not found: {missing_cols}. Available columns: {df.columns.tolist()}")

    # Extract required columns
    eeg_data = df[['Timestamp', 'PM.Attention.Scaled', 'PM.CognitiveStress.Scaled']].copy()

    # Convert timestamp to numeric (assuming it's in milliseconds or similar format)
    eeg_data['Timestamp'] = pd.to_numeric(eeg_data['Timestamp'], errors='coerce')
    eeg_data['PM.Attention.Scaled'] = pd.to_numeric(eeg_data['PM.Attention.Scaled'], errors='coerce')
    eeg_data['PM.CognitiveStress.Scaled'] = pd.to_numeric(eeg_data['PM.CognitiveStress.Scaled'], errors='coerce')

    # Drop rows with NaN values
    eeg_data = eeg_data.dropna()

    # Sort by timestamp
    eeg_data = eeg_data.sort_values('Timestamp').reset_index(drop=True)

    print(f"Loaded {len(eeg_data)} EEG data points")
    print(f"EEG timestamp range: {eeg_data['Timestamp'].min()} - {eeg_data['Timestamp'].max()}")

    return eeg_data


def load_reference_timestamps(timestamp_file_path):
    """
    Load reference timestamps from camera timestamp file (no header)
    """
    if not os.path.exists(timestamp_file_path):
        raise FileNotFoundError(f"Reference timestamp file not found: {timestamp_file_path}")

    # Read CSV without header
    df = pd.read_csv(timestamp_file_path, header=None)

    # Assume first column contains timestamps
    timestamps = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values

    print(f"Loaded {len(timestamps)} reference timestamps")
    print(f"Reference timestamp range: {timestamps.min()} - {timestamps.max()}")

    return timestamps


def interpolate_linear(eeg_data, reference_timestamps):
    """
    Linearly interpolate EEG attention and cognitive stress values to match reference timestamps
    """
    eeg_timestamps = eeg_data['Timestamp'].values
    eeg_attention = eeg_data['PM.Attention.Scaled'].values
    eeg_stress = eeg_data['PM.CognitiveStress.Scaled'].values

    # Create interpolation function for attention
    interp_func_attention = interp1d(
        eeg_timestamps,
        eeg_attention,
        kind='linear',
        bounds_error=False,
        fill_value=(eeg_attention[0], eeg_attention[-1])  # Use boundary values for extrapolation
    )

    # Create interpolation function for cognitive stress
    interp_func_stress = interp1d(
        eeg_timestamps,
        eeg_stress,
        kind='linear',
        bounds_error=False,
        fill_value=(eeg_stress[0], eeg_stress[-1])  # Use boundary values for extrapolation
    )

    # Interpolate values for reference timestamps
    interpolated_attention = interp_func_attention(reference_timestamps)
    interpolated_stress = interp_func_stress(reference_timestamps)

    return interpolated_attention, interpolated_stress


def main():
    parser = argparse.ArgumentParser(description='Generate EEG attention data aligned with reference timestamps')
    parser.add_argument('--path', type=str, required=True,
                        help='Root data directory path')
    parser.add_argument('--time-reference', type=int, required=True,
                        help='Time reference number (e.g., 0 for timestamp_0.csv)')
    parser.add_argument('--method', type=str, default='linear', choices=['linear'],
                        help='Interpolation method (currently only linear is supported)')

    args = parser.parse_args()

    # Construct paths
    data_root = Path(args.path)
    eeg_csv_path = data_root / 'eeg' / 'csv'
    timestamp_file = data_root / 'camera' / f'timestamp_{args.time_reference}.csv'
    output_file = data_root / 'eeg_performance.csv'

    print(f"Data root: {data_root}")
    print(f"EEG CSV path: {eeg_csv_path}")
    print(f"Reference timestamp file: {timestamp_file}")
    print(f"Output file: {output_file}")
    print()

    # Find and load EEG data
    print("Step 1: Finding EEG file...")
    eeg_file = find_eeg_file(eeg_csv_path)
    print(f"Found EEG file: {eeg_file}")
    print()

    print("Step 2: Loading EEG data...")
    eeg_data = load_eeg_data(eeg_file)
    print()

    print("Step 3: Loading reference timestamps...")
    reference_timestamps = load_reference_timestamps(timestamp_file)
    print()

    print("Step 4: Interpolating attention and cognitive stress values...")
    if args.method == 'linear':
        interpolated_attention, interpolated_stress = interpolate_linear(eeg_data, reference_timestamps)
    else:
        raise ValueError(f"Unsupported interpolation method: {args.method}")

    print(f"Interpolated {len(interpolated_attention)} attention and stress values")
    print()

    # Create output dataframe
    output_df = pd.DataFrame({
        'timestamp': reference_timestamps,
        'attention': interpolated_attention,
        'cognitive_stress': interpolated_stress
    })

    # Save to CSV
    print(f"Step 5: Saving to {output_file}...")
    output_df.to_csv(output_file, index=False)
    print(f"Successfully saved {len(output_df)} rows to {output_file}")
    print()

    # Print summary statistics
    print("Summary:")
    print(f"  Timestamp range: {output_df['timestamp'].min()} - {output_df['timestamp'].max()}")
    print(f"  Attention range: {output_df['attention'].min():.2f} - {output_df['attention'].max():.2f}")
    print(f"  Attention mean: {output_df['attention'].mean():.2f}")
    print(f"  Attention std: {output_df['attention'].std():.2f}")
    print(f"  Cognitive Stress range: {output_df['cognitive_stress'].min():.2f} - {output_df['cognitive_stress'].max():.2f}")
    print(f"  Cognitive Stress mean: {output_df['cognitive_stress'].mean():.2f}")
    print(f"  Cognitive Stress std: {output_df['cognitive_stress'].std():.2f}")


if __name__ == '__main__':
    main()
