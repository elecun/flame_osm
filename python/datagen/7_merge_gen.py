import argparse
import os
import pandas as pd
from pathlib import Path


def load_timestamp_file(timestamp_file_path):
    """
    Load timestamp file (no header)
    """
    if not os.path.exists(timestamp_file_path):
        raise FileNotFoundError(f"Timestamp file not found: {timestamp_file_path}")

    # Read CSV without header
    df = pd.read_csv(timestamp_file_path, header=None, names=['timestamp'])
    print(f"Loaded timestamp file: {len(df)} rows")
    return df


def load_csv_file(file_path, file_description):
    """
    Load CSV file with header
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_description} file not found: {file_path}")

    df = pd.read_csv(file_path)
    print(f"Loaded {file_description}: {len(df)} rows")
    return df


def check_row_counts(dataframes, file_names):
    """
    Check if all dataframes have the same number of rows
    Returns True if all have same row count, False otherwise
    """
    row_counts = [len(df) for df in dataframes]

    print("\nChecking row counts:")
    for name, count in zip(file_names, row_counts):
        print(f"  {name}: {count} rows")

    if len(set(row_counts)) == 1:
        print("\nAll files have the same number of rows. Proceeding with merge.")
        return True
    else:
        print("\nError: Files have different number of rows!")
        print("Row count details:")
        for name, count in zip(file_names, row_counts):
            print(f"  {name}: {count} rows")
        return False


def merge_dataframes(timestamp_df, body_kps_df, face_kps_df, head_pose_df, eeg_performance_df):
    """
    Merge all dataframes by concatenating columns
    Assumes all dataframes have the same number of rows
    """
    # Prepare dataframes by excluding timestamp columns (except from timestamp_df)
    body_kps_clean = body_kps_df[[col for col in body_kps_df.columns if col.lower() != 'timestamp']]
    face_kps_clean = face_kps_df[[col for col in face_kps_df.columns if col.lower() != 'timestamp']]
    head_pose_clean = head_pose_df[[col for col in head_pose_df.columns if col.lower() != 'timestamp']]
    eeg_performance_clean = eeg_performance_df[[col for col in eeg_performance_df.columns if col.lower() != 'timestamp']]

    # Concatenate all dataframes at once for better performance
    merged_df = pd.concat([
        timestamp_df,
        body_kps_clean,
        face_kps_clean,
        head_pose_clean,
        eeg_performance_clean
    ], axis=1)

    return merged_df


def main():
    parser = argparse.ArgumentParser(description='Merge multiple CSV files into one')
    parser.add_argument('--path', type=str, required=True,
                        help='Working directory path')
    parser.add_argument('--time-reference', type=int, required=True,
                        help='Reference number (e.g., 0 for timestamp_0.csv)')

    args = parser.parse_args()

    # Construct paths
    working_dir = Path(args.path)
    ref = args.time_reference

    timestamp_file = working_dir / 'camera' / f'timestamp_{ref}.csv'
    body_kps_file = working_dir / f'body_kps_{ref}.csv'
    face_kps_file = working_dir / f'face_kps_{ref}.csv'
    head_pose_file = working_dir / f'head_pose_{ref}.csv'
    eeg_performance_file = working_dir / 'eeg_performance.csv'
    output_file = working_dir / f'merge_{ref}.csv'

    print(f"Working directory: {working_dir}")
    print(f"Reference number: {ref}")
    print()

    # Load all files
    print("Step 1: Loading files...")
    print()

    try:
        timestamp_df = load_timestamp_file(timestamp_file)
        body_kps_df = load_csv_file(body_kps_file, f"body_kps_{ref}.csv")
        face_kps_df = load_csv_file(face_kps_file, f"face_kps_{ref}.csv")
        head_pose_df = load_csv_file(head_pose_file, f"head_pose_{ref}.csv")
        eeg_performance_df = load_csv_file(eeg_performance_file, "eeg_performance.csv")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return

    # Check if all files have the same number of rows
    print()
    print("Step 2: Validating row counts...")

    dataframes = [timestamp_df, body_kps_df, face_kps_df, head_pose_df, eeg_performance_df]
    file_names = [
        f'timestamp_{ref}.csv',
        f'body_kps_{ref}.csv',
        f'face_kps_{ref}.csv',
        f'head_pose_{ref}.csv',
        'eeg_performance.csv'
    ]

    if not check_row_counts(dataframes, file_names):
        print("\nMerge aborted due to row count mismatch.")
        return

    # Merge dataframes
    print()
    print("Step 3: Merging dataframes...")
    merged_df = merge_dataframes(timestamp_df, body_kps_df, face_kps_df, head_pose_df, eeg_performance_df)
    print(f"Merged dataframe has {len(merged_df)} rows and {len(merged_df.columns)} columns")
    print()

    # Save to CSV
    print(f"Step 4: Saving to {output_file}...")
    merged_df.to_csv(output_file, index=False)
    print(f"Successfully saved merged data to {output_file}")
    print()

    # Print summary
    print("Summary:")
    print(f"  Total rows: {len(merged_df)}")
    print(f"  Total columns: {len(merged_df.columns)}")
    print(f"  Column names: {list(merged_df.columns[:10])}..." if len(merged_df.columns) > 10 else f"  Column names: {list(merged_df.columns)}")
    print(f"  Output file: {output_file}")


if __name__ == '__main__':
    main()
