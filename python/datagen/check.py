#!/usr/bin/env python3
import argparse
import csv
import os
import sys

RED = "\033[91m"
RESET = "\033[0m"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CSV checker: scan all CSV files in a directory (non-recursive) "
        "and report row/column counts plus empty rows."
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Directory containing CSV files to analyze (non-recursive).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories for CSV files.",
    )
    return parser.parse_args()


def analyze_csv(file_path: str) -> tuple[int, int, int, int]:
    """
    Returns a tuple of (row_count, column_count, empty_row_count, nan_row_count).
    - row_count: total rows encountered (including empty rows)
    - column_count: number of columns based on the first non-empty row (0 if none)
    - empty_row_count: rows where all fields are empty/whitespace
    - nan_row_count: rows where any field is a NaN marker
    """
    row_count = 0
    empty_row_count = 0
    nan_row_count = 0
    column_count = 0

    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            row_count += 1
            # Detect NUL bytes in raw cells
            if any('\x00' in cell for cell in row):
                raise csv.Error(f"line contains NUL at row {row_count}")
            # Treat a row as empty if every cell is empty/whitespace.
            if all(cell.strip() == "" for cell in row):
                empty_row_count += 1
                continue
            if any(cell.strip().lower() in {"nan", "na", "null"} for cell in row):
                nan_row_count += 1
            if column_count == 0:
                column_count = len(row)

    return row_count, column_count, empty_row_count, nan_row_count


def list_csv_files(directory: str) -> list[str]:
    return sorted(
        entry
        for entry in os.listdir(directory)
        if entry.lower().endswith(".csv")
        and os.path.isfile(os.path.join(directory, entry))
    )


def list_csv_files_recursive(directory: str) -> list[str]:
    csv_paths = []
    for root, _, files in os.walk(directory):
        for name in files:
            if name.lower().endswith(".csv"):
                csv_paths.append(os.path.join(root, name))
    return sorted(csv_paths)


def main() -> int:
    args = parse_args()
    target_dir = args.path

    if not os.path.isdir(target_dir):
        print(f"{RED}Error: directory not found -> {target_dir}{RESET}")
        return 1

    if args.recursive:
        csv_files = list_csv_files_recursive(target_dir)
        if not csv_files:
            print(f"{RED}No CSV files found under: {target_dir}{RESET}")
            return 1
        print(f"[INFO] Found {len(csv_files)} CSV file(s) recursively under {target_dir}")
    else:
        csv_files = list_csv_files(target_dir)
        if not csv_files:
            print(f"{RED}No CSV files found in directory: {target_dir}{RESET}")
            return 1

    had_error = False
    for csv_file in csv_files:
        if os.path.isabs(csv_file):
            file_path = csv_file
        else:
            file_path = os.path.join(target_dir, csv_file)
        try:
            row_count, column_count, empty_row_count, nan_row_count = analyze_csv(file_path)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"{RED}[{csv_file}] Error reading CSV: {exc}{RESET}")
            had_error = True
            continue

        print(f"[{csv_file}]")
        print(f"  Total rows: {row_count}")
        print(f"  Columns: {column_count}")
        print(f"  Empty rows: {empty_row_count}")
        print(f"  Rows with NaN: {nan_row_count}")

    return 0 if not had_error else 1


if __name__ == "__main__":
    sys.exit(main())
