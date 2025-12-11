#!/usr/bin/env python3
import argparse
import csv
import os
import sys


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
    return parser.parse_args()


def analyze_csv(file_path: str) -> tuple[int, int, int]:
    """
    Returns a tuple of (row_count, column_count, empty_row_count).
    - row_count: total rows encountered (including empty rows)
    - column_count: number of columns based on the first non-empty row (0 if none)
    - empty_row_count: rows where all fields are empty/whitespace
    """
    row_count = 0
    empty_row_count = 0
    column_count = 0

    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            row_count += 1
            # Treat a row as empty if every cell is empty/whitespace.
            if all(cell.strip() == "" for cell in row):
                empty_row_count += 1
                continue
            if column_count == 0:
                column_count = len(row)

    return row_count, column_count, empty_row_count


def list_csv_files(directory: str) -> list[str]:
    return sorted(
        entry
        for entry in os.listdir(directory)
        if entry.lower().endswith(".csv")
        and os.path.isfile(os.path.join(directory, entry))
    )


def main() -> int:
    args = parse_args()
    target_dir = args.path

    if not os.path.isdir(target_dir):
        print(f"Error: directory not found -> {target_dir}")
        return 1

    csv_files = list_csv_files(target_dir)
    if not csv_files:
        print(f"No CSV files found in directory: {target_dir}")
        return 1

    had_error = False
    for csv_file in csv_files:
        file_path = os.path.join(target_dir, csv_file)
        try:
            row_count, column_count, empty_row_count = analyze_csv(file_path)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[{csv_file}] Error reading CSV: {exc}")
            had_error = True
            continue

        print(f"[{csv_file}]")
        print(f"  Total rows: {row_count}")
        print(f"  Columns: {column_count}")
        print(f"  Empty rows: {empty_row_count}")

    return 0 if not had_error else 1


if __name__ == "__main__":
    sys.exit(main())
