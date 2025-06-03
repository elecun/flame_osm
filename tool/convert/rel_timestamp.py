import csv
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import argparse
import pathlib
import logging
import colorlog
import os

class ConsoleLogger:
    _logger = None
    
    @classmethod
    def get_logger(cls):
        if cls._logger is None:
            cls._logger = logging.getLogger(__name__)
            cls._logger.setLevel(logging.DEBUG)
            
            formatter = colorlog.ColoredFormatter(
                '[%(asctime)s] %(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s',
                log_colors={
                    'DEBUG':    'cyan',
                    'INFO':     'green',
                    'WARNING':  'yellow',
                    'ERROR':    'red',
                    'CRITICAL': 'bold_red,bg_white',
                }
            )
            
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)
            
            cls._logger.addHandler(ch)
        return cls._logger
console = ConsoleLogger.get_logger()

def convert_unix_to_relative(input_csv, output_csv):
    with open(input_csv, newline='', encoding='utf-8') as f:
        reader = list(csv.reader(f))

    if not reader:
        print("CSV 파일이 비어 있습니다.")
        return

    base_time = float(reader[0][0])
    converted_rows = []

    for row in reader:
        try:
            unix_time = float(row[0])
            relative_time = unix_time - base_time
            converted_rows.append([f"{relative_time:.3f}"] + row[1:])
        except (IndexError, ValueError):
            print(f"행 무시됨: {row}")

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(converted_rows)

    print(f"상대 시간으로 변환된 CSV가 저장되었습니다: {output_csv}")


def get_file_pathlist(root_path:str, find_list:list):
    pass

def get_scenario_time(scenario_history_file:str):
    pass

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Generatre EAF files')
    parser.add_argument('--input', nargs='?', required=False, help="Input CSV", default="./")
    parser.add_argument('--output', nargs='?', required=False, help="Output CSV", default="./")
    args = parser.parse_args()
    
    convert_unix_to_relative(args.input, args.output)