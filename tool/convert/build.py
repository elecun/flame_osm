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

def find_csv_file(start_dir, filename:str):
    for root, dirs, files in os.walk(start_dir):
        if filename in files:
            return os.path.abspath(os.path.join(root, filename))
    return None

def build_file_pathlist(input_path):
    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)
    try:
        if not input_path.exists():
            raise FileNotFoundError(f"Input path '{input_path}' does not exist.")
        if not output_path.exists():    
            output_path.mkdir(parents=True, exist_ok=True)

        required_files = ["timestamp_0.csv", "timestamp_2.csv", "timestamp_5.csv", "timestamp_6.csv", "scenario_history.csv", "nback_response.csv", "blinks.csv", "3d_eye_states.csv", "fixations.csv", "gaze.csv", "saccades.csv", "world_timestamps.csv", "events.csv"]
        required_file_path = {}
        # find all required files in input path
        for file in required_files:
            file_path = find_csv_file(input_path, file)
            if file_path is None:
                console.error(f"Required file '{file}' not found in '{input_path}'.")
            required_file_path[file] = file_path

def __build_path_list(input_path):
    input_path = pathlib.Path(input_path)
    required_files = ["timestamp_0.csv", "timestamp_2.csv", "timestamp_5.csv", "timestamp_6.csv", "scenario_history.csv", "nback_response.csv", "blinks.csv", "3d_eye_states.csv", "fixations.csv", "gaze.csv", "saccades.csv", "world_timestamps.csv", "events.csv"]
    required_file_path = {}
    # find all required files in input path
    for file in required_files:
        file_path = find_csv_file(input_path, file)
        if file_path is None:
            console.error(f"Required file '{file}' not found in '{input_path}'.")
        required_file_path[file] = file_path

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Generatre EAF files')
    parser.add_argument('--input', nargs='?', required=False, help="Input Path", default="./")
    parser.add_argument('--output', nargs='?', required=False, help="Output Path", default="./")
    args = parser.parse_args()

    # 1. listup path
    __build_path_list(args.input)
    
    generate_eaf(args.input, args.output)