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

# Find the CSV file in the directory tree
def find_csv_file(start_dir, filename:str):
    for root, dirs, files in os.walk(start_dir):
        if filename in files:
            return os.path.abspath(os.path.join(root, filename))
    return None

def unix_to_timeval(unix_start, current):
    delta = current - unix_start
    return int(delta.total_seconds() * 1000)  # milliseconds

import csv
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

def unix_to_relative_ms(base_ts, current_ts):
    return int((current_ts - base_ts) * 1000)  # milliseconds

def create_eaf_from_csv(csv_file, output_eaf_file):
    # CSV 파일 읽기 (헤더 없음)
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = list(csv.reader(f))

    if not reader:
        print("CSV 파일이 비어 있습니다.")
        return

    # 기준 시간 설정 (첫 줄의 첫 번째 값)
    base_ts = float(reader[0][0])

    # EAF 루트 생성
    root = ET.Element('ANNOTATION_DOCUMENT', {
        'AUTHOR': '',
        'DATE': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S'),
        'FORMAT': '3.0',
        'VERSION': '3.0',
        'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
        'xsi:noNamespaceSchemaLocation': 'http://www.mpi.nl/tools/elan/EAFv3.0.xsd'
    })

    time_order = ET.SubElement(root, 'TIME_ORDER')
    tier = ET.SubElement(root, 'TIER', {'TIER_ID': 'Comments'})
    time_id_map = {}

    for i, row in enumerate(reader):
        try:
            unix_ts = float(row[0])
            comment = row[1]
        except IndexError:
            print(f"행 {i}가 잘못되었습니다. 건너뜁니다.")
            continue

        rel_time_ms = unix_to_relative_ms(base_ts, unix_ts)
        ts_id = f"ts{i}"
        time_id_map[i] = ts_id

        ET.SubElement(time_order, 'TIME_SLOT', {
            'TIME_SLOT_ID': ts_id,
            'TIME_VALUE': str(rel_time_ms)
        })

        annotation = ET.SubElement(tier, 'ANNOTATION')
        alignable = ET.SubElement(annotation, 'ALIGNABLE_ANNOTATION', {
            'ANNOTATION_ID': f"a{i}",
            'TIME_SLOT_REF1': ts_id,
            'TIME_SLOT_REF2': ts_id
        })
        ET.SubElement(alignable, 'ANNOTATION_VALUE').text = comment

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(output_eaf_file, encoding='utf-8', xml_declaration=True)
    print(f"EAF 파일이 생성되었습니다: {output_eaf_file}")



def generate_eaf(input_path, output_path):
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
        
        for k in required_file_path.keys():
            if k is "scenario_history.csv":
                create_eaf_from_csv(required_file_path[k], os.path.splitext(required_file_path[k])[0] + ".eaf")

    except FileNotFoundError as e:
        console.error(f"{e}")
        return
    



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Generatre EAF files')
    parser.add_argument('--input', nargs='?', required=False, help="Input Path", default="./")
    parser.add_argument('--output', nargs='?', required=False, help="Output Path", default="./")
    args = parser.parse_args()
    
    generate_eaf(args.input, args.output)