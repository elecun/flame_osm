'''
Dataset Builder for OSM
@author Byunghun Hwang <bh.hwang@iae.re.kr>
'''

import csv
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import argparse
import pathlib
import logging
import colorlog
import os
import pandas as pd

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Building Occupant Monitoring Dataset')
    parser.add_argument('--path', nargs='?', required=False, help="Raw-data Root Path", default="./")
    parser.add_argument('--out', nargs='?', required=False, help="Result CSV", default="./")
    args = parser.parse_args()