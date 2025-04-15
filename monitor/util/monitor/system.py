'''
System Usage Monitoring
@author Byunghun Hwang<bh.hwang@iae.re.kr>
'''


import psutil
try:
    from PyQt6.QtCore import QObject, QThread, pyqtSignal
except ImportError:
    from PyQt5.QtCore import QObject, QThread, pyqtSignal
import time


# System Status Monitoring with QThread
class SystemStatusMonitor(QThread):
    
    usage_update_signal = pyqtSignal(dict)
    
    def __init__(self, interval_ms:int=1000):
        super().__init__()
        
        self.interval = interval_ms
        self.usage = {}
    
    def run(self):
        while True:
            if self.isInterruptionRequested():
                break
            
            self.usage["cpu"] = psutil.cpu_percent()
            self.usage["memory"] = psutil.virtual_memory().percent
            self.usage["storage"] = psutil.disk_usage('/').percent
            self.usage["net_send"] = psutil.net_io_counters().bytes_sent
            self.usage["net_recv"] = psutil.net_io_counters().bytes_recv
            
            # emit signal
            self.usage_update_signal.emit(self.usage)
            
            QThread.msleep(self.interval)

    # close thread
    def close(self) -> None:
        self.requestInterruption()
        self.quit()
        self.wait(1000)