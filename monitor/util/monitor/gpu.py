'''
GPU Monitoring Class
@author Byunghun Hwang<bh.hwang@iae.re.kr>
'''


import pynvml
try:
    from PyQt6.QtCore import QThread, pyqtSignal
except ImportError:
    from PyQt5.QtCore import QThread, pyqtSignal
from util.logger.console import ConsoleLogger


class GPUStatusMonitor(QThread):
    
    usage_update_signal = pyqtSignal(dict)
    
    def __init__(self, interval_ms:int=1000):
        super().__init__()
        
        self.console = ConsoleLogger.get_logger()
        
        self.interval = interval_ms
        self.usage = {}
        self.gpu_handle = []
        self.is_available = False
        
        try:
            pynvml.nvmlInit()
            self.usage["gpu_count"] = pynvml.nvmlDeviceGetCount()
            for gpu_id in range(self.usage["gpu_count"]):
                self.gpu_handle.append(pynvml.nvmlDeviceGetHandleByIndex(gpu_id))
            
            self.is_available = True
        except pynvml.nvml.NVMLError:
            self.console.warning(f"pynvml does not support for this device")
            self.is_available = False
    
    # loop function
    def run(self):
        while True:
            try:
                if self.is_available==False:
                    self.console.warning("Warning : GPU Status Monitor id not able to perform for this device")
                    break
                
                if self.isInterruptionRequested():
                    break
                
                for id, handle in enumerate(self.gpu_handle):
                    info = pynvml.nvmlDeviceGetUtilizationRates(handle)    
                    self.usage[f"gpu_{id}"] = int(info.gpu)
                    self.usage[f"memory_{id}"] = int(info.memory)
                    
                self.usage_update_signal.emit(self.usage)
                
                QThread.msleep(self.interval)
            except pynvml.nvml.NVMLError:
                self.console.warning(f"This device does not support pynvml for GPU status monitoring")
                break # break threading
    
    # close thread        
    def close(self) -> None:
        if self.is_available:
            self.requestInterruption()
            self.quit()
            self.wait(1000)
        
            pynvml.nvmlShutdown()