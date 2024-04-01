from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates
from transformers import TrainerCallback

class GpuUsageCallback(TrainerCallback):
    """
    A bare :class:`~transformers.TrainerCallback` that just prints the logs.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.print_gpu_usage()

    def print_gpu_usage(self):
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        mem = nvmlDeviceGetMemoryInfo(handle)
        util = nvmlDeviceGetUtilizationRates(handle)
        print(f"GPU memory occupied: {mem.used//1024**2} MB.\nGPU utlization: {util.utilization}")
