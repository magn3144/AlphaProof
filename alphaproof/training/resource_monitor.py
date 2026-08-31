"""Periodic W&B logging for resources allocated to an LSF job."""

import os
import subprocess
import threading
import time
from typing import Any

import psutil


RESOURCE_LOG_INTERVAL_SECONDS = 15.0


class ResourceMonitor(threading.Thread):
    """Log job-scoped CPU, RAM, and GPU utilization to W&B."""

    def __init__(self, wandb_run: Any):
        super().__init__(name='resource-monitor', daemon=True)
        self.wandb_run = wandb_run
        self.process = psutil.Process()
        self.allocated_cpus = int(os.environ['LSB_DJOB_NUMPROC'])
        self.allocated_memory_bytes = int(
            os.environ['ALPHAPROOF_ALLOCATED_MEMORY_BYTES']
        )
        self.gpu_ids = os.environ['CUDA_VISIBLE_DEVICES']
        self.stop_event = threading.Event()
        self.previous_cpu_seconds = self._cpu_seconds()
        self.previous_time = time.monotonic()

    def run(self) -> None:
        while not self.stop_event.wait(RESOURCE_LOG_INTERVAL_SECONDS):
            self.log_resources()

    def stop(self) -> None:
        self.stop_event.set()
        self.join()

    def log_resources(self) -> None:
        now = time.monotonic()
        cpu_seconds = self._cpu_seconds()
        cpu_percent = 100.0 * (
            cpu_seconds - self.previous_cpu_seconds
        ) / ((now - self.previous_time) * self.allocated_cpus)
        self.previous_cpu_seconds = cpu_seconds
        self.previous_time = now

        memory_bytes = sum(
            process.memory_info().rss for process in self._processes()
        )
        gpu_utilization, gpu_memory_percent = self._gpu_metrics()
        self.wandb_run.log({
            'resources/cpu_utilization_percent': cpu_percent,
            'resources/cpu_memory_percent': (
                100.0 * memory_bytes / self.allocated_memory_bytes
            ),
            'resources/gpu_utilization_percent': gpu_utilization,
            'resources/gpu_memory_percent': gpu_memory_percent,
        })

    def _processes(self) -> list[psutil.Process]:
        return [self.process, *self.process.children(recursive=True)]

    def _cpu_seconds(self) -> float:
        cpu_times = [process.cpu_times() for process in self._processes()]
        return sum(times.user + times.system for times in cpu_times)

    def _gpu_metrics(self) -> tuple[float, float]:
        result = subprocess.run(
            [
                'nvidia-smi',
                f'--id={self.gpu_ids}',
                '--query-gpu=utilization.gpu,memory.used,memory.total',
                '--format=csv,noheader,nounits',
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        rows = [
            tuple(float(value.strip()) for value in line.split(','))
            for line in result.stdout.splitlines()
        ]
        gpu_utilization = sum(row[0] for row in rows) / len(rows)
        memory_used = sum(row[1] for row in rows)
        memory_total = sum(row[2] for row in rows)
        return gpu_utilization, 100.0 * memory_used / memory_total


def start_resource_monitor(wandb_run: Any) -> ResourceMonitor | None:
    """Start monitoring when training is running as an LSF job."""
    if (
        'LSB_JOBID' not in os.environ
        or 'ALPHAPROOF_ALLOCATED_MEMORY_BYTES' not in os.environ
    ):
        return None
    monitor = ResourceMonitor(wandb_run)
    monitor.start()
    return monitor
