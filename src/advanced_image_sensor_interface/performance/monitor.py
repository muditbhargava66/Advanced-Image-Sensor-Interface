"""
Real-time performance monitoring for system resources and application metrics.
"""

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import psutil

from ..types import Megabytes, Percentage, Seconds

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System resource metrics snapshot."""

    timestamp: float
    cpu_percent: Percentage
    memory_percent: Percentage
    memory_used_mb: Megabytes
    memory_available_mb: Megabytes
    disk_io_read_mb: Megabytes
    disk_io_write_mb: Megabytes
    network_sent_mb: Megabytes
    network_recv_mb: Megabytes
    temperature_c: Optional[float] = None
    gpu_utilization: Optional[Percentage] = None
    gpu_memory_mb: Optional[Megabytes] = None


@dataclass
class ApplicationMetrics:
    """Application-specific metrics."""

    timestamp: float
    frames_per_second: float
    buffer_utilization: Percentage
    active_connections: int
    error_rate: float
    latency_ms: float
    throughput_mbps: float


class PerformanceMonitor:
    """
    Real-time performance monitoring system.

    Monitors system resources, application metrics, and provides
    alerts when thresholds are exceeded.
    """

    def __init__(self, history_size: int = 1000, sampling_interval: Seconds = 1.0, enable_alerts: bool = True):
        """Initialize performance monitor."""
        self.history_size = history_size
        self.sampling_interval = sampling_interval
        self.enable_alerts = enable_alerts

        # Metric storage
        self.system_history: deque = deque(maxlen=history_size)
        self.app_history: deque = deque(maxlen=history_size)

        # Monitoring state
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Alert thresholds
        self.alert_thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_io_mb_per_sec": 100.0,
            "error_rate": 0.05,  # 5%
            "latency_ms": 1000.0,
        }

        # Alert callbacks
        self.alert_callbacks: list[Callable[[str, dict[str, Any]], None]] = []

        # System monitoring
        self._process = psutil.Process()
        self._last_disk_io = None
        self._last_network_io = None

    def add_alert_callback(self, callback: Callable[[str, dict[str, Any]], None]) -> None:
        """Add callback for alert notifications."""
        self.alert_callbacks.append(callback)

    def set_alert_threshold(self, metric: str, threshold: float) -> None:
        """Set alert threshold for a metric."""
        self.alert_thresholds[metric] = threshold

    def start_monitoring(self) -> None:
        """Start continuous monitoring."""
        if self._monitoring:
            logger.warning("Monitoring already started")
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Performance monitoring started")

    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        if not self._monitoring:
            return

        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("Performance monitoring stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()

                with self._lock:
                    self.system_history.append(system_metrics)

                # Check for alerts
                if self.enable_alerts:
                    self._check_system_alerts(system_metrics)

                time.sleep(self.sampling_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.sampling_interval)

    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()

        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_read_mb = 0.0
        disk_write_mb = 0.0

        if disk_io and self._last_disk_io:
            read_bytes = disk_io.read_bytes - self._last_disk_io.read_bytes
            write_bytes = disk_io.write_bytes - self._last_disk_io.write_bytes
            disk_read_mb = read_bytes / (1024 * 1024)
            disk_write_mb = write_bytes / (1024 * 1024)

        self._last_disk_io = disk_io

        # Network I/O
        network_io = psutil.net_io_counters()
        network_sent_mb = 0.0
        network_recv_mb = 0.0

        if network_io and self._last_network_io:
            sent_bytes = network_io.bytes_sent - self._last_network_io.bytes_sent
            recv_bytes = network_io.bytes_recv - self._last_network_io.bytes_recv
            network_sent_mb = sent_bytes / (1024 * 1024)
            network_recv_mb = recv_bytes / (1024 * 1024)

        self._last_network_io = network_io

        # Temperature (if available)
        temperature = None
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Get CPU temperature if available
                for name, entries in temps.items():
                    if "cpu" in name.lower() or "core" in name.lower():
                        temperature = entries[0].current
                        break
        except (AttributeError, OSError):
            pass  # Temperature sensors not available

        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_available_mb=memory.available / (1024 * 1024),
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
            temperature_c=temperature,
        )

    def _check_system_alerts(self, metrics: SystemMetrics) -> None:
        """Check system metrics against alert thresholds."""
        alerts = []

        if metrics.cpu_percent > self.alert_thresholds.get("cpu_percent", 80.0):
            alerts.append(("high_cpu", {"current": metrics.cpu_percent, "threshold": self.alert_thresholds["cpu_percent"]}))

        if metrics.memory_percent > self.alert_thresholds.get("memory_percent", 85.0):
            alerts.append(
                ("high_memory", {"current": metrics.memory_percent, "threshold": self.alert_thresholds["memory_percent"]})
            )

        disk_io_total = metrics.disk_io_read_mb + metrics.disk_io_write_mb
        if disk_io_total > self.alert_thresholds.get("disk_io_mb_per_sec", 100.0):
            alerts.append(("high_disk_io", {"current": disk_io_total, "threshold": self.alert_thresholds["disk_io_mb_per_sec"]}))

        # Trigger alert callbacks
        for alert_type, alert_data in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert_type, alert_data)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")

    def record_application_metrics(self, metrics: ApplicationMetrics) -> None:
        """Record application-specific metrics."""
        with self._lock:
            self.app_history.append(metrics)

        # Check application alerts
        if self.enable_alerts:
            self._check_application_alerts(metrics)

    def _check_application_alerts(self, metrics: ApplicationMetrics) -> None:
        """Check application metrics against thresholds."""
        alerts = []

        if metrics.error_rate > self.alert_thresholds.get("error_rate", 0.05):
            alerts.append(("high_error_rate", {"current": metrics.error_rate, "threshold": self.alert_thresholds["error_rate"]}))

        if metrics.latency_ms > self.alert_thresholds.get("latency_ms", 1000.0):
            alerts.append(("high_latency", {"current": metrics.latency_ms, "threshold": self.alert_thresholds["latency_ms"]}))

        # Trigger alert callbacks
        for alert_type, alert_data in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert_type, alert_data)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")

    def get_system_metrics(self, last_n: Optional[int] = None) -> list[SystemMetrics]:
        """Get system metrics history."""
        with self._lock:
            metrics = list(self.system_history)

        if last_n is not None:
            metrics = metrics[-last_n:]

        return metrics

    def get_application_metrics(self, last_n: Optional[int] = None) -> list[ApplicationMetrics]:
        """Get application metrics history."""
        with self._lock:
            metrics = list(self.app_history)

        if last_n is not None:
            metrics = metrics[-last_n:]

        return metrics

    def get_current_system_metrics(self) -> SystemMetrics:
        """Get current system metrics snapshot."""
        return self._collect_system_metrics()

    def get_summary_stats(self, window_minutes: int = 5) -> dict[str, Any]:
        """Get summary statistics for the specified time window."""
        cutoff_time = time.time() - (window_minutes * 60)

        with self._lock:
            # Filter recent system metrics
            recent_system = [m for m in self.system_history if m.timestamp >= cutoff_time]
            recent_app = [m for m in self.app_history if m.timestamp >= cutoff_time]

        if not recent_system:
            return {"error": "No data available for the specified window"}

        # Calculate system statistics
        avg_cpu = sum(m.cpu_percent for m in recent_system) / len(recent_system)
        max_cpu = max(m.cpu_percent for m in recent_system)
        avg_memory = sum(m.memory_percent for m in recent_system) / len(recent_system)
        max_memory = max(m.memory_percent for m in recent_system)

        total_disk_read = sum(m.disk_io_read_mb for m in recent_system)
        total_disk_write = sum(m.disk_io_write_mb for m in recent_system)
        total_network_sent = sum(m.network_sent_mb for m in recent_system)
        total_network_recv = sum(m.network_recv_mb for m in recent_system)

        stats = {
            "window_minutes": window_minutes,
            "sample_count": len(recent_system),
            "system": {
                "cpu_percent_avg": avg_cpu,
                "cpu_percent_max": max_cpu,
                "memory_percent_avg": avg_memory,
                "memory_percent_max": max_memory,
                "disk_read_mb_total": total_disk_read,
                "disk_write_mb_total": total_disk_write,
                "network_sent_mb_total": total_network_sent,
                "network_recv_mb_total": total_network_recv,
            },
        }

        # Add application statistics if available
        if recent_app:
            avg_fps = sum(m.frames_per_second for m in recent_app) / len(recent_app)
            avg_latency = sum(m.latency_ms for m in recent_app) / len(recent_app)
            avg_throughput = sum(m.throughput_mbps for m in recent_app) / len(recent_app)
            avg_error_rate = sum(m.error_rate for m in recent_app) / len(recent_app)

            stats["application"] = {
                "fps_avg": avg_fps,
                "latency_ms_avg": avg_latency,
                "throughput_mbps_avg": avg_throughput,
                "error_rate_avg": avg_error_rate,
                "sample_count": len(recent_app),
            }

        return stats


class SystemMonitor:
    """Simplified system resource monitor."""

    @staticmethod
    def get_cpu_usage() -> Percentage:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=1.0)

    @staticmethod
    def get_memory_usage() -> dict[str, float]:
        """Get memory usage information."""
        memory = psutil.virtual_memory()
        return {
            "percent": memory.percent,
            "used_mb": memory.used / (1024 * 1024),
            "available_mb": memory.available / (1024 * 1024),
            "total_mb": memory.total / (1024 * 1024),
        }

    @staticmethod
    def get_disk_usage(path: str = "/") -> dict[str, float]:
        """Get disk usage for specified path."""
        usage = psutil.disk_usage(path)
        return {
            "percent": (usage.used / usage.total) * 100,
            "used_gb": usage.used / (1024**3),
            "free_gb": usage.free / (1024**3),
            "total_gb": usage.total / (1024**3),
        }

    @staticmethod
    def get_network_stats() -> dict[str, float]:
        """Get network I/O statistics."""
        stats = psutil.net_io_counters()
        return {
            "bytes_sent_mb": stats.bytes_sent / (1024 * 1024),
            "bytes_recv_mb": stats.bytes_recv / (1024 * 1024),
            "packets_sent": stats.packets_sent,
            "packets_recv": stats.packets_recv,
        }

    @staticmethod
    def get_process_info() -> dict[str, Any]:
        """Get current process information."""
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "pid": process.pid,
            "cpu_percent": process.cpu_percent(),
            "memory_mb": memory_info.rss / (1024 * 1024),
            "memory_percent": process.memory_percent(),
            "num_threads": process.num_threads(),
            "create_time": process.create_time(),
            "status": process.status(),
        }
