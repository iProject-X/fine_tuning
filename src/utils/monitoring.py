"""
Monitoring and metrics utilities
"""

import time
import logging
from typing import Dict, Any, Callable
from functools import wraps
from collections import defaultdict, deque
import threading

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collect and track metrics for the ASR service"""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics = defaultdict(deque)
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.lock = threading.Lock()

    def record_transcription(
        self,
        processing_time: float,
        language: str,
        audio_duration: float
    ):
        """Record transcription metrics"""
        with self.lock:
            # Processing time histogram
            self.metrics['processing_times'].append(processing_time)
            if len(self.metrics['processing_times']) > self.max_history:
                self.metrics['processing_times'].popleft()

            # Language counter
            self.counters[f'transcriptions_{language}'] += 1
            self.counters['transcriptions_total'] += 1

            # Real-time factor
            rtf = processing_time / max(audio_duration, 0.001)
            self.metrics['rtf'].append(rtf)
            if len(self.metrics['rtf']) > self.max_history:
                self.metrics['rtf'].popleft()

    def record_error(self, error_type: str):
        """Record error occurrence"""
        with self.lock:
            self.counters[f'errors_{error_type}'] += 1
            self.counters['errors_total'] += 1

    def set_gauge(self, name: str, value: float):
        """Set gauge value"""
        with self.lock:
            self.gauges[name] = value

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot"""
        with self.lock:
            # Calculate statistics
            processing_times = list(self.metrics['processing_times'])
            rtf_values = list(self.metrics['rtf'])

            metrics = {
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'histograms': {}
            }

            if processing_times:
                metrics['histograms']['processing_time'] = {
                    'count': len(processing_times),
                    'mean': sum(processing_times) / len(processing_times),
                    'min': min(processing_times),
                    'max': max(processing_times),
                    'p50': self._percentile(processing_times, 50),
                    'p95': self._percentile(processing_times, 95),
                    'p99': self._percentile(processing_times, 99)
                }

            if rtf_values:
                metrics['histograms']['rtf'] = {
                    'count': len(rtf_values),
                    'mean': sum(rtf_values) / len(rtf_values),
                    'min': min(rtf_values),
                    'max': max(rtf_values),
                    'p50': self._percentile(rtf_values, 50),
                    'p95': self._percentile(rtf_values, 95),
                    'p99': self._percentile(rtf_values, 99)
                }

            return metrics

    def _percentile(self, values: list, percentile: int) -> float:
        """Calculate percentile"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

def track_request(operation_name: str):
    """Decorator to track request metrics"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                logger.error(f"Operation {operation_name} failed: {e}")
                raise
            finally:
                end_time = time.time()
                duration = end_time - start_time

                # Log metrics (would integrate with your metrics collector)
                logger.info(
                    f"Operation: {operation_name}, "
                    f"Duration: {duration:.3f}s, "
                    f"Success: {success}"
                )

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                logger.error(f"Operation {operation_name} failed: {e}")
                raise
            finally:
                end_time = time.time()
                duration = end_time - start_time

                logger.info(
                    f"Operation: {operation_name}, "
                    f"Duration: {duration:.3f}s, "
                    f"Success: {success}"
                )

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator