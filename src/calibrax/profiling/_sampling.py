"""Shared sampling thread lifecycle for background monitors.

Provides a context-manager helper that encapsulates the daemon thread
start/stop pattern used by ResourceMonitor and EnergyMonitor.
"""

from __future__ import annotations

import threading
from collections.abc import Callable


class SamplingThread:
    """Reusable daemon thread lifecycle for background sampling.

    Usage:

    ```python
    thread = SamplingThread(target=self._sample_loop)
    thread.start()   # in __enter__
    thread.stop()    # in __exit__
    ```

    Args:
        target: The sampling loop callable (runs in the daemon thread).
    """

    def __init__(self, target: Callable[[], None]) -> None:
        """Initialize the sampling thread helper.

        Args:
            target: Callable to run in the background thread.
        """
        self._target = target
        self._thread: threading.Thread | None = None
        self.stop_event = threading.Event()

    def start(self) -> None:
        """Clear stop event and start the daemon thread."""
        self.stop_event.clear()
        self._thread = threading.Thread(target=self._target, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the thread to stop and wait for it to finish."""
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
