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
    """

    def __init__(self, target: Callable[[], None]) -> None:
        """Initialize the sampling thread helper.

        Args:
            target: Callable to run in the background thread.
        """
        self._target = target
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self.stop_event = threading.Event()

    def start(self) -> None:
        """Clear stop event and start the daemon thread."""
        self.stop_event.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:  # noqa: DOC503  # re-raises the target's own error
        """Signal the thread to stop, wait for it, and raise the error that ended it early.

        Raises:
            Exception: Whatever the target raised; the samples taken before it stay.
        """
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._error is not None:
            raise self._error

    def _run(self) -> None:
        """Run the target, keeping its error for ``stop`` to raise in the caller's thread."""
        try:
            self._target()
        except Exception as error:  # re-raised by stop() in the caller's thread
            self._error = error
