"""The sampling loop shared by the background monitors (ResourceMonitor, EnergyMonitor).

A monitor gives one reading; the thread takes it at a fixed interval, and once more after the stop
signal so the window ends with a reading. memory_profiler 0.61 reads the same way (its sampler
reads once more after its stop signal, ``memory_profiler.py``), and codecarbon 3.2.2 measures
"from last scheduled measurement to shutdown" when it stops. Every reading runs on the sampling
thread, so no two ever run at once.
"""

from __future__ import annotations

import threading
from collections.abc import Callable


# How long stop waits for a reading in progress and the closing one; a thread still alive after it
# is an error, since its readings would otherwise run beside whatever the caller does next.
STOP_TIMEOUT_SEC = 2.0


class SamplingThreadStillRunningError(RuntimeError):
    """The sampling thread outlived the bounded join, so the window did not close."""


class SamplingThread:
    """A daemon thread that takes a reading every ``interval_sec`` and one when it stops.

    ```python
    thread = SamplingThread(self._take_sample, interval_sec=0.1)
    thread.start()   # in __enter__
    thread.stop()    # in __exit__: the closing reading, then the join
    ```
    """

    def __init__(
        self,
        sample: Callable[[], None],
        *,
        interval_sec: float,
        join_timeout_sec: float = STOP_TIMEOUT_SEC,
    ) -> None:
        """Initialize the sampling thread.

        Args:
            sample: Takes and records one reading; it runs on the sampling thread only.
            interval_sec: Seconds between readings.
            join_timeout_sec: How long ``stop`` waits for the thread to finish.
        """
        self._sample = sample
        self._interval = interval_sec
        self._join_timeout = join_timeout_sec
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start sampling on a new daemon thread."""
        self._stop_event.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:  # noqa: DOC503  # re-raises the reading's own error
        """Signal the thread, wait for its closing reading, and raise what went wrong.

        Raises:
            SamplingThreadStillRunningError: If the thread is still alive after the bounded join.
            Exception: Whatever a reading raised; the readings taken before it stay.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._join_timeout)
            if self._thread.is_alive():
                msg = f"the sampling thread did not stop within {self._join_timeout} s"
                raise SamplingThreadStillRunningError(msg)
        if self._error is not None:
            raise self._error

    def _run(self) -> None:
        """Read every interval until the stop signal, then once more.

        Any error is kept for ``stop`` to raise in the caller's thread.
        """
        try:
            stopping = False
            while True:
                self._sample()
                if stopping:
                    return
                stopping = self._stop_event.wait(timeout=self._interval)
        except Exception as error:  # re-raised by stop() in the caller's thread
            self._error = error
