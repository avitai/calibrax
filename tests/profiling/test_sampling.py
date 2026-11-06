"""SamplingThread: the sampling loop, the reading that closes the window, and its errors."""

from __future__ import annotations

import threading

import pytest

from calibrax.profiling._sampling import SamplingThread, SamplingThreadStillRunningError
from tests.factories import wait_until


class _ReadingFailedError(Exception):
    pass


def test_the_window_closes_with_a_reading_after_stop() -> None:
    """One reading when sampling starts and one after the stop signal, whatever the interval.

    memory_profiler reads once more after its stop signal; without that reading the last
    interval of a block, up to a whole ``interval_sec``, is never measured.
    """
    readings: list[int] = []
    thread = SamplingThread(lambda: readings.append(1), interval_sec=60.0)
    thread.start()
    wait_until(lambda: len(readings) == 1)

    thread.stop()

    assert len(readings) == 2


def test_every_reading_runs_on_the_sampling_thread() -> None:
    """The closing reading too, so no reading ever runs beside another."""
    threads: list[int] = []
    thread = SamplingThread(lambda: threads.append(threading.get_ident()), interval_sec=60.0)
    thread.start()
    wait_until(lambda: len(threads) == 1)

    thread.stop()

    assert len(threads) == 2
    assert len(set(threads)) == 1
    assert threads[0] != threading.get_ident()


def test_a_thread_still_running_after_the_bounded_join_is_an_error() -> None:
    """A reading that outlasts the join leaves the thread alive; stop says so rather than
    returning as if the window had closed."""
    entered, release = threading.Event(), threading.Event()

    def stuck() -> None:
        entered.set()
        release.wait()

    thread = SamplingThread(stuck, interval_sec=60.0, join_timeout_sec=0.05)
    thread.start()
    assert entered.wait(timeout=5.0)
    try:
        with pytest.raises(SamplingThreadStillRunningError):
            thread.stop()
    finally:
        release.set()


def test_an_error_in_a_reading_is_raised_by_stop() -> None:
    def failing() -> None:
        raise _ReadingFailedError

    thread = SamplingThread(failing, interval_sec=0.05)
    thread.start()

    with pytest.raises(_ReadingFailedError):
        thread.stop()


def test_an_error_in_the_closing_reading_is_raised_by_stop() -> None:
    calls: list[int] = []

    def fails_after_the_first() -> None:
        calls.append(1)
        if len(calls) > 1:
            raise _ReadingFailedError

    thread = SamplingThread(fails_after_the_first, interval_sec=60.0)
    thread.start()
    wait_until(lambda: len(calls) == 1)

    with pytest.raises(_ReadingFailedError):
        thread.stop()


def test_a_restart_forgets_the_previous_error() -> None:
    calls: list[int] = []

    def fails_first_time() -> None:
        calls.append(1)
        if len(calls) == 1:
            raise _ReadingFailedError

    thread = SamplingThread(fails_first_time, interval_sec=0.05)
    thread.start()
    with pytest.raises(_ReadingFailedError):
        thread.stop()

    thread.start()
    thread.stop()
