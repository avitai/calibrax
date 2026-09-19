"""SamplingThread: the background loop's lifecycle and its errors."""

from __future__ import annotations

import pytest

from calibrax.profiling._sampling import SamplingThread


class _ReadingFailedError(Exception):
    pass


def test_an_error_in_the_loop_is_raised_by_stop() -> None:
    def failing_loop() -> None:
        raise _ReadingFailedError

    thread = SamplingThread(target=failing_loop)
    thread.start()

    with pytest.raises(_ReadingFailedError):
        thread.stop()


def test_a_loop_that_ends_cleanly_stops_without_error() -> None:
    thread = SamplingThread(target=lambda: None)
    thread.start()

    thread.stop()


def test_a_restart_forgets_the_previous_error() -> None:
    calls: list[int] = []

    def fails_first_time() -> None:
        calls.append(1)
        if len(calls) == 1:
            raise _ReadingFailedError

    thread = SamplingThread(target=fails_first_time)
    thread.start()
    with pytest.raises(_ReadingFailedError):
        thread.stop()

    thread.start()
    thread.stop()
