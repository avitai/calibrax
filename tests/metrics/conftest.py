"""Fixtures shared by the metric tests.

Lives at this level because the masked and weighted statistic tests sit both directly under
``tests/metrics`` and under ``tests/metrics/functional``, and three of them had grown their
own byte-identical copy of the same sample.
"""

from __future__ import annotations

import pytest

from tests.factories import make_weighted_sample, WeightedSample


@pytest.fixture
def sample() -> WeightedSample:
    """Targets, predictions, strictly positive weights and a mask that excludes some."""
    return make_weighted_sample()
