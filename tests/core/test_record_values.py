"""Metadata values: JSON values plus the array scalars a run records, converted to JSON."""

from __future__ import annotations

import json
from datetime import datetime, UTC

import jax.numpy as jnp
import numpy as np
import pytest

from calibrax.core.record_values import aware, metadata_to_json, MetadataValue


def test_json_values_pass_through() -> None:
    value: MetadataValue = {"a": [1, 2.5, "x", True, None], "b": {"c": "d"}}

    assert metadata_to_json(value, "metadata") == value


def test_array_scalars_become_python_numbers_at_any_depth() -> None:
    value: MetadataValue = {
        "loss": jnp.float32(0.5),
        "steps": np.int64(1000),
        "flag": np.bool_(True),
        "history": (jnp.float32(1.0), 2.0),
    }

    converted = metadata_to_json(value, "metadata")

    assert converted == {"loss": 0.5, "steps": 1000, "flag": True, "history": [1.0, 2.0]}
    json.dumps(converted)
    assert isinstance(converted, dict)
    assert type(converted["steps"]) is int
    assert type(converted["flag"]) is bool


def test_an_array_of_several_elements_is_refused() -> None:
    with pytest.raises(ValueError, match="size 1"):
        metadata_to_json({"weights": jnp.ones(3)}, "metadata")


def test_a_complex_scalar_is_refused_naming_its_path() -> None:
    with pytest.raises(TypeError, match=r"^metadata\.z: .* has no JSON value"):
        metadata_to_json({"z": np.complex64(1 + 2j)}, "metadata")


class TestAware:
    """``aware``: a stored or given time is aware; a naive one is the writer's local time."""

    def test_an_aware_time_is_kept(self) -> None:
        moment = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)

        assert aware(moment) is moment

    def test_a_naive_time_is_read_as_local_time(self) -> None:
        naive = datetime(2026, 9, 19, 12, 0)

        result = aware(naive)

        assert result.tzinfo is not None
        assert result == naive.astimezone()
        assert result.replace(tzinfo=None) == naive
