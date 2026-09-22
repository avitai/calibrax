"""Metadata values: JSON values plus the array scalars a run records, converted to JSON."""

from __future__ import annotations

import json
from datetime import datetime, UTC

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from calibrax.core.record_values import (
    aware,
    metadata_to_json,
    MetadataValue,
    read_metadata,
    read_metadata_entry,
)


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


class TestReadMetadata:
    """Reading a record's free-form value back as the type the caller expects."""

    def test_a_json_number_reads_as_a_number(self) -> None:
        assert read_metadata(int, 3, "config.batch_size") == 3
        assert read_metadata(float, 2.5, "config.rate") == 2.5

    def test_an_array_scalar_reads_as_the_number_it_holds(self) -> None:
        assert read_metadata(int, np.int64(1000), "config.steps") == 1000
        assert read_metadata(int, jnp.int32(7), "config.depth") == 7
        assert read_metadata(float, jnp.float32(0.5), "metadata.loss") == pytest.approx(0.5)

    def test_a_sequence_reads_as_a_list(self) -> None:
        assert read_metadata(list[int], [2, 3, 4], "config.element_shape") == [2, 3, 4]

    def test_a_mapping_reads_as_a_mapping(self) -> None:
        extra = read_metadata(dict[str, MetadataValue], {"chain_depth": 3}, "config.extra")

        assert read_metadata(int, extra["chain_depth"], "config.extra.chain_depth") == 3

    def test_a_mapping_keeps_the_array_scalars_it_holds(self) -> None:
        """The arm that recognises array scalars must take them, not leave them to coercion.

        Every other arm of the union would accept ``jnp.float32(0.5)`` as a plain float, so a
        broken arm reads as working and only the scalar's type says otherwise.
        """
        read = read_metadata(
            dict[str, MetadataValue],
            {"loss": jnp.float32(0.5), "steps": np.int64(7)},
            "metadata",
        )

        assert isinstance(read["loss"], jax.Array)
        assert isinstance(read["steps"], np.integer)

    def test_a_value_that_is_no_metadata_value_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"metadata"):
            read_metadata(dict[str, MetadataValue], {"opaque": object()}, "metadata")

    def test_a_missing_value_is_refused_naming_its_path(self) -> None:
        with pytest.raises(ValueError, match=r"config\.batch_size"):
            read_metadata(int, None, "config.batch_size")

    def test_a_value_of_another_type_is_refused_naming_its_path(self) -> None:
        with pytest.raises(ValueError, match=r"config\.element_shape"):
            read_metadata(list[int], 256, "config.element_shape")

    def test_a_fractional_number_is_not_an_integer(self) -> None:
        with pytest.raises(ValueError, match=r"config\.batch_size"):
            read_metadata(int, 2.5, "config.batch_size")

    def test_the_refusal_says_what_was_expected_and_what_was_there(self) -> None:
        with pytest.raises(ValueError, match=r"config\.workers") as refusal:
            read_metadata(int, "many", "config.workers")

        assert "config.workers" in str(refusal.value)
        assert "'many'" in str(refusal.value)


class TestReadMetadataEntry:
    """Reading one entry of a free-form field, with an explicit default for its absence."""

    def test_a_recorded_entry_reads_as_the_kind(self) -> None:
        assert read_metadata_entry(float, {"execution_time": 3}, "execution_time") == 3.0

    def test_an_array_scalar_entry_reads_as_the_number_it_holds(self) -> None:
        assert read_metadata_entry(int, {"steps": jnp.int32(8)}, "steps") == 8

    def test_an_absent_entry_is_the_default(self) -> None:
        assert read_metadata_entry(float, {}, "execution_time", default=-1.0) == -1.0

    def test_an_entry_recorded_as_none_is_the_default(self) -> None:
        metadata: dict[str, MetadataValue] = {"execution_time": None}
        assert read_metadata_entry(float, metadata, "execution_time", default=-1.0) == -1.0

    def test_an_absent_entry_without_a_default_is_refused_naming_it(self) -> None:
        with pytest.raises(ValueError, match="execution_time"):
            read_metadata_entry(float, {}, "execution_time")

    def test_an_entry_recorded_as_none_without_a_default_is_refused(self) -> None:
        with pytest.raises(ValueError, match="execution_time"):
            read_metadata_entry(float, {"execution_time": None}, "execution_time")

    def test_a_value_of_the_wrong_kind_is_refused_even_with_a_default(self) -> None:
        with pytest.raises(ValueError, match="'fast'"):
            read_metadata_entry(float, {"execution_time": "fast"}, "execution_time", default=0.0)

    def test_the_refusal_names_the_path_the_caller_gives(self) -> None:
        with pytest.raises(ValueError, match=r"config\.dataset_size"):
            read_metadata_entry(
                int, {"dataset_size": 2.5}, "dataset_size", name="config.dataset_size"
            )
