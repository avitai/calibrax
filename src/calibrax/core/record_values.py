"""The values a record's free-form fields hold: JSON values and the array scalars a run measures.

A run's metadata, environment or configuration often holds values straight from a
computation: a ``jnp.float32`` loss, a ``numpy.int64`` step count. They are kept as given and
converted when the record is written: anything with an ``item()`` method (JAX and NumPy
scalars) becomes the Python number it holds. A record read back from JSON holds JSON values,
which are metadata values too; :data:`Metadata` is the field type that reads them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Protocol

from pydantic import PlainValidator, TypeAdapter, ValidationError
from substrax.typing import JsonValue


class SupportsItem(Protocol):
    """An array scalar: ``item()`` returns the Python value it holds."""

    def item(self) -> object:
        """The scalar as a Python value."""
        ...


type MetadataValue = (
    str
    | int
    | float
    | bool
    | None
    | SupportsItem
    | Sequence[MetadataValue]
    | Mapping[str, MetadataValue]
)


_JSON_OBJECT: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])


def _metadata_from_json(value: object) -> dict[str, MetadataValue]:
    """A free-form field read from JSON: an object of JSON values."""
    return dict(_JSON_OBJECT.validate_python(value, strict=True))


type Metadata = Annotated[dict[str, MetadataValue], PlainValidator(_metadata_from_json)]
"""A record's free-form field: metadata values in memory, a JSON object when read back."""


def require_stored(record_type: type[object], data: Mapping[str, JsonValue], *keys: str) -> None:
    """Refuse a stored record that lacks a field whose default is a fresh value.

    A record's id or time defaults to a new value when the record is built; read from JSON,
    the same default would invent one, so a stored record must carry the field.

    Args:
        record_type: The record's dataclass, named in the error.
        data: The JSON object.
        *keys: The fields the object must hold.

    Raises:
        ValidationError: Listing each missing field as pydantic's ``missing`` error.
    """
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValidationError.from_exception_data(
            record_type.__name__,
            [{"type": "missing", "loc": (key,), "input": data} for key in missing],
        )


def metadata_to_json(value: MetadataValue, name: str) -> JsonValue:
    """The JSON form of a metadata value; array scalars become Python numbers.

    Args:
        value: The value.
        name: Its path in the record, for errors.

    Returns:
        The value with every array scalar replaced by its Python number, tuples as lists.

    Raises:
        TypeError: If an array scalar holds a value JSON cannot (a complex number).
    """
    if value is None or isinstance(value, str | int | float):
        return value
    if isinstance(value, Mapping):
        return {str(key): metadata_to_json(item, f"{name}.{key}") for key, item in value.items()}
    if isinstance(value, Sequence):
        return [metadata_to_json(item, f"{name}[{index}]") for index, item in enumerate(value)]
    scalar = value.item()
    if not isinstance(scalar, str | int | float):
        msg = f"{name}: {value!r} has no JSON value"
        raise TypeError(msg)
    return scalar
