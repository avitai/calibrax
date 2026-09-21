"""The values a record's free-form fields hold: JSON values and the array scalars a run measures.

A record's times are aware (:func:`aware`), so any two compare. A run's metadata,
environment or configuration often holds values straight from a
computation: a ``jnp.float32`` loss, a ``numpy.int64`` step count. They are kept as given and
converted when the record is written: anything with an ``item()`` method (JAX and NumPy
scalars) becomes the Python number it holds. A record read back from JSON holds JSON values,
which are metadata values too; :data:`Metadata` is the field type that reads them.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Annotated, cast, Protocol

from pydantic import PlainValidator, TypeAdapter, ValidationError
from substrax.typing import JsonValue


class SupportsItem(Protocol):
    """An array scalar: ``item()`` returns the Python value it holds."""

    def item(self) -> object:
        """The scalar as a Python value."""
        ...


def _array_scalar(value: object) -> SupportsItem:
    """An array scalar, recognised by carrying ``item()``.

    A bare protocol has no validator, and a union holding one cannot be built into a schema at
    all, so :data:`MetadataValue` names this one. It recognises the value rather than converting
    it, keeping a run's ``jnp.float32`` loss as the scalar it recorded.

    It asks what the value carries rather than what it is, because this module is on the light
    import path: a record's models read without loading JAX, and naming an array type here would
    end that.

    Args:
        value: The value.

    Returns:
        The value, unchanged.

    Raises:
        ValueError: If the value is not an array scalar.
    """
    if callable(getattr(value, "item", None)):
        return cast(SupportsItem, value)
    msg = f"{value!r} is not an array scalar"
    raise ValueError(msg)


type MetadataValue = (
    str
    | int
    | float
    | bool
    | None
    | Annotated[SupportsItem, PlainValidator(_array_scalar)]
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


@functools.cache
def _value_adapter(kind: object) -> TypeAdapter[object]:
    """One validator per type, built on first use (about 15 ms each)."""
    return TypeAdapter(kind)


def read_metadata[T](kind: type[T], value: object, name: str) -> T:
    """A record's free-form value as the type the caller expects.

    A field such as a run's ``config`` or ``metadata`` holds a :data:`MetadataValue`: the value
    a run recorded, or the JSON it was read back as. Reading one means saying what it should
    be, and being refused when it is not — a missing key and a key holding a string both reach
    the same arithmetic otherwise, and fail somewhere else.

    Array scalars are read as the numbers they hold, so ``jnp.int32(8)`` and ``8`` both read as
    ``8``.

    Args:
        kind: The type the value should have, such as ``int`` or ``list[int]``.
        value: The value, as the record holds it. It is whatever was stored, which is the
            reason to read it through here rather than to trust it.
        name: The value's path in the record, for the refusal (``"config.batch_size"``).

    Returns:
        The value as ``kind``.

    Raises:
        ValueError: If the value is not a ``kind``, naming its path and what was there.
    """
    try:
        return _value_adapter(kind).validate_python(value)  # type: ignore[return-value]
    except ValidationError as error:
        msg = f"{name}: expected {kind}, got {value!r}"
        raise ValueError(msg) from error


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


def aware(moment: datetime) -> datetime:
    """``moment`` with a timezone; a naive time is read as this machine's local time.

    Records stamped their time with ``datetime.now()``, which is naive local time, so a stored
    naive time is local time; it is converted, not relabelled, so the instant is kept.

    Args:
        moment: A time, aware or naive.

    Returns:
        The same instant, aware.
    """
    return moment if moment.tzinfo is not None else moment.astimezone()
