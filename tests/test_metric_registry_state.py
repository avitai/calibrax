"""``restored_metric_registry`` leaves the metric registry as it found it."""

from __future__ import annotations

from calibrax.metrics import MetricRegistry, register_metric
from calibrax.metrics.functional.regression import mae
from tests.metric_registry_state import restored_metric_registry


def test_a_metric_registered_inside_is_gone_after() -> None:
    with restored_metric_registry():
        register_metric("registered_inside")(mae)
        assert MetricRegistry().has("registered_inside")
    assert not MetricRegistry().has("registered_inside")


def test_a_built_in_removed_inside_is_back_after_as_the_same_entry() -> None:
    built_in = MetricRegistry().get("mse")
    with restored_metric_registry():
        MetricRegistry().remove("mse")
    assert MetricRegistry().get("mse") is built_in


def test_a_built_in_replaced_inside_is_restored_after() -> None:
    built_in = MetricRegistry().get("mse")
    with restored_metric_registry():
        MetricRegistry().remove("mse")
        register_metric("mse")(mae)
    assert MetricRegistry().get("mse") is built_in


def test_the_names_and_their_order_are_restored() -> None:
    names = MetricRegistry().list_names()
    with restored_metric_registry():
        MetricRegistry().clear()
    assert MetricRegistry().list_names() == names
