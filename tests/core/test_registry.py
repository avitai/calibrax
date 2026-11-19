"""Tests for calibrax core registry.

Covers generic Registry[T], BenchmarkRegistry singleton, and
convenience functions/decorators for benchmark registration.
"""

import pytest

from calibrax.core import DuplicateRegistrationError as ExportedDuplicateRegistrationError
from calibrax.core.registry import (
    BenchmarkRegistry,
    DuplicateRegistrationError,
    get_benchmark,
    list_benchmarks,
    register_benchmark,
    Registry,
)
from calibrax.metrics import MetricRegistry, register_metric
from calibrax.metrics.functional.regression import mae, rmse


class TestRegistry:
    """Tests for generic Registry[T]."""

    def test_register_and_get(self) -> None:
        reg: Registry[str] = Registry()
        reg.register("greeting", "hello")
        assert reg.get("greeting") == "hello"

    def test_get_missing_raises_key_error(self) -> None:
        reg: Registry[str] = Registry()
        with pytest.raises(KeyError, match="missing"):
            reg.get("missing")

    def test_list_names_empty(self) -> None:
        reg: Registry[str] = Registry()
        assert reg.list_names() == []

    def test_list_names_populated(self) -> None:
        reg: Registry[str] = Registry()
        reg.register("a", "1")
        reg.register("b", "2")
        names = reg.list_names()
        assert "a" in names
        assert "b" in names
        assert len(names) == 2

    def test_has_returns_true_for_registered(self) -> None:
        reg: Registry[str] = Registry()
        reg.register("key", "val")
        assert reg.has("key") is True

    def test_has_returns_false_for_missing(self) -> None:
        reg: Registry[str] = Registry()
        assert reg.has("missing") is False

    def test_remove_existing(self) -> None:
        reg: Registry[str] = Registry()
        reg.register("key", "val")
        reg.remove("key")
        assert reg.has("key") is False

    def test_remove_missing_raises_key_error(self) -> None:
        reg: Registry[str] = Registry()
        with pytest.raises(KeyError, match="missing"):
            reg.remove("missing")

    def test_clear(self) -> None:
        reg: Registry[str] = Registry()
        reg.register("a", "1")
        reg.register("b", "2")
        reg.clear()
        assert len(reg) == 0

    def test_len(self) -> None:
        reg: Registry[str] = Registry()
        assert len(reg) == 0
        reg.register("a", "1")
        assert len(reg) == 1

    def test_contains(self) -> None:
        reg: Registry[str] = Registry()
        reg.register("a", "1")
        assert "a" in reg
        assert "b" not in reg

    def test_iter(self) -> None:
        reg: Registry[str] = Registry()
        reg.register("a", "1")
        reg.register("b", "2")
        assert set(reg) == {"a", "b"}


class TestBenchmarkRegistry:
    """Tests for BenchmarkRegistry singleton."""

    def setup_method(self) -> None:
        BenchmarkRegistry.reset()

    def test_singleton_same_instance(self) -> None:
        r1 = BenchmarkRegistry()
        r2 = BenchmarkRegistry()
        assert r1 is r2

    def test_register_and_get(self) -> None:
        reg = BenchmarkRegistry()
        reg.register("bench1", "value1")
        assert reg.get("bench1") == "value1"

    def test_reset_clears_state(self) -> None:
        reg = BenchmarkRegistry()
        reg.register("bench1", "value1")
        BenchmarkRegistry.reset()
        assert len(reg) == 0

    def test_list_names(self) -> None:
        reg = BenchmarkRegistry()
        reg.register("bench1", "v1")
        reg.register("bench2", "v2")
        assert set(reg.list_names()) == {"bench1", "bench2"}


class TestConvenienceFunctions:
    """Tests for register_benchmark, get_benchmark, list_benchmarks."""

    def setup_method(self) -> None:
        BenchmarkRegistry.reset()

    def test_decorator_registration(self) -> None:
        @register_benchmark("my_bench")
        class MyBench:
            pass

        assert get_benchmark("my_bench") is MyBench
        assert "my_bench" in list_benchmarks()

    def test_decorated_class_returned_unchanged(self) -> None:
        @register_benchmark("original")
        class Original:
            pass

        assert Original.__name__ == "Original"

    def test_get_benchmark_missing_raises(self) -> None:
        with pytest.raises(KeyError):
            get_benchmark("nonexistent")

    def test_list_benchmarks_empty(self) -> None:
        assert list_benchmarks() == []


class TestDuplicateRegistration:
    """A taken name is refused; replacing an item is ``remove`` then ``register``."""

    def setup_method(self) -> None:
        BenchmarkRegistry.reset()

    def test_a_taken_name_is_refused_and_the_first_item_stays(self) -> None:
        reg: Registry[str] = Registry()
        reg.register("a", "first")
        with pytest.raises(DuplicateRegistrationError):
            reg.register("a", "second")
        assert reg.get("a") == "first"
        assert len(reg) == 1

    def test_the_refusal_names_the_name_the_registry_type_and_the_replacement(self) -> None:
        reg: Registry[str] = Registry()
        reg.register("a", "first")
        with pytest.raises(
            DuplicateRegistrationError,
            match=r"^'a' is already registered in Registry; remove it first to replace it\.$",
        ):
            reg.register("a", "second")

    def test_the_refusal_is_a_value_error(self) -> None:
        assert issubclass(DuplicateRegistrationError, ValueError)

    def test_an_equal_item_registered_again_is_refused(self) -> None:
        reg: Registry[str] = Registry()
        reg.register("a", "same")
        with pytest.raises(DuplicateRegistrationError):
            reg.register("a", "same")

    def test_remove_then_register_replaces(self) -> None:
        reg: Registry[str] = Registry()
        reg.register("a", "first")
        reg.remove("a")
        reg.register("a", "second")
        assert reg.get("a") == "second"

    def test_a_singleton_registry_names_its_own_type(self) -> None:
        reg = BenchmarkRegistry()
        reg.register("bench", "first")
        with pytest.raises(DuplicateRegistrationError, match="in BenchmarkRegistry;"):
            reg.register("bench", "second")
        assert reg.get("bench") == "first"

    def test_reset_then_register_succeeds(self) -> None:
        BenchmarkRegistry().register("bench", "first")
        BenchmarkRegistry.reset()
        BenchmarkRegistry().register("bench", "second")
        assert get_benchmark("bench") == "second"

    def test_register_benchmark_twice_is_refused(self) -> None:
        @register_benchmark("twice")
        class First:
            pass

        with pytest.raises(DuplicateRegistrationError, match="'twice'"):

            @register_benchmark("twice")
            class Second:
                pass

        assert get_benchmark("twice") is First

    def test_register_metric_with_a_built_in_name_is_refused(self) -> None:
        built_in = MetricRegistry().get("mse")
        with pytest.raises(DuplicateRegistrationError, match="'mse' is already registered in"):
            register_metric("mse")(mae)
        assert MetricRegistry().get("mse") is built_in

    def test_register_metric_twice_is_refused(self) -> None:
        register_metric("planted_metric")(mae)
        with pytest.raises(DuplicateRegistrationError, match="in MetricRegistry;"):
            register_metric("planted_metric")(rmse)
        assert MetricRegistry().get("planted_metric").fn is mae

    def test_the_error_is_exported_from_calibrax_core(self) -> None:
        assert ExportedDuplicateRegistrationError is DuplicateRegistrationError
