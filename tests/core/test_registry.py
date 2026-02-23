"""Tests for calibrax core registry.

Covers generic Registry[T], BenchmarkRegistry singleton, and
convenience functions/decorators for benchmark registration.
"""

import pytest

from calibrax.core.registry import (
    BenchmarkRegistry,
    get_benchmark,
    list_benchmarks,
    register_benchmark,
    Registry,
)


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
