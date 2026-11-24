"""Root conftest for calibrax test suite."""

from pathlib import Path

import pytest

from tests.doc_examples import DOCS, keep_pages_in_order


pytest_plugins = ["pytester"]

# The documentation's Python code blocks, one item each (tests/doc_examples.py).
pytest_collect_file = DOCS.pytest()


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Run each documentation page's blocks together and in page order, whatever reordered them."""
    keep_pages_in_order(items)


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test outputs."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir
