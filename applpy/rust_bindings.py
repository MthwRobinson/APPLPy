"""Helpers for APPLPy Rust extension bindings."""

from functools import lru_cache
from importlib import import_module


@lru_cache(maxsize=1)
def _extension_module():
    try:
        return import_module("applpy_rust")
    except ImportError as exc:
        raise ImportError(
            "applpy_rust extension is not built. "
            "Run `uv sync --extra rust` then "
            "`uv run --no-sync maturin develop -m rust/Cargo.toml`."
        ) from exc


def dummy_ping():
    """Return a constant string from the Rust extension."""
    return _extension_module().dummy_ping()


def discrete_order_stat(random_variable, n, r, replace):
    """Return the discrete order statistic random variable."""
    return _extension_module().discrete_order_stat(random_variable, n, r, replace)


def discrete_range_stat(random_variable, n, replace):
    """Return the discrete range statistic random variable."""
    return _extension_module().discrete_range_stat(random_variable, n, replace)


def discrete_maximum(random_variable_1, random_variable_2):
    """Return the discrete maximum of two random variables."""
    return _extension_module().discrete_maximum(random_variable_1, random_variable_2)


def discrete_minimum(random_variable_1, random_variable_2):
    """Return the discrete minimum of two random variables."""
    return _extension_module().discrete_minimum(random_variable_1, random_variable_2)


def next_combination(previous, n):
    """Return the next lexicographical combination."""
    return _extension_module().next_combination(previous, n)


def next_permutation(previous):
    """Return the next lexicographical permutation."""
    return _extension_module().next_permutation(previous)
