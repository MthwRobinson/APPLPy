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


def next_combination(previous, n):
    """Return the next lexicographical combination."""
    return _extension_module().next_combination(previous, n)


def next_permutation(previous):
    """Return the next lexicographical permutation."""
    return _extension_module().next_permutation(previous)


def verify_discrete_pdf(function, tolerance=1e-6):
    """Verify that the area under a discrete PDF sums to 1"""
    return _extension_module().verify_discrete_pdf(function, tolerance)
