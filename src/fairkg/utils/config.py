"""YAML configuration loading and validation."""

import pathlib

import numpy as np
import yaml


def load_config(path: str | pathlib.Path) -> dict:
    """Load a YAML config file into a plain dict."""
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return {} if config is None else config


def to_plain(value):
    """Convert a value into basic types that YAML and JSON can store.

    Handles nested dicts, lists, tuples and sets, pathlib paths and numpy scalars.
    Raises TypeError for anything else, so an unreproducible value is never saved silently.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if isinstance(value, set):
        return [to_plain(v) for v in sorted(value)]
    if isinstance(value, pathlib.PurePath):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot store value of type {type(value).__name__}: {value!r}")
