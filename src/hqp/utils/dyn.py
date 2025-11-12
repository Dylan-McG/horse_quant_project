# src/hqp/utils/dyn.py

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from importlib import import_module
from typing import Any

PREFERRED_FUNC_NAMES = [
    # common
    "build",
    "run",
    "main",
    # ratings
    "build_ratings",
    "compute_ratings",
    "ratings",
    "blend",
    # features
    "build_features",
    "features",
    "compute_features",
]


def resolve_callable(candidates: list[str]) -> Callable[..., Any]:
    for spec in candidates:
        try:
            mod_path, fn_name = spec.split(":", 1)
            mod = import_module(mod_path)
            fn = getattr(mod, fn_name, None)
            if inspect.isfunction(fn) and fn.__module__ == mod.__name__:
                return fn
        except Exception:
            continue
    raise ImportError(f"No callable found among: {candidates}")


def find_callable_in_module(
    mod_name: str, prefer: Iterable[str] = PREFERRED_FUNC_NAMES
) -> Callable[..., Any] | None:
    try:
        mod = import_module(mod_name)
    except Exception:
        return None
    # 1) preferred names
    for name in prefer:
        fn = getattr(mod, name, None)
        if inspect.isfunction(fn) and fn.__module__ == mod.__name__:
            return fn
    # 2) first public, top-level function defined in the module
    for name, obj in inspect.getmembers(mod, inspect.isfunction):
        if not name.startswith("_") and obj.__module__ == mod.__name__:
            return obj
    return None
