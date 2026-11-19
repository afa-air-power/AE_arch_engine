"""Loader for the native AE core pybind11 module."""

from __future__ import annotations

import importlib
import pathlib
import sys
from types import ModuleType

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _candidate_dirs() -> list[pathlib.Path]:
    dirs = [ROOT / "build", ROOT / "build-cmake", ROOT / "cmake-build-debug", pathlib.Path.cwd() / "build"]
    seen: list[pathlib.Path] = []
    for directory in dirs:
        if directory.is_dir() and directory not in seen:
            seen.append(directory)
    return seen


def load_core() -> ModuleType:
    """Import ``ae_mind_pybind`` from an existing CMake build directory."""
    try:
        return importlib.import_module("ae_mind_pybind")
    except ImportError:
        pass

    for directory in _candidate_dirs():
        if not any(directory.glob("ae_mind_pybind*.so")):
            continue
        sys.path.insert(0, str(directory))
        try:
            return importlib.import_module("ae_mind_pybind")
        except ImportError:
            sys.path.pop(0)

    raise ImportError(
        "ae_mind_pybind is not built. Run:\n"
        "  cmake -S . -B build -DAE_BUILD_PYBIND11=ON\n"
        "  cmake --build build -j"
    )
