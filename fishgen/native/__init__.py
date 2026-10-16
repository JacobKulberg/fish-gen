"""Loads the C versions of the renderer's hot loops (``fast.c``) through ctypes.

``pip install`` builds the library as ``fishgen._fast``. In a source checkout without that
build it is compiled once with the system C compiler into a cache folder. If neither works,
``lib`` is None and callers fall back to their numpy implementations (same results, slower).
Set ``FISHGEN_NO_NATIVE=1`` to force the numpy path.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).with_name("fast.c")

_D = ctypes.POINTER(ctypes.c_double)
_I = ctypes.POINTER(ctypes.c_int32)
_i32, _i64, _f64 = ctypes.c_int32, ctypes.c_int64, ctypes.c_double

_SIGNATURES = {
    "fg_raster": [_D, _I, _I, _i64, _i32, _i32, _I, _D, _I, _D, _D],
    "fg_interp": [_D, _i32, _I, _I, _D, _D, _i64, _D],
    "fg_ssao": [_D, _i32, _i32, _f64, _D],
    "fg_hash": [_D, _D, _i64, _f64, _D],
    "fg_fbm": [_D, _D, _i64, _f64, _i32, _D],
    "fg_srgb": [_D, _i64],
    "fg_spines": [_D, _D, _i64, _i32, _D, _i32, _D, _D, _i32, _i32, _D, _I, _I, _I, _I],
}


def _built_extension() -> Path | None:
    """The library pip built next to the package, if any."""
    pkg = Path(__file__).resolve().parent.parent
    for suffix in (sysconfig.get_config_var("EXT_SUFFIX") or ".so", ".so", ".pyd"):
        p = pkg / f"_fast{suffix}"
        if p.exists():
            return p
    return None


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or (os.environ.get("LOCALAPPDATA") if sys.platform == "win32" else None)
    return Path(base or Path.home() / ".cache") / "fishgen"


def _compile() -> Path | None:
    """Compile fast.c with the system compiler into the cache (once per source version)."""
    src = SOURCE.read_bytes()
    ext = ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"
    out = _cache_dir() / f"fast-{hashlib.sha256(src).hexdigest()[:12]}{ext}"
    if out.exists():
        return out
    cc = os.environ.get("CC") or next((c for c in ("cc", "gcc", "clang") if shutil.which(c)), None)
    if cc is None:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        tmp_out = Path(tmp) / out.name
        cmd = [cc, "-O3", "-shared", "-fPIC", "-o", str(tmp_out), str(SOURCE), "-lm"]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        except (OSError, subprocess.SubprocessError):
            return None
        os.replace(tmp_out, out)  # atomic, so parallel first runs don't trip over each other
    return out


def _load():
    if os.environ.get("FISHGEN_NO_NATIVE"):
        return None
    for find in (_built_extension, _compile):
        try:
            path = find()
            if path is None:
                continue
            lib = ctypes.CDLL(str(path))
            for name, args in _SIGNATURES.items():
                fn = getattr(lib, name)
                fn.argtypes, fn.restype = args, None
            return lib
        except (OSError, AttributeError):
            continue
    return None


lib = _load()


def f64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def i32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.int32)


def ptr(a: np.ndarray, kind=_D):
    return a.ctypes.data_as(kind)


def iptr(a: np.ndarray):
    return a.ctypes.data_as(_I)
