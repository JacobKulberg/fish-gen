"""Open the fishgen designer:  py gui.py

Works with any Python 3.10+. If this Python doesn't have fishgen's dependencies,
it hands over to the project's .venv, creating it (and installing numpy + Pillow)
on the first run.
"""

import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_PY = VENV / ("Scripts/python.exe" if sys.platform.startswith("win") else "bin/python")


def _has_deps() -> bool:
    try:
        import numpy  # noqa: F401
        import PIL  # noqa: F401
        import tkinter  # noqa: F401
    except ImportError:
        return False
    return True


def _venv_has_deps() -> bool:
    return subprocess.call([str(VENV_PY), "-c", "import numpy, PIL, tkinter"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def main() -> int:
    if sys.version_info < (3, 10):
        print("fishgen needs Python 3.10 or newer.")
        return 1
    if _has_deps():
        sys.path.insert(0, str(ROOT))
        from fishgen.cli import main as cli_main

        return cli_main(["gui", *sys.argv[1:]])

    if Path(sys.executable).resolve() == VENV_PY.resolve():
        print("The .venv is missing packages. Try:  .venv\\Scripts\\pip install -e .")
        return 1
    if not VENV_PY.exists():
        print("First run: creating .venv ...")
        venv.create(VENV, with_pip=True)
    if not _venv_has_deps():
        print("Installing numpy and Pillow into .venv (one time) ...")
        rc = subprocess.call([str(VENV_PY), "-m", "pip", "install", "--disable-pip-version-check", "-q", "-e", str(ROOT)])
        if rc != 0:
            print("Install failed - see the pip output above.")
            return rc
    return subprocess.call([str(VENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
