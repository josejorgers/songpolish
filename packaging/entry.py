"""Entry point of the installed desktop app (PyInstaller bundle).

Sets up what a frozen app lacks before the engine is imported: a writable numba cache
(librosa's compiled functions are cached next to their source, which a bundle doesn't
have), a log file instead of a console (a windowed app has no stdout/stderr), and the
multiprocessing guard Windows and macOS need for frozen apps.

``SongPolish --self-test [recording] [out_dir]`` checks an installed build without a
display or network: it opens the window, then polishes the recording (or a synthetic
voice + guitar) with separation and prints the report. CI runs it on every platform.
"""

import multiprocessing
import os
import sys
from pathlib import Path


def _data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    path = base / "SongPolish"
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    multiprocessing.freeze_support()
    data = _data_dir()
    os.environ.setdefault("NUMBA_CACHE_DIR", str(data / "numba-cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the model is bundled
    if sys.stderr is None or sys.stdout is None:
        log = open(data / "songpolish.log", "a", buffering=1, encoding="utf-8")  # noqa: SIM115
        sys.stdout = sys.stdout or log
        sys.stderr = sys.stderr or log

    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        from self_test import run

        sys.exit(run(sys.argv[2:]))

    from songpolish.gui.app import main

    sys.exit(main())
