"""Smoke test of an installed build (``SongPolish --self-test``): see entry.py."""

import json
import tempfile
from pathlib import Path

import numpy as np


def _synthetic(path: Path) -> Path:
    from songpolish.audio_io import WORK_SR, save

    t = np.arange(int(6 * WORK_SR)) / WORK_SR
    chord = sum(np.sin(2 * np.pi * f * t) * np.exp(-1.5 * (t % 1.5)) for f in (110, 220, 277, 330))
    voice = 0.3 * np.sin(2 * np.pi * 442 * t + 0.3 * np.sin(2 * np.pi * 5 * t))
    save(path, (0.2 * chord + voice * (t > 1))[None].astype(np.float32))
    return path


def run(args: list[str]) -> int:
    from PySide6.QtWidgets import QApplication

    from songpolish import separation
    from songpolish.gui.app import MainWindow
    from songpolish.pipeline import Inputs, process
    from songpolish.settings import PitchMode, Settings, TargetStrategy

    app = QApplication.instance() or QApplication(["SongPolish"])
    window = MainWindow()
    window.show()
    app.processEvents()
    window.close()
    print("window: ok")
    print("separation available:", separation.is_available(), separation.bundled_model_dir())
    if not separation.is_available() or separation.bundled_model_dir() is None:
        print("FAIL: separation or bundled model missing")
        return 1

    out = Path(args[1]) if len(args) > 1 else Path(tempfile.mkdtemp(prefix="songpolish-"))
    take = Path(args[0]) if args else _synthetic(out / "self_test.wav")
    settings = Settings(pitch_mode=PitchMode.KEY, note_targeting=TargetStrategy.PHRASE)
    result = process(Inputs(mix=take), settings, out)
    print(json.dumps({"output": str(result.output), **result.report}, indent=2, default=str))
    ok = result.output.is_file() and result.report["output_lufs"] > -30
    print("self-test:", "ok" if ok else "FAIL")
    return 0 if ok else 1
