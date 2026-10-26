import json
import os

import numpy as np
import pytest
from conftest import strummed_progression, sung_notes

from songpolish import cli, melody, pitch, thresholds
from songpolish.audio_io import save
from songpolish.pipeline import Inputs, process
from songpolish.settings import Settings


def test_defaults_are_the_engine_constants_and_in_range():
    d = thresholds.defaults()
    assert d["tolerance_cents"] == pitch.TOLERANCE
    for t in thresholds.THRESHOLDS:
        assert t.lo <= t.to_shown(d[t.name]) <= t.hi, t.name


def test_applied_overrides_then_restores():
    before = pitch.TOLERANCE, melody.WIDE_MAX_MOVE
    with thresholds.applied({"tolerance_cents": 0.2, "phrase_max_move": 2.0}):
        assert (pitch.TOLERANCE, melody.WIDE_MAX_MOVE) == (0.2, 2.0)
    assert (pitch.TOLERANCE, melody.WIDE_MAX_MOVE) == before
    with pytest.raises(RuntimeError), thresholds.applied({"tolerance_cents": 0.2}):
        raise RuntimeError
    assert pitch.TOLERANCE == before[0]


@pytest.mark.parametrize(
    "overrides",
    [
        {"nope": 1.0},
        {"tolerance_cents": 0.9},  # 90 cents, range is 0..50
        {"boost_min_snr_db": 50.0, "boost_full_snr_db": 40.0},
    ],
)
def test_invalid_overrides_are_rejected(overrides):
    with pytest.raises(ValueError):
        thresholds.validate(overrides)


def test_cli_values_are_in_shown_units():
    assert cli.parse_overrides(["tolerance_cents=20", "leak_max_of_voice=25"]) == {
        "tolerance_cents": pytest.approx(0.2),
        "leak_max_of_voice": pytest.approx(0.25),
    }
    with pytest.raises(ValueError):
        cli.parse_overrides(["tolerance_cents"])


def test_threshold_changes_the_result_and_is_reported(tmp_path, out_dir):
    guitar = strummed_progression(repeats=2)
    vocal = sung_notes([60.15, 64.15, 67.15, 64.15])  # 15 cents sharp
    vocal = np.pad(vocal, ((0, 0), (0, guitar.shape[1] - vocal.shape[1])))
    save(tmp_path / "v.wav", vocal)
    save(tmp_path / "g.wav", guitar)
    inputs = Inputs(vocal=tmp_path / "v.wav", guitar=tmp_path / "g.wav")

    default = process(inputs, Settings(), out_dir / "a").report
    assert default["pitch"]["notes_corrected"] > 0
    loose = process(inputs, Settings(thresholds={"tolerance_cents": 0.3}), out_dir / "b").report
    assert loose["pitch"]["notes_corrected"] == 0
    assert any("Leave notes within 30 cents" in n for n in loose["notes"])
    saved = json.loads((out_dir / "b" / "v_report.json").read_text())
    assert saved["settings"]["thresholds"] == {"tolerance_cents": 0.3}
    assert pitch.TOLERANCE == thresholds.defaults()["tolerance_cents"]


def test_gui_setup_roundtrip(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from songpolish.gui.app import MainWindow

    QSettings.setPath(QSettings.Format.NativeFormat, QSettings.Scope.UserScope, str(tmp_path))
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    _app = QApplication.instance() or QApplication([])
    w = MainWindow()
    assert w.current_settings().thresholds == {}  # untouched panel = engine defaults
    w.thresholds.boxes["tolerance_cents"].setValue(15)
    w.thresholds.boxes["key_change_cost"].setValue(6)
    w.key_mode.setChecked(True)
    s = w.current_settings()
    assert s.thresholds == {"tolerance_cents": pytest.approx(0.15), "key_change_cost": 6.0}
    setup = json.loads(json.dumps(w.setup()))

    w2 = MainWindow()
    w2.apply_setup(setup)
    assert w2.current_settings() == s
    w2.thresholds.set_values({})
    assert w2.current_settings().thresholds == {}
    w.close(), w2.close()


def test_settings_accept_plain_strings_from_uis():
    s = Settings(pitch_mode="key", note_targeting="phrase")
    assert s.note_targeting is melody.TargetStrategy.PHRASE
    assert s.to_dict()["note_targeting"] == "phrase"
