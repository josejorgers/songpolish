"""Desktop app (PySide6). A thin shell over :func:`songpolish.pipeline.process`.

No DSP lives here: the window collects inputs + settings, runs the pipeline on a
worker thread and offers a loudness-matched A/B player for the result.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from songpolish import separation
from songpolish.melody import TargetStrategy
from songpolish.pipeline import Inputs, Result, process
from songpolish.settings import DEFAULT_PRESET, PRESETS, PitchMode, Settings

AUDIO_FILTER = "Audio (*.wav *.flac *.mp3 *.m4a *.aac *.ogg *.aiff *.aif);;All files (*)"
LOUDNESS_CHOICES = [("Streaming (-14 LUFS)", -14.0), ("Quiet / dynamic (-16 LUFS)", -16.0),
                    ("Loud (-11 LUFS)", -11.0)]  # fmt: skip

TARGETING_CHOICES = [
    (TargetStrategy.NEAREST, "Nearest note (never changes the melody)"),
    (TargetStrategy.PHRASE, "Follow the phrase and the guitar (up to 3 semitones)"),
    (TargetStrategy.REPETITION, "Match how other verses were sung"),
]

# (settings field, label, tooltip) for the 0..1 sliders.
AMOUNT_SLIDERS = [
    ("pitch_correction", "Pitch correction", "How far sung notes move toward the right pitch. "
     "Vibrato and slides are always kept."),
    ("noise_reduction", "Noise cleanup", "Removes hiss, hum and room noise."),
    ("tone_match", "Tone", "Automatic EQ toward a balanced, studio-like tone."),
    ("compression", "Evenness", "Smooths out loud and quiet moments."),
    ("reverb", "Space", "Adds a small natural room."),
]  # fmt: skip


class Worker(QObject):
    progress = Signal(str, float)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, inputs: Inputs, settings: Settings, out_dir: Path):
        super().__init__()
        self.inputs, self.settings, self.out_dir = inputs, settings, out_dir

    @Slot()
    def run(self) -> None:
        try:
            result = process(self.inputs, self.settings, self.out_dir, self.progress.emit)
        except Exception as e:  # surfaced to the user, never crash the UI
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")
        else:
            self.finished.emit(result)


class FileRow(QWidget):
    changed = Signal()

    def __init__(self, placeholder: str):
        super().__init__()
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.edit.textChanged.connect(self.changed)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.edit)
        row.addWidget(browse)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose recording", "", AUDIO_FILTER)
        if path:
            self.edit.setText(path)

    def path(self) -> Path | None:
        text = self.edit.text().strip()
        return Path(text) if text else None


class ABPlayer(QGroupBox):
    """Two players; switching keeps the playback position so you compare the same moment."""

    def __init__(self):
        super().__init__("Compare (both at the same loudness)")
        self.players: dict[str, QMediaPlayer] = {}
        self.outputs: list[QAudioOutput] = []
        self.current: str | None = None
        self.buttons: dict[str, QPushButton] = {}
        row = QHBoxLayout(self)
        for key, label in (("original", "▶ Original"), ("polished", "▶ Polished")):
            player, out = QMediaPlayer(self), QAudioOutput(self)
            player.setAudioOutput(out)
            self.players[key] = player
            self.outputs.append(out)
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _=False, k=key: self.play(k))
            self.buttons[key] = btn
            row.addWidget(btn)
        stop = QPushButton("■ Stop")
        stop.clicked.connect(self.stop)
        row.addWidget(stop)
        self.setEnabled(False)

    def load(self, original: Path, polished: Path) -> None:
        self.stop()
        self.players["original"].setSource(QUrl.fromLocalFile(str(original)))
        self.players["polished"].setSource(QUrl.fromLocalFile(str(polished)))
        self.setEnabled(True)

    def play(self, key: str) -> None:
        position = self.players[self.current].position() if self.current else 0
        for k, p in self.players.items():
            if k != key:
                p.pause()
            self.buttons[k].setChecked(k == key)
        self.players[key].setPosition(position)
        self.players[key].play()
        self.current = key

    def stop(self) -> None:
        for k, p in self.players.items():
            p.stop()
            self.buttons[k].setChecked(False)
        self.current = None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SongPolish")
        self.setMinimumWidth(560)
        self.thread: QThread | None = None
        self.result: Result | None = None
        self.sliders: dict[str, QSlider] = {}

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addWidget(self._build_inputs())
        layout.addWidget(self._build_settings())

        self.polish_btn = QPushButton("✨ Polish my recording")
        self.polish_btn.setMinimumHeight(40)
        self.polish_btn.clicked.connect(self.start)
        layout.addWidget(self.polish_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.stage = QLabel("Choose a recording to start.")
        layout.addWidget(self.progress)
        layout.addWidget(self.stage)

        self.ab = ABPlayer()
        layout.addWidget(self.ab)
        self.report = QLabel()
        self.report.setWordWrap(True)
        self.report.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.report)
        self.open_btn = QPushButton("Open output folder")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self.open_folder)
        layout.addWidget(self.open_btn)
        layout.addStretch()
        self.setCentralWidget(root)

        self.apply_preset(DEFAULT_PRESET)
        self._update_mode()

    # --- layout -----------------------------------------------------------------

    def _build_inputs(self) -> QGroupBox:
        box = QGroupBox("Recording")
        v = QVBoxLayout(box)
        self.one_take = QRadioButton("One take: voice and guitar recorded together")
        self.two_tracks = QRadioButton("Separate tracks: a vocal file and a guitar file")
        self.one_take.setChecked(True)
        group = QButtonGroup(box)
        for b in (self.one_take, self.two_tracks):
            group.addButton(b)
            b.toggled.connect(self._update_mode)
            v.addWidget(b)
        self.take_row = FileRow("Recording (wav, mp3, m4a, flac…)")
        self.vocal_row = FileRow("Vocal track")
        self.guitar_row = FileRow("Guitar track")
        for r in (self.take_row, self.vocal_row, self.guitar_row):
            v.addWidget(r)
        if not separation.is_available():
            hint = QLabel(
                "ℹ Voice/guitar separation is not installed, so a single take gets cleanup "
                "and mastering but no vocal pitch correction. Install the 'separation' extra."
            )
            hint.setWordWrap(True)
            v.addWidget(hint)
        return box

    def _build_settings(self) -> QGroupBox:
        box = QGroupBox("Sound")
        form = QFormLayout(box)
        self.preset = QComboBox()
        for name in PRESETS:
            self.preset.addItem(name.capitalize(), name)
        self.preset.setCurrentIndex(list(PRESETS).index(DEFAULT_PRESET))
        self.preset.currentIndexChanged.connect(
            lambda _: self.apply_preset(self.preset.currentData())
        )
        form.addRow("Style", self.preset)

        for field, label, tip in AMOUNT_SLIDERS:
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(0, 100)
            s.setToolTip(tip)
            self.sliders[field] = s
            form.addRow(label, s)

        self.balance = QSlider(Qt.Orientation.Horizontal)
        self.balance.setRange(-6, 6)
        self.balance.setToolTip("Left: guitar louder · Right: voice louder")
        form.addRow("Voice ↔ guitar", self.balance)

        self.loudness = QComboBox()
        for label, value in LOUDNESS_CHOICES:
            self.loudness.addItem(label, value)
        form.addRow("Loudness", self.loudness)

        self.targeting = QComboBox()
        for strategy, label in TARGETING_CHOICES:
            self.targeting.addItem(label, strategy)
        self.targeting.setToolTip("What to do with sung notes that clash with the guitar's chord")
        form.addRow("Wrong notes", self.targeting)

        self.fix_tuning = QCheckBox("Tune the whole song to A=440 if the guitar was off")
        self.key_mode = QCheckBox("Stick to the song's key (tighter, less natural)")
        form.addRow(self.fix_tuning)
        form.addRow(self.key_mode)
        return box

    # --- state ------------------------------------------------------------------

    def _update_mode(self) -> None:
        one = self.one_take.isChecked()
        self.take_row.setVisible(one)
        self.vocal_row.setVisible(not one)
        self.guitar_row.setVisible(not one)

    def apply_preset(self, name: str) -> None:
        s = PRESETS[name]
        for field, slider in self.sliders.items():
            slider.setValue(round(getattr(s, field) * 100))
        self.balance.setValue(round(s.vocal_balance_db))
        idx = next((i for i, (_, v) in enumerate(LOUDNESS_CHOICES) if v == s.target_lufs), 0)
        self.loudness.setCurrentIndex(idx)
        self.fix_tuning.setChecked(s.fix_reference_tuning)
        self.key_mode.setChecked(s.pitch_mode is PitchMode.KEY)
        self.targeting.setCurrentIndex(
            next(i for i, (t, _) in enumerate(TARGETING_CHOICES) if t is s.note_targeting)
        )

    def current_settings(self) -> Settings:
        base = PRESETS[self.preset.currentData()]
        return replace(
            base,
            **{f: s.value() / 100 for f, s in self.sliders.items()},
            vocal_balance_db=float(self.balance.value()),
            target_lufs=float(self.loudness.currentData()),
            fix_reference_tuning=self.fix_tuning.isChecked(),
            pitch_mode=PitchMode.KEY if self.key_mode.isChecked() else PitchMode.HARMONY,
            note_targeting=self.targeting.currentData(),
        )

    def current_inputs(self) -> Inputs:
        if self.one_take.isChecked():
            return Inputs(mix=self.take_row.path())
        return Inputs(vocal=self.vocal_row.path(), guitar=self.guitar_row.path())

    # --- actions ----------------------------------------------------------------

    def start(self) -> None:
        inputs = self.current_inputs()
        try:
            inputs.validate()
            for p in (inputs.mix, inputs.vocal, inputs.guitar):
                if p and not p.is_file():
                    raise ValueError(f"File not found: {p}")
        except ValueError as e:
            QMessageBox.warning(self, "SongPolish", str(e))
            return
        out_dir = Path(inputs.mix or inputs.vocal).parent / "polished"

        self.ab.stop()
        self.polish_btn.setEnabled(False)
        self.report.clear()
        self.thread = QThread(self)
        self.worker = Worker(inputs, self.current_settings(), out_dir)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        for sig in (self.worker.finished, self.worker.failed):
            sig.connect(self.thread.quit)
        self.thread.start()

    @Slot(str, float)
    def on_progress(self, stage: str, fraction: float) -> None:
        self.progress.setValue(round(fraction * 100))
        self.stage.setText(stage + "…" if fraction < 1 else stage)

    @Slot(object)
    def on_finished(self, result: Result) -> None:
        self.result = result
        self.polish_btn.setEnabled(True)
        self.open_btn.setEnabled(True)
        self.ab.load(result.original_matched, result.output)
        self.stage.setText(f"Done in {result.report['seconds']} s → {result.output.name}")
        self.report.setText(describe(result.report))

    @Slot(str)
    def on_failed(self, message: str) -> None:
        self.polish_btn.setEnabled(True)
        self.stage.setText("Something went wrong.")
        QMessageBox.critical(self, "SongPolish", message)

    def open_folder(self) -> None:
        if self.result:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.result.output.parent)))


def describe(r: dict) -> str:
    lines = []
    cents = r["reference_tuning_cents"]
    if r["retuned_to_a440"]:
        lines.append(f"• Guitar was {abs(cents):.0f} cents {'sharp' if cents > 0 else 'flat'}; "
                     "the song is now at A=440.")  # fmt: skip
    else:
        lines.append(f"• Tuning reference: {cents:+.0f} cents from A=440 (left as is).")
    if "pitch_summary" in r:
        lines.append(f"• Voice: {r['pitch_summary']}")
    if "key" in r:
        lines.append(f"• Key: {r['key']}")
    if "key_changes" in r:
        lines.append(f"• Key changes: {', '.join(r['key_changes'])}")
    lines.append(f"• Loudness: {r['input_lufs']} → {r['output_lufs']} LUFS")
    lines += [f"• {n}" for n in r["notes"]]
    return "\n".join(lines)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("SongPolish")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
