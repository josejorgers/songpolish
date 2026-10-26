"""Desktop app (PySide6). A thin shell over :func:`songpolish.pipeline.process`.

No DSP lives here: the window collects recordings + settings (including the expert
thresholds of :mod:`songpolish.thresholds`), runs the pipeline on a worker thread one song
after another, and offers a loudness-matched A/B player for every edited song.
"""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, QStandardPaths, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from songpolish import separation, thresholds
from songpolish.melody import TargetStrategy
from songpolish.pipeline import Inputs, Result, process
from songpolish.settings import DEFAULT_PRESET, PRESETS, PitchMode, Settings

AUDIO_EXTENSIONS = ("*.wav", "*.flac", "*.mp3", "*.m4a", "*.aac", "*.ogg", "*.aiff", "*.aif")
AUDIO_FILTER = f"Audio ({' '.join(AUDIO_EXTENSIONS)});;All files (*)"
SETUP_FILTER = "SongPolish setup (*.json)"
LOUDNESS_CHOICES = [("Streaming (-14 LUFS)", -14.0), ("Quiet / dynamic (-16 LUFS)", -16.0),
                    ("Loud (-12 LUFS)", -12.0), ("Very loud (-11 LUFS)", -11.0)]  # fmt: skip

TARGETING_CHOICES = [
    (TargetStrategy.NEAREST, "Nearest note (never changes the melody)"),
    (TargetStrategy.PHRASE, "Follow the phrase and the guitar (up to 3 semitones)"),
    (TargetStrategy.REPETITION, "Match how other verses were sung"),
]

# (settings field, label, tooltip) for the 0..1 sliders.
AMOUNT_SLIDERS = [
    ("pitch_correction", "Pitch correction", "How far sung notes move toward the right pitch. "
     "Vibrato and slides are always kept. 0 turns pitch correction off."),
    ("noise_reduction", "Noise cleanup", "Removes hiss, hum and room noise."),
    ("tone_match", "Tone", "Automatic EQ toward a balanced, studio-like tone."),
    ("compression", "Evenness", "Smooths out loud and quiet moments."),
    ("reverb", "Space", "Adds a small natural room."),
]  # fmt: skip


def default_out_dir() -> Path:
    downloads = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DownloadLocation)
    return Path(downloads or Path.home()) / "polished"


class Worker(QObject):
    """Runs the jobs one after another; a failed song does not stop the others."""

    progress = Signal(int, str, float)
    song_done = Signal(int, object)
    song_failed = Signal(int, str)
    all_done = Signal()

    def __init__(self, jobs: list[Inputs], settings: Settings, out_dir: Path, compare: bool):
        super().__init__()
        self.jobs, self.settings, self.out_dir, self.compare = jobs, settings, out_dir, compare
        self.stop_requested = False

    @Slot()
    def run(self) -> None:
        strategies = list(TargetStrategy) if self.compare else None
        for i, inputs in enumerate(self.jobs):
            if self.stop_requested:
                break
            try:
                result = process(
                    inputs,
                    self.settings,
                    self.out_dir,
                    lambda stage, f, i=i: self.progress.emit(i, stage, f),
                    strategies=strategies,
                )
            except Exception as e:  # surfaced to the user, never crash the UI
                traceback.print_exc()
                self.song_failed.emit(i, f"{type(e).__name__}: {e}")
            else:
                self.song_done.emit(i, result)
        self.all_done.emit()


class FileRow(QWidget):
    changed = Signal()

    def __init__(self, placeholder: str, folder: bool = False):
        super().__init__()
        self.folder = folder
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
        if self.folder:
            path = QFileDialog.getExistingDirectory(self, "Save edited songs in", self.edit.text())
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Choose recording", "", AUDIO_FILTER)
        if path:
            self.edit.setText(path)

    def path(self) -> Path | None:
        text = self.edit.text().strip()
        return Path(text).expanduser() if text else None


class SongList(QListWidget):
    """Recordings to polish; files can be added with the button or dropped on the list."""

    def __init__(self):
        super().__init__()
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setAcceptDrops(True)
        self.setMinimumHeight(60)
        self.setMaximumHeight(130)

    def add(self, paths) -> None:
        present = set(self.paths())
        for p in map(Path, paths):
            if p.is_file() and p not in present:
                item = QListWidgetItem(p.name)
                item.setData(Qt.ItemDataRole.UserRole, str(p))
                item.setToolTip(str(p))
                self.addItem(item)
                present.add(p)

    def paths(self) -> list[Path]:
        return [Path(self.item(i).data(Qt.ItemDataRole.UserRole)) for i in range(self.count())]

    def remove_selected(self) -> None:
        for item in self.selectedItems():
            self.takeItem(self.row(item))

    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    dragMoveEvent = dragEnterEvent

    def dropEvent(self, e) -> None:
        self.add(u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile())
        e.acceptProposedAction()


class ThresholdPanel(QWidget):
    """One spin box per engine threshold, grouped in tabs. Changed values are bold."""

    def __init__(self):
        super().__init__()
        self.defaults = thresholds.defaults()
        self.boxes: dict[str, QDoubleSpinBox] = {}
        self.labels: dict[str, QLabel] = {}
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        intro = QLabel(
            "The engine's tuned limits. Defaults were set on real recordings; change them "
            "to fix something specific you hear, one at a time. Hover for an explanation."
        )
        intro.setWordWrap(True)
        v.addWidget(intro)
        tabs = QTabWidget()
        for group in thresholds.GROUPS:
            page = QWidget()
            form = QFormLayout(page)
            for t in thresholds.THRESHOLDS:
                if t.group != group:
                    continue
                box = QDoubleSpinBox()
                box.setRange(t.lo, t.hi)
                box.setDecimals(t.decimals)
                box.setSingleStep(10**-t.decimals if t.decimals else 1)
                if t.unit:
                    box.setSuffix(f" {t.unit}")
                default = t.to_shown(self.defaults[t.name])
                box.setToolTip(f"{t.help}\n\nDefault: {default:g} {t.unit}".rstrip())
                box.valueChanged.connect(lambda _=0, n=t.name: self._mark(n))
                label = QLabel(t.label)
                label.setToolTip(box.toolTip())
                self.boxes[t.name], self.labels[t.name] = box, label
                form.addRow(label, box)
            tabs.addTab(page, group)
        v.addWidget(tabs)
        reset = QPushButton("Reset thresholds to defaults")
        reset.clicked.connect(lambda: self.set_values({}))
        v.addWidget(reset, alignment=Qt.AlignmentFlag.AlignLeft)
        self.set_values({})

    def _shown_default(self, name: str) -> float:
        t = thresholds.BY_NAME[name]
        return round(t.to_shown(self.defaults[name]), t.decimals)

    def _mark(self, name: str) -> None:
        font = QFont(self.labels[name].font())
        font.setBold(self.boxes[name].value() != self._shown_default(name))
        self.labels[name].setFont(font)

    def values(self) -> dict[str, float]:
        """Changed thresholds only, in engine units."""
        return {
            name: thresholds.BY_NAME[name].from_shown(box.value())
            for name, box in self.boxes.items()
            if box.value() != self._shown_default(name)
        }

    def set_values(self, overrides: dict[str, float]) -> None:
        for name, box in self.boxes.items():
            t = thresholds.BY_NAME[name]
            box.setValue(t.to_shown(overrides.get(name, self.defaults[name])))
            self._mark(name)


class ABPlayer(QGroupBox):
    """Two players; switching keeps the playback position so you compare the same moment."""

    def __init__(self):
        super().__init__("Compare (both at the same loudness)")
        self.players: dict[str, QMediaPlayer] = {}
        self.outputs: list[QAudioOutput] = []
        self.current: str | None = None
        self.buttons: dict[str, QPushButton] = {}
        v = QVBoxLayout(self)
        self.title = QLabel("Select an edited song.")
        self.title.setWordWrap(True)
        v.addWidget(self.title)
        row = QHBoxLayout()
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
        v.addLayout(row)
        self.setEnabled(False)

    def load(self, title: str, original: Path, polished: Path) -> None:
        position = self.players[self.current].position() if self.current else 0
        playing = self.current
        self.stop()
        self.title.setText(title)
        self.players["original"].setSource(QUrl.fromLocalFile(str(original)))
        self.players["polished"].setSource(QUrl.fromLocalFile(str(polished)))
        self.setEnabled(True)
        if playing:  # switching between versions of one song keeps the moment
            self.players[playing].setPosition(position)

    def play(self, key: str) -> None:
        position = self.players[self.current].position() if self.current else None
        for k, p in self.players.items():
            if k != key:
                p.pause()
            self.buttons[k].setChecked(k == key)
        if position is not None:
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
        self.resize(640, 860)
        self.thread: QThread | None = None
        self.worker: Worker | None = None
        self.jobs: list[Inputs] = []
        self.sliders: dict[str, QSlider] = {}
        self.store = QSettings("SongPolish", "SongPolish")

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.addWidget(self._build_inputs())
        layout.addWidget(self._build_settings())
        layout.addWidget(self._build_advanced())
        layout.addWidget(self._build_run())
        layout.addWidget(self._build_results())
        layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        self.setCentralWidget(scroll)

        self.restore()
        self._update_mode()

    # --- layout -----------------------------------------------------------------

    def _build_inputs(self) -> QGroupBox:
        box = QGroupBox("Recordings")
        v = QVBoxLayout(box)
        self.one_take = QRadioButton("Takes with voice and guitar recorded together (one or more)")
        self.two_tracks = QRadioButton("Separate tracks: a vocal file and a guitar file")
        self.one_take.setChecked(True)
        group = QButtonGroup(box)
        for b in (self.one_take, self.two_tracks):
            group.addButton(b)
            b.toggled.connect(self._update_mode)
            v.addWidget(b)

        self.songs_widget = QWidget()
        sv = QVBoxLayout(self.songs_widget)
        sv.setContentsMargins(0, 0, 0, 0)
        self.songs = SongList()
        sv.addWidget(self.songs)
        buttons = QHBoxLayout()
        add = QPushButton("Add recordings…")
        add.clicked.connect(self.add_songs)
        remove = QPushButton("Remove")
        remove.clicked.connect(self.songs.remove_selected)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.songs.clear)
        for b in (add, remove, clear):
            buttons.addWidget(b)
        buttons.addStretch()
        sv.addLayout(buttons)
        hint = QLabel("Tip: you can also drop audio files on the list.")
        hint.setEnabled(False)
        sv.addWidget(hint)
        v.addWidget(self.songs_widget)

        self.vocal_row = FileRow("Vocal track")
        self.guitar_row = FileRow("Guitar track")
        v.addWidget(self.vocal_row)
        v.addWidget(self.guitar_row)

        form = QFormLayout()
        self.out_row = FileRow("Folder for the edited songs", folder=True)
        form.addRow("Save in", self.out_row)
        v.addLayout(form)
        if not separation.is_available():
            note = QLabel(
                "ℹ Voice/guitar separation is not installed, so a single take gets cleanup "
                "and mastering but no vocal pitch correction. Install the 'separation' extra."
            )
            note.setWordWrap(True)
            v.addWidget(note)
        return box

    def _build_settings(self) -> QGroupBox:
        box = QGroupBox("Sound")
        form = QFormLayout(box)
        self.preset = QComboBox()
        for name in PRESETS:
            self.preset.addItem(name.capitalize(), name)
        self.preset.activated.connect(lambda _: self.apply_settings(PRESETS[self.preset_name()]))
        form.addRow("Style", self.preset)

        for field, label, tip in AMOUNT_SLIDERS:
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(0, 100)
            s.setToolTip(tip)
            value = QLabel()
            value.setMinimumWidth(32)
            s.valueChanged.connect(lambda x, lbl=value: lbl.setText(f"{x}%"))
            value.setText(f"{s.value()}%")
            row = QHBoxLayout()
            row.addWidget(s)
            row.addWidget(value)
            self.sliders[field] = s
            form.addRow(label, row)

        self.balance = QSlider(Qt.Orientation.Horizontal)
        self.balance.setRange(-6, 6)
        self.balance.setToolTip(
            "Left: guitar louder · Right: voice louder. On a single take, raising the voice "
            "adds separated voice on top, with its separation artefacts: 0 keeps the take."
        )
        balance_value = QLabel()
        balance_value.setMinimumWidth(32)
        self.balance.valueChanged.connect(lambda x: balance_value.setText(f"{x:+d} dB"))
        balance_value.setText("+0 dB")
        row = QHBoxLayout()
        row.addWidget(self.balance)
        row.addWidget(balance_value)
        form.addRow("Voice ↔ guitar", row)

        self.loudness = QComboBox()
        for label, value in LOUDNESS_CHOICES:
            self.loudness.addItem(label, value)
        form.addRow("Loudness", self.loudness)

        self.targeting = QComboBox()
        for strategy, label in TARGETING_CHOICES:
            self.targeting.addItem(label, strategy)
        self.targeting.setToolTip("What to do with sung notes that clash with the guitar's chord")
        form.addRow("Wrong notes", self.targeting)

        self.key_mode = QCheckBox("Stick to the key (tighter; fixes accurately sung wrong notes)")
        self.fix_tuning = QCheckBox("Tune the whole song to A=440 if the guitar was off")
        self.compare = QCheckBox("Also make one version per wrong-note strategy, to compare")
        for c in (self.key_mode, self.fix_tuning, self.compare):
            form.addRow(c)
        return box

    def _build_advanced(self) -> QWidget:
        wrapper = QWidget()
        v = QVBoxLayout(wrapper)
        v.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.advanced_btn = QPushButton("▸ Advanced thresholds")
        self.advanced_btn.setCheckable(True)
        self.advanced_btn.setFlat(True)
        self.advanced_btn.toggled.connect(self._toggle_advanced)
        row.addWidget(self.advanced_btn)
        row.addStretch()
        save = QPushButton("Save setup…")
        save.setToolTip("Save every setting and threshold to a file, to reuse or share")
        save.clicked.connect(self.save_setup)
        load = QPushButton("Load setup…")
        load.clicked.connect(self.load_setup)
        row.addWidget(save)
        row.addWidget(load)
        v.addLayout(row)
        self.thresholds = ThresholdPanel()
        self.thresholds.setVisible(False)
        v.addWidget(self.thresholds)
        return wrapper

    def _build_run(self) -> QWidget:
        wrapper = QWidget()
        v = QVBoxLayout(wrapper)
        v.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.polish_btn = QPushButton("✨ Polish")
        self.polish_btn.setMinimumHeight(40)
        self.polish_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("Stop after this song")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.request_stop)
        row.addWidget(self.polish_btn, 3)
        row.addWidget(self.stop_btn, 1)
        v.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.stage = QLabel("Add a recording to start.")
        self.stage.setWordWrap(True)
        v.addWidget(self.progress)
        v.addWidget(self.stage)
        return wrapper

    def _build_results(self) -> QGroupBox:
        box = QGroupBox("Edited songs")
        v = QVBoxLayout(box)
        self.results = QListWidget()
        self.results.setMinimumHeight(60)
        self.results.setMaximumHeight(130)
        self.results.currentItemChanged.connect(self.show_result)
        self.results.itemDoubleClicked.connect(lambda item: self.open_folder())
        v.addWidget(self.results)
        self.ab = ABPlayer()
        v.addWidget(self.ab)
        self.report = QLabel()
        self.report.setWordWrap(True)
        self.report.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.report)
        self.open_btn = QPushButton("Open output folder")
        self.open_btn.clicked.connect(self.open_folder)
        v.addWidget(self.open_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        return box

    # --- state ------------------------------------------------------------------

    def _update_mode(self) -> None:
        one = self.one_take.isChecked()
        self.songs_widget.setVisible(one)
        self.vocal_row.setVisible(not one)
        self.guitar_row.setVisible(not one)

    def _toggle_advanced(self, shown: bool) -> None:
        self.thresholds.setVisible(shown)
        self.advanced_btn.setText(("▾" if shown else "▸") + " Advanced thresholds")

    def preset_name(self) -> str:
        return self.preset.currentData()

    def apply_settings(self, s: Settings, keep_thresholds: bool = True) -> None:
        """Show ``s`` in the widgets. Presets leave the thresholds as they are."""
        for field, slider in self.sliders.items():
            slider.setValue(round(getattr(s, field) * 100))
        self.balance.setValue(round(s.vocal_balance_db))
        idx = min(
            range(len(LOUDNESS_CHOICES)), key=lambda i: abs(LOUDNESS_CHOICES[i][1] - s.target_lufs)
        )
        self.loudness.setCurrentIndex(idx)
        self.fix_tuning.setChecked(s.fix_reference_tuning)
        self.key_mode.setChecked(s.pitch_mode is PitchMode.KEY)
        self.targeting.setCurrentIndex(
            next(i for i, (t, _) in enumerate(TARGETING_CHOICES) if t is s.note_targeting)
        )
        if not keep_thresholds:
            self.thresholds.set_values(s.thresholds)

    def current_settings(self) -> Settings:
        return replace(
            PRESETS[self.preset_name()],
            **{f: s.value() / 100 for f, s in self.sliders.items()},
            vocal_balance_db=float(self.balance.value()),
            target_lufs=float(self.loudness.currentData()),
            fix_reference_tuning=self.fix_tuning.isChecked(),
            pitch_mode=PitchMode.KEY if self.key_mode.isChecked() else PitchMode.HARMONY,
            note_targeting=TargetStrategy(self.targeting.currentData()),
            thresholds=self.thresholds.values(),
        )

    def setup(self) -> dict:
        return {
            "preset": self.preset_name(),
            "settings": self.current_settings().to_dict(),
            "compare": self.compare.isChecked(),
        }

    def apply_setup(self, data: dict) -> None:
        settings = Settings.from_dict(data["settings"])
        thresholds.validate(settings.thresholds)
        name = data.get("preset", DEFAULT_PRESET)
        self.preset.setCurrentIndex(max(0, self.preset.findData(name)))
        self.apply_settings(settings, keep_thresholds=False)
        self.compare.setChecked(bool(data.get("compare", False)))

    def restore(self) -> None:
        """The last setup, songs list excluded; defaults on first run or a bad entry."""
        self.preset.setCurrentIndex(list(PRESETS).index(DEFAULT_PRESET))
        self.apply_settings(PRESETS[DEFAULT_PRESET], keep_thresholds=False)
        self.out_row.edit.setText(str(self.store.value("out_dir", str(default_out_dir()))))
        saved = self.store.value("setup")
        if saved:
            try:
                self.apply_setup(json.loads(saved))
            except (ValueError, KeyError, TypeError):
                self.store.remove("setup")
        if self.store.value("advanced") in (True, "true"):
            self.advanced_btn.setChecked(True)

    def remember(self) -> None:
        self.store.setValue("setup", json.dumps(self.setup()))
        self.store.setValue("out_dir", self.out_row.edit.text())
        self.store.setValue("advanced", self.advanced_btn.isChecked())

    def closeEvent(self, event) -> None:
        self.remember()
        if self.worker:
            self.worker.stop_requested = True
        super().closeEvent(event)

    def current_jobs(self) -> list[Inputs]:
        if self.one_take.isChecked():
            return [Inputs(mix=p) for p in self.songs.paths()]
        return [Inputs(vocal=self.vocal_row.path(), guitar=self.guitar_row.path())]

    # --- actions ----------------------------------------------------------------

    def add_songs(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Choose recordings", "", AUDIO_FILTER)
        self.songs.add(paths)

    def save_setup(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save setup", "songpolish_setup.json",
                                              SETUP_FILTER)  # fmt: skip
        if path:
            Path(path).write_text(json.dumps(self.setup(), indent=2))

    def load_setup(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load setup", "", SETUP_FILTER)
        if not path:
            return
        try:
            self.apply_setup(json.loads(Path(path).read_text()))
        except (OSError, ValueError, KeyError, TypeError) as e:
            QMessageBox.warning(self, "SongPolish", f"Could not load {Path(path).name}: {e}")

    def start(self) -> None:
        jobs = self.current_jobs()
        try:
            if not jobs:
                raise ValueError("Add at least one recording.")
            for inputs in jobs:
                inputs.validate()
                for p in (inputs.mix, inputs.vocal, inputs.guitar):
                    if p and not p.is_file():
                        raise ValueError(f"File not found: {p}")
            out_dir = self.out_row.path()
            if out_dir is None:
                raise ValueError("Choose a folder for the edited songs.")
            settings = self.current_settings()
            thresholds.validate(settings.thresholds)
        except ValueError as e:
            QMessageBox.warning(self, "SongPolish", str(e))
            return
        self.remember()

        self.jobs = jobs
        self.ab.stop()
        self.polish_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.thread = QThread(self)
        self.worker = Worker(jobs, settings, out_dir, self.compare.isChecked())
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.song_done.connect(self.on_song_done)
        self.worker.song_failed.connect(self.on_song_failed)
        self.worker.all_done.connect(self.on_all_done)
        self.worker.all_done.connect(self.thread.quit)
        self.thread.start()

    def request_stop(self) -> None:
        if self.worker:
            self.worker.stop_requested = True
        self.stop_btn.setEnabled(False)
        self.stage.setText(self.stage.text() + " (stopping after this song)")

    @Slot(int, str, float)
    def on_progress(self, i: int, stage: str, fraction: float) -> None:
        n = len(self.jobs)
        self.progress.setValue(round(100 * (i + fraction) / n))
        prefix = f"{i + 1}/{n} {self.jobs[i].name}: " if n > 1 else f"{self.jobs[i].name}: "
        self.stage.setText(prefix + (stage + "…" if fraction < 1 else stage))

    @Slot(int, object)
    def on_song_done(self, i: int, result: Result) -> None:
        name = self.jobs[i].name
        versions = [("", result.output)] if not result.variants else list(result.variants.items())
        first = None
        for version, path in versions:
            label = f"✓ {name}" + (f"  ·  {version}" if version else "")
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, (result, version, path))
            item.setToolTip(str(path))
            self.results.addItem(item)
            first = first or item
        self.results.setCurrentItem(first)

    @Slot(int, str)
    def on_song_failed(self, i: int, message: str) -> None:
        item = QListWidgetItem(f"✗ {self.jobs[i].name}: failed")
        item.setToolTip(message)
        self.results.addItem(item)
        QMessageBox.critical(self, "SongPolish", f"{self.jobs[i].name}: {message}")

    @Slot()
    def on_all_done(self) -> None:
        self.polish_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.worker = None
        self.progress.setValue(100)
        self.stage.setText(f"Done. Edited songs are in {self.out_row.path()}")

    def show_result(self, item: QListWidgetItem | None, _previous=None) -> None:
        data = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not data:
            return
        result, version, path = data
        self.ab.load(item.text().lstrip("✓ "), result.original_matched, path)
        report = dict(result.report)
        if version:
            report.update(report["strategies"][version])
        self.report.setText(f"Done in {report['seconds']} s → {path.name}\n" + describe(report))

    def open_folder(self) -> None:
        item = self.results.currentItem()
        data = item.data(Qt.ItemDataRole.UserRole) if item else None
        folder = data[2].parent if data else self.out_row.path()
        if folder:
            folder.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))


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
