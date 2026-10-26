"""Expert thresholds: the engine's tuning constants that are worth changing per recording.

The engine reads its thresholds as module-level constants (each with a comment saying
why it has its value). This registry names the ones a user may override, with a unit, a
safe range and an explanation, so the GUI and CLI can offer them without the DSP code
taking dozens of extra parameters. :func:`applied` sets the overrides for the duration of
one job. Jobs that change thresholds run one at a time (a lock), so two jobs in one
process never see each other's values.

Defaults were tuned on real takes; changing them is for diagnosing a specific recording.
"""

from __future__ import annotations

import importlib
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class Threshold:
    name: str  # key used in Settings.thresholds and on the command line
    module: str  # songpolish module holding the constant
    constant: str
    group: str
    label: str
    unit: str  # unit shown to the user
    scale: float  # shown value = engine value * scale (e.g. semitones -> cents)
    lo: float  # allowed range, in shown units
    hi: float
    decimals: int
    help: str

    def to_shown(self, value: float) -> float:
        return value * self.scale

    def from_shown(self, shown: float) -> float:
        return shown / self.scale


def _t(name, module, constant, group, label, unit, lo, hi, help, scale=1.0, decimals=1):
    return Threshold(name, module, constant, group, label, unit, scale, lo, hi, decimals, help)


NOTES, TARGETS, KEYS, LEAK, EQ, TUNING = (
    "Sung notes", "Wrong notes", "Key changes", "Ghost voice", "Tone (EQ)", "Tuning",
)  # fmt: skip

THRESHOLDS: tuple[Threshold, ...] = (
    _t("tolerance_cents", "pitch", "TOLERANCE", NOTES, "Leave notes within", "cents", 0, 50,
       "Notes this close to their target are already in tune and are not touched.",
       scale=100, decimals=0),
    _t("phantom_db", "pitch", "PHANTOM_DB", NOTES, "Ignore faint notes below the singing",
       "dB", 10, 80, "The separated voice holds faint guitar bleed where nobody sings. Notes "
       "this far below the loud notes are treated as bleed, not singing.", decimals=0),
    _t("min_note_ms", "pitch", "MIN_NOTE_S", NOTES, "Shortest note to correct", "ms", 40, 400,
       "Shorter pieces are transitions between notes and are left alone.", scale=1000,
       decimals=0),
    _t("max_note_spread", "pitch", "MAX_NOTE_SPREAD", NOTES, "Most pitch wobble in a note",
       "semitones", 0.3, 1.5, "A 'note' that wobbles more than this (standard deviation) is a "
       "glide or a slide and is left alone.", decimals=2),
    _t("nearest_max_move", "melody", "HARMONY_MAX_MOVE", TARGETS,
       "Nearest note: furthest move", "semitones", 0.5, 2.0,
       "With the 'nearest note' strategy, a note never moves further than this.", decimals=2),
    _t("in_key_bonus", "melody", "HARMONY_IN_KEY_BONUS", TARGETS, "Preference for in-key notes",
       "", 0, 4, "Higher: out-of-key notes must be sung more accurately to be kept as "
       "intentional. 1.5 keeps them only within ~18 cents.", decimals=2),
    _t("phrase_max_move", "melody", "WIDE_MAX_MOVE", TARGETS, "Phrase: furthest move",
       "semitones", 1, 5, "With the 'phrase' strategy, how far a note that clashes with the "
       "chord may be moved to fit the melody.", decimals=1),
    _t("phrase_gap_s", "melody", "PHRASE_GAP_S", TARGETS, "Pause that ends a phrase", "s",
       0.2, 2.0, "Notes separated by a longer silence belong to different phrases.",
       decimals=2),
    _t("clash_weight", "melody", "CLASH_WEIGHT", TARGETS, "Chord clash threshold", "",
       0.1, 0.8, "A note whose pitch class has less than this weight in the guitar's chord "
       "clashes and may be moved by 'phrase'. Higher moves more notes.", decimals=2),
    _t("guide_penalty", "melody", "GUIDE_PENALTY", TARGETS, "Follow the guitar's direction",
       "", 0, 8, "How strongly 'phrase' avoids a melody that moves against the guitar's line "
       "(the high-low-high vs low-mid-high case).", decimals=1),
    _t("repeat_max_move", "melody", "REPEAT_MAX_MOVE", TARGETS, "Repetition: furthest move",
       "semitones", 1, 5, "With 'repetition', how far a note may move to match how the other "
       "repetitions of the phrase were sung.", decimals=1),
    _t("modulated_max_move", "melody", "MODULATED_MAX_MOVE", TARGETS,
       "Repeat in a new key: furthest move", "semitones", 1, 6,
       "After a key change, how far a note may move to follow the earlier repetition "
       "transposed into the new key.", decimals=1),
    _t("key_change_cost", "analysis", "KEY_CHANGE_COST", KEYS, "Reluctance to change key", "",
       0.5, 15, "Higher: only long, clear modulations count as a key change. Lower: shorter "
       "passages in another key are followed too (and a single chord may be mistaken for "
       "one).", decimals=1),
    _t("key_window_s", "analysis", "KEY_WINDOW_S", KEYS, "Key detection window", "s", 4, 20,
       "Length of music used to judge the key at each moment.", decimals=0),
    _t("leak_min_shift", "pitch", "LEAK_MIN_SHIFT", LEAK, "Reclaim voice on moves of at least",
       "semitones", 0.2, 2.0, "Leftover voice in the guitar stem is moved with the note only "
       "for notes moved at least this far (on fine-tuned notes the guitar often plays the "
       "same note).", decimals=2),
    _t("leak_max_of_voice", "pitch", "LEAK_MAX_OF_VOICE", LEAK, "Most taken from the guitar",
       "% of voice", 0, 100, "Never take more than this from the guitar stem, relative to the "
       "voice. Higher removes more ghost voice but can pitch-shift guitar notes (robotic).",
       scale=100, decimals=0),
    _t("leak_depth", "pitch", "LEAK_DEPTH", LEAK, "Fraction of leftover voice moved", "%",
       0, 100, "How much of the voice found in the guitar stem is moved.", scale=100,
       decimals=0),
    _t("max_boost_db", "tone", "MAX_BOOST_DB", EQ, "Largest EQ boost", "dB", 0, 12,
       "Upper limit for any band the automatic EQ raises.", decimals=1),
    _t("max_cut_db", "tone", "MAX_CUT_DB", EQ, "Largest EQ cut", "dB", 0, 15,
       "Upper limit for any band the automatic EQ lowers (e.g. phone boom).", decimals=1),
    _t("boost_min_snr_db", "tone", "BOOST_MIN_SNR_DB", EQ, "No boost below SNR", "dB", 10, 60,
       "Bands with less signal-to-noise than this are never boosted: it would raise hiss "
       "('old recorder').", decimals=0),
    _t("boost_full_snr_db", "tone", "BOOST_FULL_SNR_DB", EQ, "Full boost from SNR", "dB", 15, 70,
       "Bands with at least this signal-to-noise may get the full boost. Must be above "
       "'no boost below SNR'.", decimals=0),
    _t("retune_from_cents", "pipeline", "TUNING_FIX_THRESHOLD_CENTS", TUNING,
       "Retune to A=440 when off by", "cents", 0, 50,
       "The whole song is retuned only when the guitar is at least this far from A=440 "
       "(and the option is on).", decimals=0),
)  # fmt: skip

BY_NAME: dict[str, Threshold] = {t.name: t for t in THRESHOLDS}
GROUPS: tuple[str, ...] = tuple(dict.fromkeys(t.group for t in THRESHOLDS))

_lock = threading.RLock()
_defaults: dict[str, float] | None = None


def _module(t: Threshold):
    return importlib.import_module(f"songpolish.{t.module}")


def defaults() -> dict[str, float]:
    """Engine default of every threshold, in engine units."""
    global _defaults
    with _lock:
        if _defaults is None:
            _defaults = {t.name: float(getattr(_module(t), t.constant)) for t in THRESHOLDS}
        return dict(_defaults)


def validate(overrides: Mapping[str, float]) -> None:
    """Raise ValueError for unknown names, values out of range or inconsistent pairs."""
    for name, value in overrides.items():
        t = BY_NAME.get(name)
        if t is None:
            raise ValueError(f"Unknown threshold {name!r}. Choose from: {', '.join(BY_NAME)}")
        shown = t.to_shown(float(value))
        if not t.lo - 1e-9 <= shown <= t.hi + 1e-9:
            raise ValueError(f"{t.label}: {shown:g} {t.unit} is outside {t.lo:g}..{t.hi:g}")
    values = defaults() | dict(overrides)
    if values["boost_full_snr_db"] <= values["boost_min_snr_db"]:
        raise ValueError("'Full boost from SNR' must be above 'No boost below SNR'.")


def changed(overrides: Mapping[str, float]) -> dict[str, float]:
    """Only the overrides that differ from the defaults."""
    base = defaults()
    return {k: float(v) for k, v in overrides.items() if abs(float(v) - base[k]) > 1e-12}


def describe(overrides: Mapping[str, float]) -> str:
    """'Leave notes within 15 cents, ...' in shown units."""
    return ", ".join(
        f"{BY_NAME[k].label} {BY_NAME[k].to_shown(v):g} {BY_NAME[k].unit}".rstrip()
        for k, v in overrides.items()
    )


@contextmanager
def applied(overrides: Mapping[str, float]) -> Iterator[None]:
    """Run the enclosed engine code with these thresholds (engine units), then restore."""
    validate(overrides)
    if not overrides:
        yield
        return
    with _lock:
        defaults()  # cache the real defaults before anything is overridden
        saved = []
        try:
            for name, value in overrides.items():
                t = BY_NAME[name]
                module = _module(t)
                saved.append((module, t.constant, getattr(module, t.constant)))
                setattr(module, t.constant, float(value))
            yield
        finally:
            for module, constant, value in reversed(saved):
                setattr(module, constant, value)
