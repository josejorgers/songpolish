"""Synthetic test signals: we need known pitch/tuning ground truth, not real recordings."""

import numpy as np
import pytest

from songpolish.audio_io import WORK_SR

SR = WORK_SR


def midi_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def sung_notes(
    notes_midi, note_s=0.6, gap_s=0.1, vibrato_cents=25.0, drift_cents=0.0, sr=SR, seed=0
):
    """A breathy voice-like harmonic tone per note, with vibrato. Returns (1, n).

    ``drift_cents``: pitch sags/creeps linearly by this much over each note's second half.
    """
    rng = np.random.default_rng(seed)
    out = []
    for m in notes_midi:
        n = int(note_s * sr)
        t = np.arange(n) / sr
        cents = vibrato_cents * np.sin(2 * np.pi * 5.5 * t)
        cents += drift_cents * np.clip(2 * t / note_s - 1, 0, 1)
        f = midi_hz(m) * 2 ** (cents / 1200)
        phase = 2 * np.pi * np.cumsum(f) / sr
        tone = sum((0.6 / k) * np.sin(k * phase) for k in range(1, 12))
        env = np.minimum(1, np.minimum(t / 0.04, (note_s - t) / 0.06))
        out.append(tone * env + 0.01 * rng.standard_normal(n))
        out.append(np.zeros(int(gap_s * sr)))
    return (0.3 * np.concatenate(out))[None, :].astype(np.float32)


def plucked_chord(notes_midi, dur_s=1.0, sr=SR, seed=1):
    """Strummed chord from decaying harmonics (exact pitch, unlike Karplus-Strong). (n,)"""
    rng = np.random.default_rng(seed)
    n = int(dur_s * sr)
    y = np.zeros(n)
    for i, m in enumerate(notes_midi):
        start = int(i * 0.015 * sr)
        t = np.arange(n - start) / sr
        f = midi_hz(m)
        for k in range(1, 16):
            if k * f > sr / 2.5:
                break
            decay = np.exp(-t * (2.0 + 0.8 * k))
            y[start:] += (0.8**k) * decay * np.sin(2 * np.pi * k * f * t + rng.uniform(0, 6.28))
    return y / len(notes_midi)


def strummed_progression(offset_cents=0.0, repeats=2, sr=SR):
    """C - G - Am - F, all shifted by offset_cents. Returns (1, n)."""
    c = offset_cents / 100
    chords = [
        [48, 52, 55, 60, 64],
        [43, 47, 50, 55, 59],
        [45, 52, 57, 60, 64],
        [41, 48, 53, 57, 60],
    ]
    parts = [plucked_chord([m + c for m in ch], 1.0, sr, seed=i) for i, ch in enumerate(chords)]
    return (0.4 * np.concatenate(parts * repeats))[None, :].astype(np.float32)


@pytest.fixture
def out_dir(tmp_path):
    return tmp_path / "out"
