"""Helpers shared by the explanatory notebooks: synthetic sounds, playback and plots.

Synthetic sounds are used on purpose: we know exactly which pitch they have, so every
measurement in the notebooks can be checked against the truth.
"""

from __future__ import annotations

import librosa
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio, display

SR = 44100  # samples per second, the same rate the app works at
PLAY_SR = 22050  # playback copies are downsampled to keep the notebooks small

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_hz(m: float) -> float:
    """MIDI note number -> frequency. 69 is the A above middle C, 440 Hz."""
    return 440.0 * 2 ** ((m - 69) / 12)


def note_name(m: float) -> str:
    m = int(round(m))
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def play(audio: np.ndarray, sr: int = SR, label: str | None = None) -> None:
    """Show an audio player. Accepts (n,) or (channels, n)."""
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=0)
    y = librosa.resample(y, orig_sr=sr, target_sr=PLAY_SR) if sr != PLAY_SR else y
    if label:
        print(label)
    display(Audio(y, rate=PLAY_SR, normalize=True))


def sine(freq: float, seconds: float, amp: float = 0.3, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def voice_like(
    midi_pitches,
    note_s: float = 0.5,
    gap_s: float = 0.08,
    vibrato_cents: float = 25.0,
    drift_cents: float = 0.0,
    amp: float = 0.3,
    sr: int = SR,
    seed: int = 0,
) -> np.ndarray:
    """A crude sung melody: harmonics with falling strength, vibrato and a little breath.

    ``midi_pitches`` may be fractional: 60.3 is a C sung 30 cents sharp.
    ``drift_cents`` makes each note sag (or rise) over its second half.
    Returns (1, n) like the app's audio.
    """
    rng = np.random.default_rng(seed)
    out = []
    for m in midi_pitches:
        n = int(note_s * sr)
        t = np.arange(n) / sr
        cents = vibrato_cents * np.sin(2 * np.pi * 5.5 * t)
        cents += drift_cents * np.clip(2 * t / note_s - 1, 0, 1)
        f = midi_to_hz(m) * 2 ** (cents / 1200)
        phase = 2 * np.pi * np.cumsum(f) / sr
        tone = sum((0.6 / k) * np.sin(k * phase) for k in range(1, 12))
        env = np.minimum(1, np.minimum(t / 0.04, (note_s - t) / 0.06))
        out.append(tone * env + 0.01 * rng.standard_normal(n))
        out.append(np.zeros(int(gap_s * sr)))
    return (amp * np.concatenate(out))[None, :].astype(np.float32)


def plucked_chord(midi_notes, seconds: float = 1.0, sr: int = SR, seed: int = 1) -> np.ndarray:
    """A strummed chord made of decaying harmonics (exactly in tune). Returns (n,)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    y = np.zeros(n)
    for i, m in enumerate(midi_notes):
        start = int(i * 0.015 * sr)  # strings are hit one after another
        t = np.arange(n - start) / sr
        f = midi_to_hz(m)
        for k in range(1, 16):
            if k * f > sr / 2.5:
                break
            decay = np.exp(-t * (2.0 + 0.8 * k))
            y[start:] += (0.8**k) * decay * np.sin(2 * np.pi * k * f * t + rng.uniform(0, 6.28))
    return (y / len(midi_notes)).astype(np.float32)


CHORDS = {  # guitar-like voicings, as MIDI numbers
    "A": [45, 52, 57, 61, 64],
    "D": [50, 57, 62, 66, 69],
    "E": [40, 47, 52, 56, 59, 64],
    "F#m": [42, 49, 54, 57, 61],
    "C": [48, 52, 55, 60, 64],
    "G": [43, 47, 50, 55, 59, 67],
    "Am": [45, 52, 57, 60, 64],
    "F": [41, 48, 53, 57, 60, 65],
}


def progression(names, seconds_each: float = 1.0, amp: float = 0.4, sr: int = SR) -> np.ndarray:
    """Strum the named chords one after another. Returns (1, n)."""
    parts = [plucked_chord(CHORDS[c], seconds_each, sr, seed=i) for i, c in enumerate(names)]
    return (amp * np.concatenate(parts))[None, :].astype(np.float32)


def fit(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cut two (1, n) signals to the same length so they can be added."""
    n = min(a.shape[-1], b.shape[-1])
    return a[..., :n], b[..., :n]


# --- plots -------------------------------------------------------------------------


def plot_wave(y: np.ndarray, sr: int = SR, start_s: float = 0.0, seconds: float | None = None,
              title: str = "", ax=None):  # fmt: skip
    y = np.asarray(y).reshape(-1) if np.asarray(y).ndim == 1 else np.asarray(y).mean(axis=0)
    a = int(start_s * sr)
    b = len(y) if seconds is None else a + int(seconds * sr)
    t = np.arange(a, min(b, len(y))) / sr
    ax = ax or plt.subplots(figsize=(10, 2.5))[1]
    ax.plot(t, y[a:b], lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    ax.set_title(title)
    return ax


def plot_spectrum(y: np.ndarray, sr: int = SR, fmax: float = 5000, title: str = "", ax=None):
    y = np.asarray(y).reshape(-1) if np.asarray(y).ndim == 1 else np.asarray(y).mean(axis=0)
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    keep = freqs <= fmax
    ax = ax or plt.subplots(figsize=(10, 3))[1]
    ax.plot(freqs[keep], 20 * np.log10(spec[keep] / spec.max() + 1e-9), lw=0.8)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("level (dB, 0 = strongest)")
    ax.set_ylim(-80, 5)
    ax.set_title(title)
    return ax


def plot_spectrogram(y: np.ndarray, sr: int = SR, fmax: float = 5000, title: str = "", ax=None):
    y = np.asarray(y).reshape(-1) if np.asarray(y).ndim == 1 else np.asarray(y).mean(axis=0)
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048, hop_length=256)), ref=np.max)
    ax = ax or plt.subplots(figsize=(10, 3.5))[1]
    img = librosa.display.specshow(S, sr=sr, hop_length=256, x_axis="time", y_axis="hz", ax=ax,
                                   vmin=-80, vmax=0)  # fmt: skip
    ax.set_ylim(0, fmax)
    ax.set_title(title)
    plt.colorbar(img, ax=ax, format="%+.0f dB")
    return ax


import librosa.display  # noqa: E402  (registers specshow)
