"""Balancing the sources, adding space, and mastering to a loudness target."""

from __future__ import annotations

import numpy as np
from pedalboard import BrickwallLimiter, Compressor, Pedalboard, Reverb

from songpolish.analysis import active_rms_db, loudness_lufs
from songpolish.audio_io import WORK_SR, match_length, to_stereo

STEM_LUFS = -20.0  # working level of each source before summing


def gain_to_lufs(audio: np.ndarray, target: float, sr: int = WORK_SR) -> np.ndarray:
    current = loudness_lufs(audio, sr)
    if current <= -69.0:
        return audio
    return (audio * 10 ** ((target - current) / 20)).astype(np.float32)


def add_space(audio: np.ndarray, amount: float, vocal: bool, sr: int = WORK_SR) -> np.ndarray:
    """A small, natural room. Voice gets a little more than guitar."""
    stereo = to_stereo(audio)
    if amount <= 0:
        return stereo
    wet = (0.06 + 0.22 * amount) * (1.0 if vocal else 0.7)
    reverb = Reverb(
        room_size=0.35 + 0.3 * amount,
        damping=0.6,
        wet_level=wet,
        dry_level=1.0 - 0.3 * wet,
        width=1.0,
    )
    return np.asarray(reverb(stereo, sr), dtype=np.float32)


def mix(
    vocal: np.ndarray, guitar: np.ndarray, vocal_balance_db: float, reverb: float, sr: int = WORK_SR
) -> np.ndarray:
    v = gain_to_lufs(vocal, STEM_LUFS + vocal_balance_db / 2, sr)
    g = gain_to_lufs(guitar, STEM_LUFS - vocal_balance_db / 2, sr)
    v = add_space(v, reverb, vocal=True, sr=sr)
    g = add_space(g, reverb, vocal=False, sr=sr)
    v, g = match_length(v, g)
    return v + g


def loudness_match(
    audio: np.ndarray, target_lufs: float, true_peak_db: float, sr: int = WORK_SR
) -> np.ndarray:
    """Bring audio to a loudness with only a transparent peak limiter, no other processing."""
    return master(audio, target_lufs, true_peak_db, glue=0.0, sr=sr)


def master(
    audio: np.ndarray,
    target_lufs: float,
    true_peak_db: float,
    glue: float = 0.5,
    sr: int = WORK_SR,
) -> np.ndarray:
    """Gentle bus compression, then loudness normalisation under a true-peak ceiling."""
    x = to_stereo(audio)
    if glue > 0:
        level = active_rms_db(x, sr)
        x = Compressor(
            threshold_db=level + 2.0, ratio=1.2 + 0.8 * glue, attack_ms=30.0, release_ms=200.0
        )(x, sr)
    # The limiter eats a little loudness, so converge in a couple of passes.
    gain_db = target_lufs - loudness_lufs(x, sr)
    limiter = Pedalboard([BrickwallLimiter(ceiling_db=true_peak_db - 0.3, true_peak=True)])
    out = x
    for _ in range(3):
        out = limiter(x * 10 ** (gain_db / 20), sr)
        err = target_lufs - loudness_lufs(out, sr)
        if abs(err) < 0.2:
            break
        gain_db += err
    return np.asarray(out, dtype=np.float32)
