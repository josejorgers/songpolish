"""Per-source cleanup and tone chains: denoise, automatic EQ, de-ess, compression.

The "automatic" part is that every stage measures the input first (noise floor,
spectral balance, dynamics) and sets its own parameters, scaled by a single 0..1
amount from :class:`~songpolish.settings.Settings`. Moves are deliberately
capped so the result stays close to the original performance.
"""

from __future__ import annotations

from enum import StrEnum

import librosa
import noisereduce as nr
import numpy as np
from pedalboard import (
    Compressor,
    HighpassFilter,
    HighShelfFilter,
    LowShelfFilter,
    PeakFilter,
    Pedalboard,
)
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from songpolish.analysis import active_rms_db, band_levels_db, band_snr_db
from songpolish.audio_io import WORK_SR, to_mono
from songpolish.settings import Settings


class Source(StrEnum):
    VOCAL = "vocal"
    GUITAR = "guitar"
    MIX = "mix"  # voice + guitar in one signal (no separation available)


EQ_BANDS = np.array([100, 200, 400, 800, 1600, 3200, 6400, 12800], dtype=float)

# Reference long-term spectra (octave-band dB, relative). Rough averages of
# well-produced acoustic recordings; the auto-EQ nudges toward these. Tune by ear.
TARGET_SPECTRUM = {
    Source.VOCAL: np.array([-7.0, 1.0, 4.0, 4.0, 2.0, 0.5, -2.5, -7.0]),
    Source.GUITAR: np.array([0.0, 4.0, 3.5, 1.5, -0.5, -2.5, -5.5, -10.0]),
    Source.MIX: np.array([-2.0, 3.0, 3.5, 2.5, 0.5, -1.5, -4.5, -9.5]),
}
HIGHPASS_HZ = {Source.VOCAL: 85.0, Source.GUITAR: 65.0, Source.MIX: 50.0}
# Phone recordings are typically ~10 dB boomy/boxy and ~9 dB dull, so small caps leave
# them unchanged. Boosting stays more limited than cutting: it raises noise and harshness.
MAX_CUT_DB = 8.0
MAX_BOOST_DB = 6.0
EQ_FIT_PASSES = 2  # adjacent bands overlap; re-measure and correct so the result lands
EMPTY_BAND_DB = -24.0  # bands this far below average hold mostly noise; never boost them
# Boosting a band boosts the hiss in it too. The first real phone take had only 20-28 dB
# between the music and the hiss floor in every band, and its "dull" highs were mostly
# hiss: the +6 dB treble boost raised the hiss ~9 dB ("sounds like an old recorder").
# So a band may only be boosted as far as its signal-to-noise ratio allows.
BOOST_MIN_SNR_DB = 30.0  # no boost below this
BOOST_FULL_SNR_DB = 45.0  # the full MAX_BOOST_DB from here on
HISS_FROM_HZ = 2500.0  # hiss is heard in the treble; lower bands may move freely


def denoise(audio: np.ndarray, amount: float, sr: int = WORK_SR) -> np.ndarray:
    """Spectral gating, with the noise profile learnt from the quietest passages."""
    if amount <= 0:
        return audio
    y = to_mono(audio)[0]
    frame = 2048
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=frame)[0]
    quiet = np.flatnonzero(rms <= np.percentile(rms, 10))
    noise_clip = (
        np.concatenate([y[i * frame : (i + 1) * frame] for i in quiet]) if len(quiet) else y
    )
    kwargs = {}
    if noise_clip.size >= sr // 2:
        kwargs = {"y_noise": noise_clip, "stationary": True}
    else:
        kwargs = {"stationary": True}
    out = nr.reduce_noise(
        y=audio,
        sr=sr,
        prop_decrease=0.25 + 0.6 * amount,  # never a full gate: avoids "underwater" artefacts
        n_std_thresh_stationary=1.5,
        **kwargs,
    )
    return np.asarray(out, dtype=np.float32).reshape(audio.shape)


def auto_eq(audio: np.ndarray, source: Source, amount: float, sr: int = WORK_SR) -> Pedalboard:
    """Build an EQ that moves the measured spectrum part-way toward the target."""
    measured = band_levels_db(audio, EQ_BANDS, sr)
    snr = band_snr_db(audio, EQ_BANDS, sr)
    max_boost = MAX_BOOST_DB * np.clip(
        (snr - BOOST_MIN_SNR_DB) / (BOOST_FULL_SNR_DB - BOOST_MIN_SNR_DB), 0.0, 1.0
    )
    max_boost[measured < EMPTY_BAND_DB] = 0.0
    target = TARGET_SPECTRUM[source] - TARGET_SPECTRUM[source].mean()
    wanted = _limit(amount * (target - measured), max_boost, measured)
    gains = wanted.copy()
    for _ in range(EQ_FIT_PASSES):
        achieved = band_levels_db(_eq_board(gains)(audio, sr), EQ_BANDS, sr) - measured
        # Levels are relative to the band average, so compare shapes, not absolute gain.
        error = (wanted - wanted.mean()) - (achieved - achieved.mean())
        gains = _limit(gains + 0.8 * error, max_boost, measured)
    return _eq_board(gains)


def _limit(gains: np.ndarray, max_boost: np.ndarray, measured: np.ndarray) -> np.ndarray:
    gains = np.clip(gains, -MAX_CUT_DB, max_boost)
    # The stem is loudness-normalised afterwards, so cutting where the energy is (boom)
    # lifts every other band. In the treble, where hiss is heard, the boost limit holds
    # relative to that overall change: the boom still goes, the hiss does not come up.
    w = 10 ** (measured / 10)
    overall = 10 * np.log10(np.sum(w * 10 ** (gains / 10)) / np.sum(w))
    treble = EQ_BANDS >= HISS_FROM_HZ
    gains[treble] = np.minimum(gains[treble], overall + max_boost[treble])
    return gains


def _eq_board(gains: np.ndarray) -> Pedalboard:
    filters = [LowShelfFilter(cutoff_frequency_hz=140.0, gain_db=float(gains[0]), q=0.7)]
    for fc, g in zip(EQ_BANDS[1:-1], gains[1:-1], strict=True):
        if abs(g) >= 0.25:
            filters.append(PeakFilter(cutoff_frequency_hz=float(fc), gain_db=float(g), q=1.4))
    filters.append(HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=float(gains[-1]), q=0.7))
    return Pedalboard(filters)


def deess(
    audio: np.ndarray, amount: float, sr: int = WORK_SR, split_hz: float = 5500.0
) -> np.ndarray:
    """Tame harsh "s"/"t" sounds by ducking only the top band when it spikes."""
    if amount <= 0:
        return audio
    high = HighpassFilter(split_hz)(audio, sr)
    low = audio - high  # complementary split: low + high == audio
    env = to_mono(np.abs(high))[0]
    env = maximum_filter1d(env, size=int(0.002 * sr))
    env = uniform_filter1d(env, size=int(0.01 * sr))
    env_db = 20 * np.log10(env + 1e-9)
    active = env_db[env_db > -60]
    if active.size == 0:
        return audio
    threshold = np.percentile(active, 90)
    ratio = 2.0 + 4.0 * amount
    reduction_db = np.maximum(0.0, env_db - threshold) * (1 - 1 / ratio)
    reduction_db = np.minimum(reduction_db, 10.0)
    gain = 10 ** (-reduction_db / 20)
    return (low + high * gain[None, :]).astype(np.float32)


def compressor(audio: np.ndarray, amount: float, source: Source, sr: int = WORK_SR) -> Compressor:
    level = active_rms_db(audio, sr)
    if source is Source.VOCAL:
        ratio, attack, release = 1.5 + 2.5 * amount, 8.0, 90.0
    else:  # guitar strums have fast transients; slower attack keeps the pick
        ratio, attack, release = 1.3 + 1.7 * amount, 20.0, 150.0
    threshold = level + 4.0 - 6.0 * amount
    return Compressor(threshold_db=threshold, ratio=ratio, attack_ms=attack, release_ms=release)


def process_source(audio: np.ndarray, source: Source, s: Settings, sr: int = WORK_SR) -> np.ndarray:
    """Full cleanup + tone chain for one source. Output is not level-normalised."""
    x = HighpassFilter(HIGHPASS_HZ[source])(audio, sr)
    x = denoise(x, s.noise_reduction, sr)
    if s.tone_match > 0:
        x = auto_eq(x, source, s.tone_match, sr)(x, sr)
    if source is not Source.GUITAR:
        x = deess(x, s.compression if source is Source.VOCAL else 0.5 * s.compression, sr)
    if s.compression > 0:
        x = compressor(x, s.compression, source, sr)(x, sr)
    return np.asarray(x, dtype=np.float32)
