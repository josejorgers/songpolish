"""Musical analysis: reference tuning, key, loudness and spectrum measurements."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
import pyloudnorm as pyln

from songpolish.audio_io import WORK_SR, to_mono

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles.
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
_MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]


@dataclass(frozen=True)
class Key:
    tonic: int  # pitch class, 0 = C
    minor: bool

    @property
    def name(self) -> str:
        return f"{NOTE_NAMES[self.tonic]} {'minor' if self.minor else 'major'}"

    @property
    def pitch_classes(self) -> list[int]:
        scale = _MINOR_SCALE if self.minor else _MAJOR_SCALE
        return sorted((self.tonic + s) % 12 for s in scale)


def tuning_offset_cents(audio: np.ndarray, sr: int = WORK_SR) -> float:
    """Deviation of the recording's tuning from A=440 Hz, in cents (-50..50)."""
    y = to_mono(audio)[0]
    # librosa returns a fraction of a semitone (bins_per_octave=12 => resolution 0.01).
    return float(librosa.estimate_tuning(y=y, sr=sr, resolution=0.01)) * 100.0


def detect_key(audio: np.ndarray, sr: int = WORK_SR, tuning_cents: float = 0.0) -> Key:
    y = to_mono(audio)[0]
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, tuning=tuning_cents / 100.0)
    return _best_key(chroma.sum(axis=1))


def _key_scores(profile: np.ndarray) -> dict[Key, float]:
    return {
        Key(tonic, minor): float(np.corrcoef(profile, np.roll(template, tonic))[0, 1])
        for tonic in range(12)
        for minor, template in ((False, _MAJOR), (True, _MINOR))
    }


def _best_key(profile: np.ndarray) -> Key:
    scores = _key_scores(profile)
    return max(scores, key=scores.__getitem__)


# Modulations: the key is followed over time. A relative major/minor pair has the same
# notes, so the states are the 12 key signatures (A major = F# minor). A change must be
# sustained: it costs KEY_CHANGE_COST (correlation x seconds), so one chord that leans to
# a neighbouring key (E major over the V chord) never switches, while a real modulation
# that fits ~0.25 better wins after ~12 s. Real case: A major -> C major / A minor at 3:19.
KEY_WINDOW_S = 8.0
KEY_STEP_S = 1.0
KEY_CHANGE_COST = 3.0


@dataclass(frozen=True, eq=False)
class KeySpan:
    start: int  # samples
    end: int
    key: Key
    profile: np.ndarray  # (12,) pitch-class profile of the accompaniment here, max 1


def key_spans(chroma: np.ndarray, sr: int, hop: int) -> list[KeySpan]:
    """Split the song into stretches in one key (more than one when it modulates)."""
    fps = sr / hop
    step = max(1, int(KEY_STEP_S * fps))
    half = max(1, int(KEY_WINDOW_S * fps / 2))
    centers = np.arange(0, chroma.shape[1], step)
    # scores[w, s]: how well window w fits signature s (best of major s or its relative minor).
    scores = np.empty((len(centers), 12))
    for w, c in enumerate(centers):
        fit = _key_scores(chroma[:, max(0, c - half) : c + half].sum(axis=1))
        scores[w] = [max(fit[Key(s, False)], fit[Key((s + 9) % 12, True)]) for s in range(12)]
    scores = np.nan_to_num(scores, nan=0.0)
    step_s = step / fps
    cost, back = -scores[0] * step_s, []
    for w in range(1, len(centers)):
        stay = cost
        switch = cost.min() + KEY_CHANGE_COST
        back.append(np.where(stay <= switch, np.arange(12), int(cost.argmin())))
        cost = np.minimum(stay, switch) - scores[w] * step_s
    path = [int(cost.argmin())]
    for b in reversed(back):
        path.append(int(b[path[-1]]))
    path.reverse()

    spans, first = [], 0
    for w in range(1, len(centers) + 1):
        if w == len(centers) or path[w] != path[first]:
            a = int(centers[first]) if first else 0
            b = int(centers[w]) if w < len(centers) else chroma.shape[1]
            profile = chroma[:, a:b].sum(axis=1)
            sig = path[first]
            fit = _key_scores(profile)
            key = max((Key(sig, False), Key((sig + 9) % 12, True)), key=fit.__getitem__)
            spans.append(KeySpan(a * hop, b * hop, key, profile / (profile.max() + 1e-9)))
            first = w
    return spans


def loudness_lufs(audio: np.ndarray, sr: int = WORK_SR) -> float:
    meter = pyln.Meter(sr)
    data = np.atleast_2d(audio).T.astype(np.float64)
    if data.shape[0] < sr // 2:  # pyloudnorm needs at least one 400 ms block
        return -70.0
    value = meter.integrated_loudness(data)
    return float(value) if np.isfinite(value) else -70.0


def active_rms_db(audio: np.ndarray, sr: int = WORK_SR, gate_db: float = -50.0) -> float:
    """RMS of the frames that actually contain signal (ignores silences)."""
    y = to_mono(audio)[0]
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = 20 * np.log10(rms + 1e-9)
    active = db[db > gate_db]
    if active.size == 0:
        return gate_db
    return float(20 * np.log10(np.sqrt(np.mean((10 ** (active / 20)) ** 2))))


def band_levels_db(audio: np.ndarray, centers: np.ndarray, sr: int = WORK_SR) -> np.ndarray:
    """Long-term average level per octave band, relative to the band average."""
    y = to_mono(audio)[0]
    spec = np.abs(librosa.stft(y, n_fft=4096, hop_length=1024)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    # Average only over frames with energy so silences do not skew the tone.
    frame_energy = spec.sum(axis=0)
    loud = frame_energy > np.percentile(frame_energy, 30)
    ltas = spec[:, loud].mean(axis=1) if loud.any() else spec.mean(axis=1)
    levels = []
    for fc in centers:
        band = (freqs >= fc / np.sqrt(2)) & (freqs < fc * np.sqrt(2))
        levels.append(10 * np.log10(ltas[band].sum() + 1e-12))
    levels = np.array(levels)
    return levels - levels.mean()


NOISE_QUIET_FRACTION = 0.02  # the quietest 2% of frames: pauses, i.e. the noise floor


def band_snr_db(audio: np.ndarray, centers: np.ndarray, sr: int = WORK_SR) -> np.ndarray:
    """Per octave band: level while playing vs level in the pauses (the hiss floor)."""
    y = to_mono(audio)[0]
    spec = np.abs(librosa.stft(y, n_fft=4096, hop_length=1024)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    energy = spec.sum(axis=0)
    quiet = energy <= np.quantile(energy, NOISE_QUIET_FRACTION)
    loud = energy > np.percentile(energy, 30)
    out = []
    for fc in centers:
        band = (freqs >= fc / np.sqrt(2)) & (freqs < fc * np.sqrt(2))
        signal = spec[band][:, loud].sum(axis=0).mean() if loud.any() else 0.0
        noise = spec[band][:, quiet].sum(axis=0).mean()
        out.append(10 * np.log10((signal + 1e-12) / (noise + 1e-12)))
    return np.array(out)


HARMONY_HOP = 2048  # ~46 ms
HARMONY_SMOOTH_S = 0.5  # chords last longer than a strum; smooth out picking detail
HARMONY_LOCAL_WEIGHT = 0.7  # the chord right now matters more than the song overall


@dataclass(frozen=True)
class HarmonyMap:
    """Which pitch classes the accompaniment is playing, over time."""

    chroma: np.ndarray  # (12, frames), each frame normalised to max 1
    profile: np.ndarray  # (12,) whole-song profile, max 1
    sr: int
    key: Key | None = None  # detected from the accompaniment
    # Register the guitar is playing in over time (energy-weighted mean pitch, MIDI-like):
    # rises when the guitar goes low-mid-high, so a melody can follow its direction.
    pitch_line: np.ndarray | None = None
    # The key over time; more than one span when the song modulates. Empty = ``key``.
    keys: tuple[KeySpan, ...] = ()

    def _span(self, start: int, end: int) -> KeySpan | None:
        mid = (start + end) // 2
        for span in self.keys:
            if mid < span.end:
                return span
        return self.keys[-1] if self.keys else None

    def key_at(self, start: int, end: int) -> Key | None:
        span = self._span(start, end)
        return span.key if span else self.key

    def weights(self, start: int, end: int) -> np.ndarray:
        """0..1 weight per pitch class (0 = C) for the span [start, end) in samples."""
        a = min(max(0, start // HARMONY_HOP), self.chroma.shape[1] - 1)
        b = max(a + 1, min(end // HARMONY_HOP, self.chroma.shape[1]))
        local = self.chroma[:, a:b].mean(axis=1)
        local = local / (local.max() + 1e-9)
        # The background is the current key's profile, not the whole song's: after a
        # modulation the old key's notes must stop pulling.
        span = self._span(start, end)
        background = span.profile if span else self.profile
        return HARMONY_LOCAL_WEIGHT * local + (1 - HARMONY_LOCAL_WEIGHT) * background

    def line(self, start: int, end: int) -> float:
        """Mean of the guitar's pitch line over [start, end) samples (0 if unknown)."""
        if self.pitch_line is None:
            return 0.0
        a = min(max(0, start // HARMONY_HOP), len(self.pitch_line) - 1)
        b = max(a + 1, min(end // HARMONY_HOP, len(self.pitch_line)))
        return float(np.mean(self.pitch_line[a:b]))


def harmony_map(
    accompaniment: np.ndarray, sr: int = WORK_SR, tuning_cents: float = 0.0
) -> HarmonyMap:
    from scipy.ndimage import uniform_filter1d

    y = to_mono(accompaniment)[0]
    chroma = librosa.feature.chroma_cqt(
        y=y, sr=sr, hop_length=HARMONY_HOP, tuning=tuning_cents / 100.0
    )
    spans = key_spans(chroma, sr, HARMONY_HOP)
    size = max(1, int(HARMONY_SMOOTH_S * sr / HARMONY_HOP))
    chroma = uniform_filter1d(chroma, size=size, axis=1, mode="nearest")
    profile = chroma.sum(axis=1)
    chroma = chroma / (chroma.max(axis=0, keepdims=True) + 1e-9)
    # The main key is the one the song spends most time in.
    time_in: dict[Key, int] = {}
    for span in spans:
        time_in[span.key] = time_in.get(span.key, 0) + span.end - span.start
    return HarmonyMap(
        chroma=chroma,
        profile=profile / (profile.max() + 1e-9),
        sr=sr,
        key=max(time_in, key=time_in.__getitem__) if time_in else _best_key(profile),
        pitch_line=_pitch_line(y, sr, tuning_cents, size),
        keys=tuple(spans),
    )


LINE_FMIN_MIDI = 40  # E2, the guitar's lowest string


def _pitch_line(y: np.ndarray, sr: int, tuning_cents: float, smooth: int) -> np.ndarray:
    from scipy.ndimage import uniform_filter1d

    n_bins = 48  # four octaves from E2
    cqt = np.abs(
        librosa.cqt(
            y, sr=sr, hop_length=HARMONY_HOP, fmin=librosa.midi_to_hz(LINE_FMIN_MIDI),
            n_bins=n_bins, tuning=tuning_cents / 100.0,
        )
    )  # fmt: skip
    midi = LINE_FMIN_MIDI + np.arange(n_bins)
    line = (cqt * midi[:, None]).sum(axis=0) / (cqt.sum(axis=0) + 1e-9)
    return uniform_filter1d(line, size=smooth, mode="nearest")
