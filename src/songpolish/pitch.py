"""Natural-sounding vocal pitch correction.

Unlike hard auto-tune (which flattens every frame onto the grid), this works
per *note*: each stable sung note is shifted by one constant amount toward its
target, so vibrato, scoops and expression inside the note are preserved. Glides
between notes and very short notes are left alone.

Each note is rendered with its own *constant* Rubber Band shift and crossfaded
into place. Do not switch to pedalboard's per-sample shift array: every change
of value emits an audible artefact burst (~18 dB below the signal), i.e. a chirp
at every corrected note, and it also applies changes ~87 ms early.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
import pedalboard
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.stats import trim_mean

from songpolish.analysis import HarmonyMap, Key
from songpolish.audio_io import WORK_SR, to_mono
from songpolish.melody import SungNote, TargetStrategy, choose_targets

ANALYSIS_SR = 22050
HOP = 256  # ~11.6 ms at 22.05 kHz
FMIN, FMAX = 65.0, 1050.0  # C2..C6 covers sung voice

MIN_NOTE_S = 0.08  # shorter segments are transitions, leave them
# A separated vocal stem still holds faint guitar bleed where nobody sings, and pYIN finds
# "notes" in it. On the first real take a third of the detected notes were such phantoms,
# 40-90 dB below the singing (real notes were within 20 dB of it, nothing in between).
PHANTOM_DB = 30.0  # below the loud notes (90th percentile of note levels)
MAX_NOTE_SPREAD = 0.7  # semitones (std); above this the "note" is a glide
# Adjacent pieces closer than this are one note split by vibrato near a half-semitone;
# real note changes are at least ~1 semitone apart.
MERGE_SEMITONES = 0.6
CENTER_WINDOW_S = 0.15  # median window used to find the note centre under vibrato
# Drift inside held notes (sagging, creeping) is measured for the report but never
# corrected. Correcting it meant cutting long notes into 0.1 s pieces, each rendered with
# its own shift and crossfaded: two renders never line up in phase, so every crossfade
# blurred the timbre, ~11 times a second. On the real take that was a clear timbre jump
# at that rate in 43 of 68 held notes ("singing in front of a fan"). Each note now gets one
# constant shift. Vibrato is smoothed out of the measure; the onset is skipped (scoops).
HELD_MIN_NOTE_S = 0.25
HELD_ONSET_S = 0.1
HELD_SMOOTH_S = 0.2
CROSSFADE_S = 0.02  # blend between a corrected note and its neighbours
RENDER_PAD_S = 0.1  # extra context around each note so Rubber Band settles before it
TOLERANCE = 0.08  # semitones; notes closer than this to target are already in tune
MIN_SHIFT = 0.03  # semitones; smaller shifts are inaudible, skip them
# Leakage: separation leaves part of the voice in the accompaniment stem (median -12 dB
# on the notes moved in the first real take, as loud as the voice on a quarter of them).
# When a note is moved, that leftover stays at the old pitch and the note sounds doubled
# and artificial. So before shifting, the voice's harmonics are moved from the
# accompaniment back into the vocal stem, only on notes that move far enough to matter.
# Only notes moved to another note (fine-tuning moves at most ~0.4 st): there the leftover
# is a wrong note. On a fine-tuned note the guitar often plays the same note, and taking
# its partials would detune the guitar; the small leftover just blends.
LEAK_MIN_SHIFT = 0.5  # semitones
LEAK_N_FFT = 8192  # fine frequency resolution: the guitar plays close to the voice
LEAK_HOP = 1024
LEAK_MAX_HZ = 6000.0  # above this the leftover is breath and sibilance, not pitch
LEAK_WIDTH = 0.025  # half-width of each harmonic's band, fraction of its frequency (~40 c)
LEAK_MIN_WIDTH_HZ = 8.0  # at least ~1.5 FFT bins, so the band catches the harmonic
LEAK_DEPTH = 0.9  # fraction moved; never strip the accompaniment to zero
# Leftover voice is quieter than the separated voice (median -12 dB). Where the
# accompaniment is louder than that in a bin, it is the guitar (a chord tone, or the
# guitar under a phantom note), and moving it would pitch-shift the guitar: it sounded
# robotic on a held F over an F chord. So never take more than this fraction of the voice.
LEAK_MAX_OF_VOICE = 0.5  # magnitude, per bin (-6 dB)


@dataclass
class PitchReport:
    notes_total: int = 0
    notes_corrected: int = 0
    mean_abs_error_cents: float = 0.0  # before correction, over detected notes
    mean_abs_shift_cents: float = 0.0  # applied, over corrected notes
    # Held notes only, after the onset: how far the (vibrato-smoothed) pitch wanders
    # from the note's own centre, before correction.
    mean_abs_drift_cents: float = 0.0
    # Notes moved to a different note than the nearest one (chord, key or melody fix).
    notes_retargeted: int = 0

    def summary(self) -> str:
        if not self.notes_total:
            return "No sung notes detected."
        text = (
            f"{self.notes_corrected}/{self.notes_total} notes nudged; average error "
            f"{self.mean_abs_error_cents:.0f} cents, average shift {self.mean_abs_shift_cents:.0f}"
            " cents."
        )
        if self.notes_retargeted:
            text += f" {self.notes_retargeted} notes moved to a different (right) note."
        return text


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) runs of True."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    return list(zip(edges[::2], edges[1::2], strict=True))


def _merge_pieces(run: np.ndarray, pieces: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Repeatedly join the closest-pitched adjacent pieces while they are one note."""
    pieces = list(pieces)
    while len(pieces) > 1:
        means = [trim_mean(run[a:b], 0.15) for a, b in pieces]
        gaps = np.abs(np.diff(means))
        i = int(np.argmin(gaps))
        if gaps[i] >= MERGE_SEMITONES:
            break
        pieces[i : i + 2] = [(pieces[i][0], pieces[i + 1][1])]
    return pieces


def _held_deviation(note: np.ndarray, center: float) -> float:
    """Mean distance of the (vibrato-smoothed) pitch from the note's centre, after onset."""
    onset = int(HELD_ONSET_S * ANALYSIS_SR / HOP)
    slow = uniform_filter1d(note, size=max(1, int(HELD_SMOOTH_S * ANALYSIS_SR / HOP)))
    return float(np.mean(np.abs(slow[onset:] - center)))


@dataclass(frozen=True)
class NoteShift:
    start: int  # samples
    end: int
    semitones: float


def detect_notes(
    vocal: np.ndarray, reference_cents: float = 0.0, sr: int = WORK_SR
) -> list[SungNote]:
    """Stable sung notes, on a grid aligned with the accompaniment's tuning."""
    y = librosa.resample(to_mono(vocal)[0], orig_sr=sr, target_sr=ANALYSIS_SR)
    f0, voiced, _ = librosa.pyin(
        y, fmin=FMIN, fmax=FMAX, sr=ANALYSIS_SR, frame_length=1024, hop_length=HOP
    )
    voiced &= np.isfinite(f0)
    midi = np.full(len(f0), np.nan)
    midi[voiced] = librosa.hz_to_midi(f0[voiced]) - reference_cents / 100.0

    center_win = max(3, int(CENTER_WINDOW_S * ANALYSIS_SR / HOP) | 1)
    min_frames = int(MIN_NOTE_S * ANALYSIS_SR / HOP)
    # pYIN frames are centred: frame i covers samples around i * HOP.
    to_samples = HOP * sr / ANALYSIS_SR
    notes = []
    for start, end in _segments(voiced):
        run = midi[start:end]
        center = median_filter(run, size=min(center_win, len(run) | 1), mode="nearest")
        target = np.round(center)
        # Split the voiced run wherever the nearest note changes, then re-join pieces
        # that are really one note.
        change = np.flatnonzero(np.diff(target)) + 1
        bounds = np.concatenate([[0], change, [len(run)]])
        for a, b in _merge_pieces(run, list(zip(bounds[:-1], bounds[1:], strict=True))):
            note = run[a:b]
            if b - a < min_frames or np.std(note) > MAX_NOTE_SPREAD:
                continue
            notes.append(
                SungNote(
                    start=max(0, int((start + a - 0.5) * to_samples)),
                    end=int((start + b - 0.5) * to_samples),
                    # Perceived pitch of a note with vibrato is its mean; trimming drops scoops.
                    pitch=float(trim_mean(note, 0.15)),
                    frames=note,
                )
            )
    return _drop_phantoms(notes, to_mono(vocal)[0])


def _drop_phantoms(notes: list[SungNote], y: np.ndarray) -> list[SungNote]:
    if not notes:
        return notes
    level = np.array([10 * np.log10(np.mean(y[n.start : n.end] ** 2) + 1e-15) for n in notes])
    keep = level >= np.percentile(level, 90) - PHANTOM_DB
    return [n for n, k in zip(notes, keep, strict=True) if k]


def plan_corrections(
    vocal: np.ndarray,
    strength: float,
    reference_cents: float = 0.0,
    key: Key | None = None,
    harmony: HarmonyMap | None = None,
    sr: int = WORK_SR,
    strategy: TargetStrategy = TargetStrategy.NEAREST,
    notes: list[SungNote] | None = None,
) -> tuple[list[NoteShift], PitchReport]:
    """Find sung notes and the constant shift(s) that move each toward its target.

    ``reference_cents`` is the tuning of the accompaniment relative to A=440, so the
    voice is tuned to the guitar, not to an abstract grid. With ``harmony``, targets
    follow the chords (see :mod:`songpolish.melody`). With ``key``, targets are limited
    to the key (to the key at each moment when ``harmony`` knows it). Pass ``notes``
    to reuse a detection across strategies.
    """
    if notes is None:
        notes = detect_notes(vocal, reference_cents, sr)
    if harmony is not None and harmony.sr != sr:
        raise ValueError("harmony map and vocal must share a sample rate")
    allowed = None
    if key is not None:
        # Strict key: each note may only land in the key where it is sung. The harmony
        # map follows modulations, which a single song-wide key cannot.
        allowed = [
            (harmony.key_at(n.start, n.end) if harmony and harmony.keys else key).pitch_classes
            for n in notes
        ]
    targets = choose_targets(notes, strategy, harmony, allowed)
    report = PitchReport(notes_total=len(notes))
    shifts: list[NoteShift] = []
    errors, note_shifts, drifts = [], [], []

    for note, target in zip(notes, targets, strict=True):
        error = target - note.pitch
        errors.append(abs(error))
        # Compared with the nearest semitone actually sung: any move to another note
        # (chord, key or melody fix) is a wrong-note fix.
        retargeted = target != round(note.pitch)
        report.notes_retargeted += retargeted
        if len(note.frames) >= HELD_MIN_NOTE_S * ANALYSIS_SR / HOP:
            drifts.append(_held_deviation(note.frames, note.pitch))
        if abs(error) < TOLERANCE:
            continue
        # A wrong note moved to the right one is fixed completely; strength only governs
        # fine-tuning (80% of a 2-semitone fix would leave it audibly wrong).
        shift = error if retargeted else strength * error
        if abs(shift) < MIN_SHIFT:
            continue
        report.notes_corrected += 1
        note_shifts.append(abs(shift))
        shifts.append(NoteShift(note.start, note.end, float(shift)))

    if errors:
        report.mean_abs_error_cents = float(np.mean(errors) * 100)
    if drifts:
        report.mean_abs_drift_cents = float(np.mean(drifts) * 100)
    if note_shifts:
        report.mean_abs_shift_cents = float(np.mean(note_shifts) * 100)
    return shifts, report


def reclaim_leakage(
    vocal: np.ndarray,
    accompaniment: np.ndarray,
    notes: list[SungNote],
    shifts: list[NoteShift],
    reference_cents: float = 0.0,
    sr: int = WORK_SR,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Move the voice left in the accompaniment into the vocal stem, on moved notes.

    Returns (vocal, accompaniment, notes treated). The two stems still sum to the
    same signal, so nothing is lost; the voice is just where the shift will reach it.
    """
    moved = [
        n
        for n in notes
        if any(
            abs(sh.semitones) >= LEAK_MIN_SHIFT and sh.start < n.end and sh.end > n.start
            for sh in shifts
        )
    ]
    if not moved:
        return vocal, accompaniment, 0
    spec = librosa.stft(accompaniment, n_fft=LEAK_N_FFT, hop_length=LEAK_HOP)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=LEAK_N_FFT)
    nbins = int(np.searchsorted(freqs, LEAK_MAX_HZ * (1 + LEAK_WIDTH)))
    mask = np.zeros((nbins, spec.shape[-1]), dtype=np.float32)
    to_samples = HOP * sr / ANALYSIS_SR
    for n in moved:
        for j in range(-(-n.start // LEAK_HOP), min(n.end // LEAK_HOP + 1, spec.shape[-1])):
            # The note's own pitch track at this frame (follows vibrato and drift).
            i = int(np.clip(round((j * LEAK_HOP - n.start) / to_samples), 0, len(n.frames) - 1))
            f0 = float(librosa.midi_to_hz(n.frames[i] + reference_cents / 100.0))
            harmonics = f0 * np.arange(1, int(LEAK_MAX_HZ // f0) + 1)
            width = np.maximum(LEAK_MIN_WIDTH_HZ, LEAK_WIDTH * harmonics)
            # Flat over the inner half of the band (vibrato, pYIN error), then tapered.
            near = (width - np.abs(freqs[:nbins, None] - harmonics[None, :])) / (width / 2)
            mask[:, j] = np.maximum(mask[:, j], np.clip(near, 0.0, 1.0).max(axis=1))
    full = np.zeros(spec.shape[-2:], dtype=np.float32)
    full[:nbins] = LEAK_DEPTH * mask
    taken = spec * full
    voice = np.abs(librosa.stft(to_mono(vocal)[0], n_fft=LEAK_N_FFT, hop_length=LEAK_HOP))
    voice = voice[:, : taken.shape[-1]]
    cap = LEAK_MAX_OF_VOICE * voice / (np.abs(taken) + 1e-12)
    taken = taken * np.minimum(1.0, cap)
    leak = librosa.istft(taken, hop_length=LEAK_HOP, length=accompaniment.shape[-1])
    leak = leak.astype(np.float32)
    into_vocal = leak if leak.shape[0] == vocal.shape[0] else to_mono(leak)
    if into_vocal.shape[0] != vocal.shape[0]:
        into_vocal = np.repeat(into_vocal, vocal.shape[0], axis=0)
    return vocal + into_vocal, accompaniment - leak, len(moved)


def apply_corrections(
    audio: np.ndarray, notes: list[NoteShift], base_shift: float = 0.0, sr: int = WORK_SR
) -> np.ndarray:
    """Shift everything by ``base_shift`` and each note by its own extra amount."""
    base = shift_pitch(audio, base_shift, sr)
    if not notes:
        return base
    notes = _merge_equal_neighbours(notes)
    n = audio.shape[-1]
    fade, pad = int(CROSSFADE_S * sr), int(RENDER_PAD_S * sr)
    acc = np.zeros(audio.shape, dtype=np.float64)
    weight = np.zeros(n)
    for note in notes:
        # Window: sin^2 ramps centred on the note edges, so neighbours sum to 1.
        w0, w1 = max(0, note.start - fade // 2), min(n, note.end + fade // 2)
        if w1 - w0 <= 2 * fade:
            continue
        r0, r1 = max(0, w0 - pad), min(n, w1 + pad)
        rendered = shift_pitch(audio[:, r0:r1], base_shift + note.semitones, sr)
        ramp = np.sin(np.linspace(0, np.pi / 2, fade)) ** 2
        w = np.ones(w1 - w0)
        w[:fade], w[-fade:] = ramp, ramp[::-1]
        acc[:, w0:w1] += rendered[:, w0 - r0 : w1 - r0] * w
        weight[w0:w1] += w
    over = weight > 1.0  # only if two notes are closer than one crossfade
    acc[:, over] /= weight[over]
    weight = np.minimum(weight, 1.0)
    return (base * (1.0 - weight) + acc).astype(np.float32)


def _merge_equal_neighbours(notes: list[NoteShift]) -> list[NoteShift]:
    """Touching pieces with the same shift render as one (fewer Rubber Band passes)."""
    merged = [notes[0]]
    for n in notes[1:]:
        last = merged[-1]
        if n.start == last.end and abs(n.semitones - last.semitones) < 1e-6:
            merged[-1] = NoteShift(last.start, n.end, last.semitones)
        else:
            merged.append(n)
    return merged


def correct_pitch(
    vocal: np.ndarray,
    strength: float,
    reference_cents: float = 0.0,
    key: Key | None = None,
    base_shift: float = 0.0,
    harmony: HarmonyMap | None = None,
    sr: int = WORK_SR,
    strategy: TargetStrategy = TargetStrategy.NEAREST,
    notes: list[SungNote] | None = None,
) -> tuple[np.ndarray, PitchReport]:
    shifts, report = [], PitchReport()
    if strength > 0:
        shifts, report = plan_corrections(
            vocal, strength, reference_cents, key, harmony, sr, strategy, notes
        )
    if not shifts and abs(base_shift) < MIN_SHIFT:
        return vocal, report
    return apply_corrections(vocal, shifts, base_shift, sr), report


def shift_pitch(audio: np.ndarray, semitones: float, sr: int = WORK_SR) -> np.ndarray:
    """Formant-preserving, length-preserving constant pitch shift."""
    if abs(semitones) < 1e-4:
        return audio
    out = pedalboard.time_stretch(
        np.ascontiguousarray(audio, dtype=np.float32),
        sr,
        stretch_factor=1.0,
        pitch_shift_in_semitones=float(semitones),
        high_quality=True,
        preserve_formants=True,
    )
    n = audio.shape[-1]
    if out.shape[-1] < n:
        out = np.pad(out, ((0, 0), (0, n - out.shape[-1])))
    return out[..., :n]
