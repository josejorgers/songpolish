"""Choosing the *target* note for each sung note.

Detection and rendering live in :mod:`songpolish.pitch`; this module only decides
where each note should go. Strategies differ in how they treat notes that clash
with the guitar (wrong notes, not just out-of-tune ones):

- ``NEAREST``: per note, nearby semitones only (≤ 1 st). Never changes the melody.
- ``PHRASE``: a whole phrase is decoded at once (Viterbi). A clashing note may move
  up to a minor third, preferring smooth melodies that move with the guitar.
- ``REPETITION``: songs repeat. Notes are compared with the same spot in other
  verses/choruses; a clear outlier moves to what the other repetitions agree on.

Pitches are on a grid where integers are notes in the guitar's tuning (MIDI-like).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from songpolish.analysis import HARMONY_HOP, HarmonyMap


class TargetStrategy(StrEnum):
    NEAREST = "nearest"
    PHRASE = "phrase"
    REPETITION = "repetition"


@dataclass
class SungNote:
    start: int  # samples
    end: int
    pitch: float  # perceived pitch (trimmed mean), reference grid
    frames: np.ndarray  # per-frame pitch track of the note (drift report, leakage mask)


# --- per-note cost (shared by all strategies) ----------------------------------------
# Linear cost: distance / HARMONY_DISTANCE - HARMONY_PULL * chord weight - key bonus.
# With these values an off-chord neighbour wins while the voice is within ~35 cents of
# it; further away the chord tone wins (a squared distance cost never lets the chord win).
HARMONY_MAX_MOVE = 1.0  # semitones, for NEAREST
HARMONY_DISTANCE = 0.35  # semitones
HARMONY_PULL = 1.5
# Notes outside the song's key clash with the guitar. Such a note is kept only when sung
# accurately (within ~18 cents: clearly intended, e.g. a blue note); otherwise the nearest
# in-key note wins. Real case: 236 Hz sung over an A chord (A=220, B=248) is A# +20 cents,
# but almost certainly a flat B.
HARMONY_IN_KEY_BONUS = 1.5

# --- PHRASE ------------------------------------------------------------------------
PHRASE_GAP_S = 0.6  # a longer silence ends a phrase
WIDE_MAX_MOVE = 3  # semitones (minor third), only for clashing notes
WIDE_PENALTY = 1.0  # extra cost per semitone moved beyond 1
CLASH_WEIGHT = 0.35  # chord weight below which the nearest choice "clashes"
LEAP_FREE = 2  # melodic steps up to a whole tone are free
LEAP_PENALTY = 0.6  # per semitone beyond LEAP_FREE
GUIDE_MIN_MOVE = 0.5  # semitones the guitar line must move to suggest a direction
GUIDE_PENALTY = 3.0  # melody moving against the guitar's direction
SUNG_CONTOUR_PENALTY = 1.0  # melody moving against the direction actually sung

# --- REPETITION ----------------------------------------------------------------------
REPEAT_CONTEXT_S = 2.0  # half-width of the context compared between repetitions
REPEAT_MIN_DISTANCE_S = 6.0  # a repetition is at least this far away
REPEAT_SIMILARITY = 0.9  # cosine similarity of chord + melody context
REPEAT_NEIGHBOURS = 3  # notes on each side used to line up the phrases
REPEAT_MAX_PHRASE_DIFF = 1.0  # mean semitone difference for "the same phrase"
REPEAT_MIN_AGREE = 2  # other repetitions that must agree
REPEAT_AGREE_FRACTION = 0.6
REPEAT_MAX_MOVE = 3.5  # semitones

# --- repeats in a new key (PHRASE and REPETITION) -----------------------------------
# A song that modulates usually repeats its chorus in the new key, but a singer may not
# follow it all the way. Real case: A major -> C major at 3:19; the voice sang the chorus
# ~1 st above the earlier one instead of 3, and snapping each note to the nearest C-major
# note put several a scale step low (a drop where the melody rises). So a phrase after a
# key change follows its earlier repetition, moved by the key change. The guitar must
# confirm that interval: of all 12 shifts of the earlier accompaniment, it must fit best
# (a change to the parallel minor shares the signature but keeps the melody in place).
MODULATED_STEP_S = 0.05  # resolution of the sung contour used to line up repetitions
MODULATED_MIN_S = 2.0  # shorter phrases are matched with this much context
MODULATED_CONTEXT_S = 0.3
MODULATED_MAX_DIFF = 0.8  # mean semitones between contours for "the same melody"
MODULATED_MIN_OVERLAP = 0.6  # fraction of the phrase that must line up with singing
MODULATED_ALIGN_S = 0.2  # a note's counterpart starts within this of the same moment
MODULATED_MAX_MOVE = 3.5  # semitones


def choose_targets(
    notes: list[SungNote],
    strategy: TargetStrategy,
    harmony: HarmonyMap | None,
    allowed: Sequence[list[int] | None] | None = None,
) -> list[float]:
    """``allowed``: per note, the only pitch classes it may land on (None = any)."""
    if allowed is None:
        allowed = [None] * len(notes)
    nearest = [_best_nearby(n, harmony, a) for n, a in zip(notes, allowed, strict=True)]
    if harmony is None or strategy is TargetStrategy.NEAREST or not notes:
        return nearest
    if strategy is TargetStrategy.PHRASE:
        targets = _phrase_targets(notes, nearest, harmony, allowed)
    else:
        targets = _repetition_targets(notes, nearest, harmony)
    return _modulated_repeat_targets(notes, targets, harmony, allowed)


def nearest_allowed(pitch: float, allowed: list[int] | None) -> float:
    if allowed is None:
        return float(np.round(pitch))
    candidates = [o * 12 + pc for o in range(-1, 11) for pc in allowed]
    return float(min(candidates, key=lambda k: abs(k - pitch)))


def note_cost(pitch: float, k: float, weights: np.ndarray, scale: list[int] | None) -> float:
    pc = int(k) % 12
    in_key = scale is not None and pc in scale
    return (
        abs(k - pitch) / HARMONY_DISTANCE
        - HARMONY_PULL * weights[pc]
        - (HARMONY_IN_KEY_BONUS if in_key else 0.0)
    )


def _scale(harmony: HarmonyMap, note: SungNote) -> list[int] | None:
    """The key's notes where this note is sung (follows modulations)."""
    key = harmony.key_at(note.start, note.end)
    return key.pitch_classes if key else None


def _candidates(pitch: float, max_move: float, allowed: list[int] | None) -> list[float]:
    lo, hi = int(np.floor(pitch - max_move)), int(np.ceil(pitch + max_move))
    return [
        float(k)
        for k in range(lo, hi + 1)
        if abs(k - pitch) <= max_move and (allowed is None or k % 12 in allowed)
    ]


def _best_nearby(note: SungNote, harmony: HarmonyMap | None, allowed: list[int] | None) -> float:
    if harmony is None:
        return nearest_allowed(note.pitch, allowed)
    candidates = _candidates(note.pitch, HARMONY_MAX_MOVE, allowed)
    if not candidates:
        return nearest_allowed(note.pitch, allowed)
    w, scale = harmony.weights(note.start, note.end), _scale(harmony, note)
    return min(candidates, key=lambda k: note_cost(note.pitch, k, w, scale))


# --- PHRASE ---------------------------------------------------------------------------


def _phrases(notes: list[SungNote], sr: int) -> list[list[int]]:
    groups: list[list[int]] = [[0]]
    for i in range(1, len(notes)):
        if notes[i].start - notes[i - 1].end > PHRASE_GAP_S * sr:
            groups.append([])
        groups[-1].append(i)
    return groups


def _clashes(note: SungNote, choice: float, harmony: HarmonyMap) -> bool:
    scale = _scale(harmony, note)
    w = harmony.weights(note.start, note.end)
    pc = int(choice) % 12
    return (scale is not None and pc not in scale) or w[pc] < CLASH_WEIGHT


def _phrase_targets(
    notes: list[SungNote],
    nearest: list[float],
    harmony: HarmonyMap,
    allowed: Sequence[list[int] | None],
) -> list[float]:
    targets = list(nearest)
    for phrase in _phrases(notes, harmony.sr):
        if len(phrase) < 2:
            continue
        cands, emit = [], []
        for i in phrase:
            n = notes[i]
            scale = _scale(harmony, n)
            if _clashes(n, nearest[i], harmony):
                c = _candidates(n.pitch, WIDE_MAX_MOVE, allowed[i])
                if scale is not None:
                    # A fix must not land on another clash: only in-key notes (or the
                    # nearest choice itself) are candidates for a clashing note.
                    c = [k for k in c if int(k) % 12 in scale or k == nearest[i]]
                c = c or [nearest[i]]
            else:
                # A note that fits the chord and key is an anchor: smoothing the line must
                # not move it (real case: a chord-tone C sung 28 cents flat became B).
                c = [nearest[i]]
            w = harmony.weights(n.start, n.end)
            cands.append(c)
            emit.append(
                np.array(
                    [
                        note_cost(n.pitch, k, w, scale)
                        + WIDE_PENALTY * max(0.0, abs(k - n.pitch) - 1.0)
                        for k in c
                    ]
                )
            )
        # Viterbi over the phrase.
        cost, back = emit[0], []
        for t in range(1, len(phrase)):
            prev, cur = notes[phrase[t - 1]], notes[phrase[t]]
            guide = harmony.line(cur.start, cur.end) - harmony.line(prev.start, prev.end)
            sung = cur.pitch - prev.pitch
            trans = np.array(
                [[_transition(kp, k, guide, sung) for k in cands[t]] for kp in cands[t - 1]]
            )
            total = cost[:, None] + trans
            back.append(total.argmin(axis=0))
            cost = total.min(axis=0) + emit[t]
        path = [int(cost.argmin())]
        for b in reversed(back):
            path.append(int(b[path[-1]]))
        path.reverse()
        for t, i in enumerate(phrase):
            targets[i] = cands[t][path[t]]
    return targets


def _transition(k_prev: float, k: float, guide: float, sung: float) -> float:
    step = k - k_prev
    cost = LEAP_PENALTY * max(0.0, abs(step) - LEAP_FREE)
    if step != 0 and abs(guide) >= GUIDE_MIN_MOVE and np.sign(step) != np.sign(guide):
        cost += GUIDE_PENALTY
    if step != 0 and abs(sung) >= 0.5 and np.sign(step) != np.sign(sung):
        cost += SUNG_CONTOUR_PENALTY
    return cost


# --- REPETITION ----------------------------------------------------------------------


def _context(notes: list[SungNote], i: int, harmony: HarmonyMap) -> np.ndarray:
    """Chords + sung pitch classes around note i, as one unit vector."""
    sr = harmony.sr
    mid = (notes[i].start + notes[i].end) // 2
    half = int(REPEAT_CONTEXT_S * sr)
    chords = harmony.weights(mid - half, mid + half)
    sung = np.zeros(12)
    for n in notes:
        if n.end > mid - half and n.start < mid + half:
            sung[int(round(n.pitch)) % 12] += (n.end - n.start) / sr
    parts = [v / (np.linalg.norm(v) + 1e-9) for v in (chords, sung)]
    v = np.concatenate(parts)
    return v / (np.linalg.norm(v) + 1e-9)


def _phrase_distance(pitches: np.ndarray, i: int, j: int) -> float:
    """Mean semitone difference between the phrases around notes i and j."""
    diffs = []
    for u in range(-REPEAT_NEIGHBOURS, REPEAT_NEIGHBOURS + 1):
        if u == 0 or not (0 <= i + u < len(pitches) and 0 <= j + u < len(pitches)):
            continue
        diffs.append(min(abs(pitches[i + u] - pitches[j + u]), 4.0))
    return float(np.mean(diffs)) if len(diffs) >= 2 else np.inf


def _one_per_repetition(
    matches: list[tuple[float, int]], times: np.ndarray
) -> list[tuple[float, int]]:
    """Matches closer in time than a repetition belong to one; keep the best-aligned."""
    out: list[tuple[float, int]] = []
    for d, j in sorted(matches, key=lambda m: times[m[1]]):
        if out and times[j] - times[out[-1][1]] < REPEAT_MIN_DISTANCE_S:
            if d < out[-1][0]:
                out[-1] = (d, j)
        else:
            out.append((d, j))
    return out


def _repetition_targets(
    notes: list[SungNote], nearest: list[float], harmony: HarmonyMap
) -> list[float]:
    sr = harmony.sr
    ctx = np.array([_context(notes, i, harmony) for i in range(len(notes))])
    sim = ctx @ ctx.T
    pitches = np.array(nearest)
    times = np.array([n.start for n in notes]) / sr
    targets = list(nearest)
    for i in range(len(notes)):
        far = np.abs(times - times[i]) >= REPEAT_MIN_DISTANCE_S
        matches: list[tuple[float, int]] = []  # (phrase distance, note index)
        for j in np.flatnonzero(far & (sim[i] >= REPEAT_SIMILARITY)):
            # Line the phrases up: the corresponding note may be a neighbour of j.
            best = min(
                (_phrase_distance(pitches, i, j + o), j + o)
                for o in (-1, 0, 1)
                if 0 <= j + o < len(notes)
            )
            if best[0] <= REPEAT_MAX_PHRASE_DIFF:
                matches.append(best)
        regions = _one_per_repetition(matches, times)
        if len(regions) < REPEAT_MIN_AGREE:
            continue
        # Octave-normalise the other repetitions' notes to the one sung here.
        votes = Counter(pitches[j] + 12 * round((nearest[i] - pitches[j]) / 12) for _, j in regions)
        value, count = votes.most_common(1)[0]
        enough = count >= REPEAT_MIN_AGREE and count >= REPEAT_AGREE_FRACTION * len(regions)
        if enough and value != nearest[i] and abs(value - notes[i].pitch) <= REPEAT_MAX_MOVE:
            targets[i] = float(value)
    return targets


# --- repeats in a new key -------------------------------------------------------------


def _signature(key) -> int:
    """Relative-major tonic: A minor and C major share one key signature (0)."""
    return (key.tonic + 3) % 12 if key.minor else key.tonic


def _sung_contour(notes: list[SungNote], sr: int) -> np.ndarray:
    end = max(n.end for n in notes) / sr
    contour = np.full(int(end / MODULATED_STEP_S) + 1, np.nan)
    for n in notes:
        contour[int(n.start / sr / MODULATED_STEP_S) : int(n.end / sr / MODULATED_STEP_S)] = n.pitch
    return contour


def _earlier_match(
    contour: np.ndarray, t0: float, t1: float, searchable: np.ndarray
) -> tuple[float, float] | None:
    """(start s, mean diff) of the passage sung most like [t0, t1), at any transposition."""
    a, b = int(t0 / MODULATED_STEP_S), int(t1 / MODULATED_STEP_S)
    query = contour[a:b]
    sung = ~np.isnan(query)
    best = None
    for s in np.flatnonzero(searchable[: len(contour) - len(query)]):
        cand = contour[s : s + len(query)]
        ok = sung & ~np.isnan(cand)
        if ok.sum() < MODULATED_MIN_OVERLAP * sung.sum():
            continue
        diff = query[ok] - cand[ok]
        dist = float(np.mean(np.minimum(np.abs(diff - np.median(diff)), 3.0)))
        if best is None or dist < best[1]:
            best = (s * MODULATED_STEP_S, dist)
    return best if best and best[1] <= MODULATED_MAX_DIFF else None


def _guitar_confirms(harmony: HarmonyMap, t_old: float, t_new: float, length: float, k: int):
    """Transposing the earlier accompaniment by k fits the new passage best of all shifts.

    Each note's mean is removed first: a drone (an open string ringing through the whole
    song) is equally strong in both passages and would otherwise favour "no transposition".
    """
    fps = harmony.sr / HARMONY_HOP
    a, b = int(t_old * fps), int(t_new * fps)
    n = min(int(length * fps), harmony.chroma.shape[1] - max(a, b))
    if n <= 1:
        return False
    old, new = harmony.chroma[:, a : a + n], harmony.chroma[:, b : b + n]
    old = old - old.mean(axis=1, keepdims=True)
    new = new - new.mean(axis=1, keepdims=True)
    fit = [np.corrcoef(np.roll(old, r, axis=0).ravel(), new.ravel())[0, 1] for r in range(12)]
    return int(np.nanargmax(fit)) == k


def _modulated_repeat_targets(
    notes: list[SungNote],
    targets: list[float],
    harmony: HarmonyMap,
    allowed: Sequence[list[int] | None],
) -> list[float]:
    if len(harmony.keys) < 2:
        return targets
    sr = harmony.sr
    contour = _sung_contour(notes, sr)
    times = np.arange(len(contour)) * MODULATED_STEP_S
    span_of = np.searchsorted([s.end / sr for s in harmony.keys], times, side="right")
    span_of = np.minimum(span_of, len(harmony.keys) - 1)
    starts = np.array([n.start for n in notes]) / sr
    out = list(targets)
    for phrase in _phrases(notes, sr):
        first = notes[phrase[0]]
        here = min(int(span_of[int(first.start / sr / MODULATED_STEP_S)]), len(harmony.keys) - 1)
        if here == 0:
            continue
        sig = _signature(harmony.keys[here].key)
        t0 = first.start / sr - MODULATED_CONTEXT_S
        t1 = max(notes[phrase[-1]].end / sr + MODULATED_CONTEXT_S, t0 + MODULATED_MIN_S)
        searchable = np.array([_signature(harmony.keys[s].key) != sig for s in span_of], dtype=bool)
        searchable[int(t0 / MODULATED_STEP_S) :] = False  # earlier repetitions only
        match = _earlier_match(contour, t0, t1, searchable)
        if match is None:
            continue
        t_old = match[0]
        old_span = harmony.keys[int(span_of[int(t_old / MODULATED_STEP_S)])]
        interval = (sig - _signature(old_span.key)) % 12
        if not _guitar_confirms(harmony, t_old, t0, t1 - t0, interval):
            continue
        for i in phrase:
            n = notes[i]
            when = t_old + n.start / sr - t0
            j = int(np.argmin(np.abs(starts - when)))
            if abs(starts[j] - when) > MODULATED_ALIGN_S:
                continue
            k = targets[j] + interval
            k += 12 * round((n.pitch - k) / 12)  # the octave that was sung
            if abs(k - n.pitch) > MODULATED_MAX_MOVE:
                continue
            if allowed[i] is not None and int(k) % 12 not in allowed[i]:
                continue
            out[i] = float(k)
    return out
