"""Target strategies, tested on hand-built notes and harmony (no audio needed)."""

import numpy as np
import pytest

from songpolish.analysis import HARMONY_HOP, HarmonyMap, Key
from songpolish.melody import SungNote, TargetStrategy, choose_targets

SR = 44100
A_MAJOR = Key(tonic=9, minor=False)


def harmony(seconds: float, rising_line: bool = False) -> HarmonyMap:
    """An A major chord throughout; optionally a guitar line that keeps rising."""
    frames = int(seconds * SR / HARMONY_HOP) + 1
    chord = np.full(12, 0.1)
    chord[[9, 1, 4]] = [1.0, 0.8, 0.8]  # A, C#, E
    # Low-mid-high: the guitar climbs ~4 semitones per second.
    line = np.linspace(45, 45 + 4 * seconds, frames) if rising_line else np.full(frames, 50.0)
    return HarmonyMap(
        chroma=np.repeat(chord[:, None], frames, axis=1),
        profile=chord,
        sr=SR,
        key=A_MAJOR,
        pitch_line=line,
    )


def notes_at(pitches, start_s=0.0, note_s=0.4, gap_s=0.1):
    out, t = [], start_s
    for p in pitches:
        out.append(SungNote(int(t * SR), int((t + note_s) * SR), p, np.full(30, p)))
        t += note_s + gap_s
    return out


def test_phrase_leaves_fitting_notes_like_nearest():
    notes = notes_at([57.1, 61.2, 63.9, 61.1])  # A C# E C#: all chord tones, in tune-ish
    h = harmony(3)
    nearest = choose_targets(notes, TargetStrategy.NEAREST, h)
    assert choose_targets(notes, TargetStrategy.PHRASE, h) == nearest == [57, 61, 64, 61]


def test_phrase_moves_clashing_note_with_the_guitar_direction():
    # Guitar rises; the middle note drops to F# (a weak, non-chord note) between A and C#.
    notes = notes_at([57.0, 54.6, 61.0])
    h = harmony(3, rising_line=True)
    assert choose_targets(notes, TargetStrategy.NEAREST, h)[1] == 54  # nearest keeps F#
    target = choose_targets(notes, TargetStrategy.PHRASE, h)[1]
    assert target == 57  # pulled up to the chord tone, moving with the guitar
    assert abs(target - 54.6) > 1  # beyond what NEAREST may do


def test_phrase_never_moves_beyond_a_minor_third():
    notes = notes_at([57.0, 52.2, 61.0])
    targets = choose_targets(notes, TargetStrategy.PHRASE, harmony(3, rising_line=True))
    assert all(abs(t - n.pitch) <= 3 for t, n in zip(targets, notes, strict=True))


PHRASE = [57.0, 59.0, 61.0, 64.0, 61.0, 59.0, 57.0]  # A B C# E C# B A


def repeated(times, wrong_at=None, wrong_pitch=None):
    notes = []
    for k, t0 in enumerate(times):
        pitches = list(PHRASE)
        if k == wrong_at:
            pitches[3] = wrong_pitch
        notes += notes_at(pitches, start_s=t0)
    return notes


def test_repetition_fixes_outlier_against_other_verses():
    # Verse 3 sings the top note as C# instead of E; three other verses agree on E.
    notes = repeated([0, 10, 20, 30], wrong_at=2, wrong_pitch=61.1)
    targets = choose_targets(notes, TargetStrategy.REPETITION, harmony(34))
    wrong = 2 * len(PHRASE) + 3
    assert targets[wrong] == 64
    others = [t for i, t in enumerate(targets) if i != wrong]
    assert others == [round(n.pitch) for i, n in enumerate(notes) if i != wrong]


def test_repetition_needs_enough_agreement():
    # Only one other verse: not enough evidence to change a note.
    notes = repeated([0, 10], wrong_at=1, wrong_pitch=61.1)
    targets = choose_targets(notes, TargetStrategy.REPETITION, harmony(14))
    assert targets[len(PHRASE) + 3] == 61


@pytest.mark.parametrize("strategy", list(TargetStrategy))
def test_strategies_without_harmony_fall_back_to_nearest(strategy):
    notes = notes_at([57.3, 60.6])
    assert choose_targets(notes, strategy, None) == [57, 61]


def modulating_harmony(transposed: bool) -> HarmonyMap:
    """A-D-E in A major for 20 s, then the same progression 3 st up in C major (or not)."""
    from songpolish.analysis import KeySpan

    per_s = SR / HARMONY_HOP
    chords = [[9, 1, 4], [2, 6, 9], [4, 8, 11]]  # A, D, E
    frames = []
    for second in range(40):
        c = np.full(12, 0.1)
        root = chords[second % 3]
        c[[(p + 3) % 12 for p in root] if transposed and second >= 20 else root] = 1.0
        frames += [c] * int(per_s)
    chroma = np.array(frames).T
    c_major = Key(tonic=0, minor=False)
    split = 20 * SR
    return HarmonyMap(
        chroma=chroma,
        profile=chroma.mean(axis=1),
        sr=SR,
        key=A_MAJOR,
        pitch_line=np.full(chroma.shape[1], 50.0),
        keys=(
            KeySpan(0, split, A_MAJOR, chroma[:, : chroma.shape[1] // 2].mean(axis=1)),
            KeySpan(split, 40 * SR, c_major, chroma[:, chroma.shape[1] // 2 :].mean(axis=1)),
        ),
    )


MELODY = [57.0, 59.0, 61.0, 62.0, 64.0]  # A B C# D E, sung in A major


def test_phrase_after_key_change_follows_its_earlier_repeat_transposed():
    # The chorus comes back 3 st up, but the singer only went 1 st up: follow the chorus.
    notes = notes_at(MELODY, start_s=2.0) + notes_at([p + 1 for p in MELODY], start_s=22.0)
    h = modulating_harmony(transposed=True)
    targets = choose_targets(notes, TargetStrategy.PHRASE, h)
    assert targets[:5] == MELODY
    assert targets[5:] == [p + 3 for p in MELODY]


def test_key_change_without_transposed_guitar_keeps_nearest_notes():
    # Same key signature change, but the accompaniment is not the earlier one moved up.
    notes = notes_at(MELODY, start_s=2.0) + notes_at([p + 1 for p in MELODY], start_s=22.0)
    h = modulating_harmony(transposed=False)
    targets = choose_targets(notes, TargetStrategy.PHRASE, h)
    assert targets[5:] != [p + 3 for p in MELODY]
    assert all(abs(t - n.pitch) <= 1 for t, n in zip(targets[5:], notes[5:], strict=True))
