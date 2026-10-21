import json

import librosa
import numpy as np
import pytest
from conftest import SR, plucked_chord, strummed_progression, sung_notes
from scipy.stats import trim_mean

from songpolish import analysis, mastering, melody, pitch, separation, tone
from songpolish.audio_io import load, save
from songpolish.pipeline import Inputs, process
from songpolish.settings import PRESETS, PitchMode, Settings, preset


def median_midi(audio):
    y = librosa.resample(audio[0], orig_sr=SR, target_sr=22050)
    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=1050, sr=22050, frame_length=1024)
    return float(trim_mean(librosa.hz_to_midi(f0[voiced]), 0.15))


def test_settings_roundtrip():
    s = preset("polished", pitch_mode=PitchMode.KEY)
    assert Settings.from_dict(s.to_dict()) == s
    with pytest.raises(ValueError):
        preset("nope")


def test_shift_pitch_keeps_length_and_alignment():
    x = sung_notes([60, 64])
    y = pitch.shift_pitch(x, 0.25)
    assert y.shape == x.shape
    # Amplitude envelope should line up (no added latency).
    ex = np.abs(x[0]).reshape(-1, 441).mean(1)
    ey = np.abs(y[0]).reshape(-1, 441)[: len(ex)].mean(1)
    lag = np.argmax(np.correlate(ey - ey.mean(), ex - ex.mean(), "full")) - (len(ex) - 1)
    assert abs(lag) <= 2  # within 20 ms


@pytest.mark.parametrize("error_cents", [30, -35])
def test_pitch_correction_moves_notes_toward_grid(error_cents):
    x = sung_notes([60 + error_cents / 100, 64 + error_cents / 100, 67 + error_cents / 100])
    y, report = pitch.correct_pitch(x, strength=1.0)
    assert report.notes_total == 3
    assert report.notes_corrected == 3
    assert report.mean_abs_error_cents == pytest.approx(abs(error_cents), abs=6)
    for k, target in enumerate([60, 64, 67]):
        note = y[:, int((k * 0.7 + 0.1) * SR) : int((k * 0.7 + 0.5) * SR)]
        assert abs(median_midi(note) - target) < 0.08  # within 8 cents after correction


def test_short_notes_are_fully_corrected():
    # Real melodies are mostly short notes; the whole note must get its shift.
    x = sung_notes([60.3, 62.3, 64.3, 65.3, 67.3], note_s=0.15, gap_s=0.0, vibrato_cents=0)
    y, report = pitch.correct_pitch(x, strength=1.0)
    assert report.notes_corrected >= 4
    for k, target in enumerate([60, 62, 64, 65, 67]):
        note = y[:, int((k * 0.15 + 0.03) * SR) : int((k * 0.15 + 0.12) * SR)]
        assert abs(median_midi(note) - target) < 0.1


def test_note_corrections_add_no_artifacts():
    # pedalboard's per-sample shift arrays chirp at every change; per-note rendering must not.
    from pedalboard import HighpassFilter

    t = np.arange(4 * SR) / SR
    x = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)[None]
    notes = [
        pitch.NoteShift(int((0.4 + k * 0.45) * SR), int((0.6 + k * 0.45) * SR), 0.3 * (-1) ** k)
        for k in range(7)
    ]

    def hf_peak_db(y):
        h = HighpassFilter(2000)(HighpassFilter(2000)(y, SR), SR)[0][SR // 2 : -SR // 2]
        h = h[: len(h) // 441 * 441].reshape(-1, 441)
        return 20 * np.log10(np.sqrt((h**2).mean(1)).max() + 1e-12)

    clean = hf_peak_db(pitch.shift_pitch(x, 0.3))
    assert hf_peak_db(pitch.apply_corrections(x, notes)) < clean + 6


def test_partial_strength_is_partial():
    x = sung_notes([62.4])
    notes, report = pitch.plan_corrections(x, strength=0.5)
    assert report.notes_corrected == 1
    assert all(n.semitones == pytest.approx(-0.2, abs=0.04) for n in notes)


def test_held_note_gets_one_constant_shift():
    # Cutting a sagging note into differently shifted pieces crossfaded every 0.1 s made a
    # "fan" flutter in the timbre; the sag is reported, not corrected.
    x = sung_notes([60.3], note_s=1.2, vibrato_cents=10, drift_cents=-40)
    shifts, report = pitch.plan_corrections(x, strength=1.0)
    assert len(shifts) == 1
    assert report.mean_abs_drift_cents > 10


def test_in_tune_singing_is_untouched():
    x = sung_notes([60, 62, 64])
    out, report = pitch.correct_pitch(x, strength=1.0)
    assert report.notes_corrected == 0
    assert out is x


def test_key_mode_snaps_to_scale():
    # 61.4 is nearest to C#, but in C major the target must be D (62).
    x = sung_notes([61.6])
    key = analysis.Key(tonic=0, minor=False)
    notes, _ = pitch.plan_corrections(x, strength=1.0, key=key)
    assert notes[0].semitones == pytest.approx(0.4, abs=0.05)


def test_tuning_offset_detected():
    g = strummed_progression(offset_cents=22, repeats=1)
    assert analysis.tuning_offset_cents(g) == pytest.approx(22, abs=5)


def test_key_detection():
    g = strummed_progression(repeats=1)
    assert analysis.detect_key(g).name in {"C major", "A minor"}


def test_master_hits_loudness_and_peak():
    x = np.concatenate([strummed_progression(repeats=2)] * 2) * 0.2
    y = mastering.master(x, target_lufs=-14, true_peak_db=-1.0)
    assert y.shape[0] == 2
    assert analysis.loudness_lufs(y) == pytest.approx(-14, abs=0.6)
    assert 20 * np.log10(np.abs(y).max()) <= -1.0


def test_tone_chain_is_gentle():
    v = sung_notes([60, 64, 67])
    out = tone.process_source(v, tone.Source.VOCAL, PRESETS["polished"])
    assert out.shape == v.shape
    assert np.isfinite(out).all()
    # Spectral balance may move, but not by more than the EQ cap per band.
    before = analysis.band_levels_db(v, tone.EQ_BANDS[1:-1])
    after = analysis.band_levels_db(out, tone.EQ_BANDS[1:-1])
    assert np.max(np.abs(after - before)) < tone.MAX_CUT_DB + 3


def test_pipeline_two_tracks(tmp_path, out_dir):
    guitar = strummed_progression(offset_cents=15, repeats=2)
    vocal = sung_notes([60.3, 64.3, 67.3, 64.3, 60.3, 62.3])
    vocal = np.pad(vocal, ((0, 0), (0, guitar.shape[1] - vocal.shape[1])))
    save(tmp_path / "v.wav", vocal)
    save(tmp_path / "g.wav", guitar)

    stages = []
    result = process(
        Inputs(vocal=tmp_path / "v.wav", guitar=tmp_path / "g.wav"),
        Settings(),
        out_dir,
        progress=lambda s, f: stages.append(f),
    )
    assert result.output.is_file() and result.original_matched.is_file()
    assert set(result.stems) == {"vocal", "guitar"}
    assert stages == sorted(stages) and stages[-1] == 1.0
    r = json.loads((out_dir / "v_report.json").read_text())
    assert r["retuned_to_a440"] is True
    assert r["reference_tuning_cents"] == pytest.approx(15, abs=5)
    assert r["output_lufs"] == pytest.approx(-14, abs=1.0)


def test_pipeline_single_take_without_separation(tmp_path, out_dir):
    guitar = strummed_progression(repeats=2)
    vocal = sung_notes([60, 64, 67, 64, 60, 62])
    take = guitar.copy()
    take[:, : vocal.shape[1]] += vocal
    save(tmp_path / "take.wav", take)
    result = process(Inputs(mix=tmp_path / "take.wav"), Settings(separate=False), out_dir)
    assert result.output.is_file()
    assert result.report["separated"] is False
    assert result.stems == {}


def test_inputs_validation():
    with pytest.raises(ValueError):
        Inputs().validate()
    with pytest.raises(ValueError):
        Inputs(mix="a.wav", vocal="b.wav").validate()


def test_ab_reference_is_loudness_matched(tmp_path, out_dir):
    take = strummed_progression(repeats=2)
    save(tmp_path / "take.wav", take * 0.5)
    result = process(Inputs(mix=tmp_path / "take.wav"), Settings(separate=False), out_dir)
    ref = analysis.loudness_lufs(load(result.original_matched))
    assert ref == pytest.approx(result.report["output_lufs"], abs=0.5)


@pytest.mark.skipif(not separation.is_available(), reason="separation extra not installed")
def test_separation_without_voice_falls_back(tmp_path, out_dir):
    # Demucs does not hear a synthetic tone as a voice: the vocal stem is near-empty,
    # and the pipeline must not boost it to mix level.
    save(tmp_path / "take.wav", strummed_progression(repeats=1))
    result = process(Inputs(mix=tmp_path / "take.wav"), Settings(), out_dir)
    assert result.report["separated"] is False
    assert any("No clear voice" in n for n in result.report["notes"])


def test_loads_aac_m4a(tmp_path):
    # Phones record AAC .m4a, which pedalboard cannot decode on Linux: ffmpeg fallback.
    import subprocess

    import imageio_ffmpeg

    save(tmp_path / "x.wav", sung_notes([60]))
    m4a = tmp_path / "x.m4a"
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(tmp_path / "x.wav"),
         "-c:a", "aac", str(m4a)],
        check=True,
    )  # fmt: skip
    audio = load(m4a)
    assert audio.shape[0] == 1
    assert abs(audio.shape[1] / SR - 0.7) < 0.1


def _over_c_chord(sung_midi):
    """A vocal note held over the first chord (C major) of the test progression."""
    guitar = strummed_progression(repeats=1)
    vocal = sung_notes([sung_midi], note_s=0.8, gap_s=0.1, vibrato_cents=10)
    harmony = analysis.harmony_map(guitar)
    notes, report = pitch.plan_corrections(vocal, strength=1.0, harmony=harmony)
    return notes, report


def test_harmony_pulls_sharp_note_back_to_chord_tone():
    # E sung 60 cents sharp: nearest semitone is F, but the guitar plays C-E-G.
    notes, report = _over_c_chord(64.6)
    assert report.notes_retargeted == 1
    assert notes[0].semitones == pytest.approx(-0.6, abs=0.06)  # back down to E


def test_harmony_keeps_clearly_sung_passing_note():
    # D over a C chord, sung nearly in tune: a passing note, not a mistake.
    notes, report = _over_c_chord(62.12)
    assert report.notes_retargeted == 0
    assert all(abs(n.semitones) < 0.2 for n in notes)


def test_harmony_never_moves_more_than_limit():
    notes, _ = _over_c_chord(63.5)  # halfway D#/E: E is a chord tone, 0.5 away
    assert all(abs(n.semitones) <= melody.HARMONY_MAX_MOVE for n in notes)


def test_harmony_moves_inaccurate_out_of_key_note_into_key():
    # C# is outside C major and not in the chord: sung 30 cents sharp, it is a flat D.
    notes, _ = _over_c_chord(61.3)
    assert notes[0].semitones == pytest.approx(0.7, abs=0.06)


def test_harmony_keeps_accurate_out_of_key_note():
    # Sung dead-on, an out-of-key note is intentional (blue note): leave it.
    notes, report = _over_c_chord(61.05)
    assert report.notes_retargeted == 0
    assert all(abs(n.semitones) < 0.15 for n in notes)


def test_pipeline_compare_renders_one_version_per_strategy(tmp_path, out_dir):
    guitar = strummed_progression(repeats=2)
    vocal = sung_notes([60.3, 64.6, 67.3, 64.3, 60.3, 62.3])
    vocal = np.pad(vocal, ((0, 0), (0, guitar.shape[1] - vocal.shape[1])))
    save(tmp_path / "v.wav", vocal)
    save(tmp_path / "g.wav", guitar)
    strategies = list(melody.TargetStrategy)
    result = process(
        Inputs(vocal=tmp_path / "v.wav", guitar=tmp_path / "g.wav"),
        Settings(),
        out_dir,
        strategies=strategies,
    )
    assert set(result.variants) == {s.value for s in strategies}
    assert all(p.is_file() for p in result.variants.values())
    assert set(result.report["strategies"]) == {s.value for s in strategies}
    assert {"guitar", "vocal_nearest", "vocal_phrase", "vocal_repetition"} == set(result.stems)


def test_key_mode_moves_wrong_note_fully():
    # An accurately sung C# in C major is a wrong note: key mode must land it on C or D
    # (both a semitone away), not 80% of the way there.
    x = sung_notes([61.0], vibrato_cents=0)
    notes, report = pitch.plan_corrections(x, 0.8, key=analysis.Key(tonic=0, minor=False))
    assert report.notes_retargeted == 1
    assert all(abs(n.semitones) == pytest.approx(1.0, abs=0.05) for n in notes)


A_MAJOR_CHORDS = [[45, 52, 57, 61, 64], [50, 57, 62, 66, 69], [52, 59, 64, 68, 71]]  # A D E
C_MAJOR_CHORDS = [[48, 52, 55, 60, 64], [43, 47, 50, 55, 59], [41, 48, 53, 57, 60]]  # C G F


def modulating_song(seconds_each=30):
    """A major (A-D-E) for ``seconds_each`` s, then C major (C-G-F). Returns (1, n)."""
    parts = []
    for chords in (A_MAJOR_CHORDS, C_MAJOR_CHORDS):
        for i in range(seconds_each):
            parts.append(plucked_chord(chords[i % 3], 1.0, seed=i))
    return (0.4 * np.concatenate(parts))[None, :].astype(np.float32)


def test_harmony_follows_a_modulation():
    h = analysis.harmony_map(modulating_song(), SR)
    assert [set(s.key.pitch_classes) for s in h.keys] == [
        set(analysis.Key(9, False).pitch_classes),
        set(analysis.Key(0, False).pitch_classes),
    ]
    assert abs(h.keys[1].start / SR - 30) < 3
    assert h.key_at(40 * SR, 41 * SR).pitch_classes == h.keys[1].key.pitch_classes


def test_harmony_without_modulation_has_one_key():
    h = analysis.harmony_map(strummed_progression(repeats=12), SR)
    assert len(h.keys) == 1


def test_key_mode_uses_the_key_after_a_modulation():
    # An accurately sung G is right in C major; in A major strict mode moves it (F# or G#).
    song = modulating_song()
    voice = np.zeros_like(song)
    for t in (10, 40):
        note = sung_notes([67.0], note_s=0.8, vibrato_cents=0)
        voice[:, t * SR : t * SR + note.shape[1]] = note
    h = analysis.harmony_map(song, SR)
    shifts, report = pitch.plan_corrections(voice, 0.8, key=h.key, harmony=h)
    moved = {round(s.start / SR): s.semitones for s in shifts}
    assert abs(moved[10]) == pytest.approx(1.0, abs=0.05)
    assert abs(moved.get(40, 0.0)) < 0.1
    assert report.notes_retargeted == 1


def test_leakage_moves_with_the_corrected_note():
    # Separation leaves some voice in the guitar stem; on a moved note it must go too.
    voice = sung_notes([61.0], note_s=1.2, vibrato_cents=0)
    guitar = (0.3 * plucked_chord([45, 52, 57, 64], voice.shape[1] / SR))[None].astype(np.float32)
    accompaniment = guitar + 0.3 * voice
    notes = pitch.detect_notes(voice)
    moved = [pitch.NoteShift(n.start, n.end, 1.0) for n in notes]
    v2, a2, count = pitch.reclaim_leakage(voice, accompaniment, notes, moved)
    assert count == len(notes) == 1
    np.testing.assert_allclose(v2 + a2, voice + accompaniment, atol=1e-4)

    def db(x, ref):
        return 10 * np.log10(np.sum(x**2) / np.sum(ref**2))

    assert db(a2 - guitar, 0.3 * voice) < -10  # most of the leftover voice is gone
    assert db(v2 - voice - 0.3 * voice, guitar) < -10  # and little guitar went with it

    fine_tuned = [pitch.NoteShift(n.start, n.end, 0.3) for n in notes]
    _, untouched, count = pitch.reclaim_leakage(voice, accompaniment, notes, fine_tuned)
    assert count == 0 and untouched is accompaniment


def test_eq_does_not_boost_hiss():
    # A dull take: the treble boost it "needs" must not happen when the treble is mostly hiss.
    from pedalboard import HighShelfFilter

    rng = np.random.default_rng(0)
    # Broadband "notes" (0.6 s on, 0.3 s off), so every band has music in it.
    on = (np.arange(20 * SR) % int(0.9 * SR)) < 0.6 * SR
    music = (0.1 * rng.standard_normal(on.size) * on)[None].astype(np.float32)
    dull = HighShelfFilter(3000, gain_db=-25, q=0.7)(music, SR)
    hiss = (0.003 * rng.standard_normal(dull.shape)).astype(np.float32)

    def treble_gain(x):
        board = tone.auto_eq(x, tone.Source.VOCAL, 1.0, SR)
        return max(f.gain_db for f in board if f.cutoff_frequency_hz >= 6000)

    assert treble_gain(dull + 1e-5 * hiss) > 3  # clean: brightening is fine
    assert treble_gain(dull + hiss) <= 0.0  # hissy: never boosted


def test_faint_bleed_in_the_vocal_stem_is_not_a_note():
    # Separated stems hold faint guitar where nobody sings; pYIN finds "notes" there.
    loud = sung_notes([60, 62])
    faint = 10 ** (-50 / 20) * sung_notes([64, 65, 67])
    notes = pitch.detect_notes(np.concatenate([loud, faint, loud], axis=1))
    assert [round(n.pitch) for n in notes] == [60, 62, 60, 62]


def test_leakage_never_takes_the_guitar_playing_the_same_note():
    # The guitar plays the sung note loudly: moving "leakage" would pitch-shift the guitar.
    voice = sung_notes([60.0], note_s=1.2, vibrato_cents=0)
    guitar = (2.0 * plucked_chord([48, 55, 60, 64], voice.shape[1] / SR))[None].astype(np.float32)
    notes = pitch.detect_notes(voice)
    moved = [pitch.NoteShift(n.start, n.end, 1.0) for n in notes]
    v2, _, _ = pitch.reclaim_leakage(voice, guitar, notes, moved)
    taken = v2 - voice
    assert 10 * np.log10(np.sum(taken**2) / np.sum(voice**2)) < -6


def test_correction_changes_the_take_only_inside_corrected_notes():
    # Rebuilding a separated take from its stems exposes separation artefacts everywhere;
    # adding only the correction leaves every uncorrected sample exactly as recorded.
    from songpolish.pipeline import with_correction

    vocal = sung_notes([60.3, 62.0, 64.4])
    guitar = strummed_progression()[:, : vocal.shape[1]]
    take = vocal + guitar
    note = pitch.NoteShift(int(0.75 * SR), int(1.25 * SR), -0.3)
    corrected = pitch.apply_corrections(vocal, [note])
    out = with_correction(take, vocal, corrected)
    margin = int(pitch.CROSSFADE_S * SR)
    outside = np.ones(take.shape[1], bool)
    outside[note.start - margin : note.end + margin] = False
    np.testing.assert_array_equal(out[:, outside], take[:, outside])
    assert not np.allclose(out[:, ~outside], take[:, ~outside])
