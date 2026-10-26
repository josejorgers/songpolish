# CLAUDE.md

Guidance for working in this repository.

## What this is

SongPolish: a cross-platform desktop app (PySide6) plus CLI that automatically and
**gently** polishes singer-songwriter recordings (voice + acoustic guitar). It does
noise cleanup, natural per-note pitch correction, reference-tuning fix, auto-EQ,
de-essing, compression, light reverb and loudness mastering.

**Product principle: preserve the performance.** Every change must be subtle by
default, automatic, and controlled by at most a handful of 0..1 knobs. When in
doubt, do less. A result that sounds "processed" is a bug.

## Commands

```bash
uv sync --extra separation      # full install (Demucs + CPU PyTorch)
uv sync                         # light install (no separation)
uv run pytest                   # all tests (~10 s); separation test skips if extra missing
uv run pytest -k pitch          # subset
uv run ruff check . && uv run ruff format .
uv run songpolish <take.wav>    # CLI
uv run songpolish <take.wav> --compare   # one version per wrong-note strategy, for listening tests
uv run songpolish --thresholds           # expert thresholds; override with --set name=value
uv run songpolish-gui           # desktop app
QT_QPA_PLATFORM=offscreen uv run python ...   # headless Qt (CI / smoke tests)
git tag v<version> && git push origin v<version>   # installers for all 3 OSes -> draft release
uv sync --extra separation --group notebooks && uv run jupyter lab notebooks/   # explanatory notebooks
```

`notebooks/` explains the whole method for non-audio readers (01–08, see its README). They call the
engine's real functions, so **after changing an engine API or behaviour, re-run them**
(`jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb`) and check that the text still matches
the printed numbers. They are excluded from ruff; `nbtools.py` is not.

## Architecture

```
src/songpolish/
  settings.py    Settings dataclass (all user knobs, 0..1) + PRESETS (natural/balanced/polished)
  thresholds.py  registry of expert overrides for engine constants (unit, safe range, help)
  audio_io.py    load/save; internal format = float32 (channels, samples) @ 44.1 kHz
  separation.py  optional Demucs voice/accompaniment split (lazy torch import)
  analysis.py    tuning offset, key (and key changes over time), LUFS, octave-band spectrum
  pitch.py       note detection (pYIN) → one shift per note → leakage reclaim → per-note
                 Rubber Band rendering
  melody.py      target note per sung note: NEAREST / PHRASE (Viterbi) / REPETITION strategies
  tone.py        per-source chain: highpass → denoise → auto-EQ → de-ess → compressor
  mastering.py   stem balance, reverb, bus glue, loudness + true-peak limiting
  pipeline.py    process(Inputs, Settings, out_dir, progress) → Result   ← single entry point
  cli.py         argparse wrapper around pipeline.process
  gui/app.py     PySide6 window: batch of songs on a QThread, thresholds panel, saved
                 setups (QSettings + JSON), A/B player per edited song / version
packaging/       PyInstaller spec, entry point (+ --self-test), AppImage / dmg / Inno Setup;
                 built by .github/workflows/release.yml (see packaging/README.md)
tests/           synthetic signals with known ground truth (conftest.py builds them)
```

Pipeline flow: load → (separate a single take) → measure tuning on the guitar →
harmony map (chords + key over time) → plan note targets → move leaked voice out of the
guitar stem → pitch-correct the voice relative to the guitar's tuning → mix → tone chain →
master → export polished, loudness-matched original, stems and report JSON.

**A separated take is never rebuilt from its stems.** The polished song is
`take + (corrected_vocal − vocal)` (`pipeline.with_correction`), then one MIX tone chain
(and the A440 retune) on the whole. Outside corrected notes it is the recording, bit for
bit. Why: rebuilt from Demucs stems, the real take had a "singing in front of a fan"
voice (separation flicker) and a guitar that sounded "far away" while the voice sang (the
guitar stem lost 2.2 dB of 1–3 kHz in sung parts, 0 elsewhere). In the mix these cancel
only if both stems are treated identically. The price: no separate voice/guitar EQ, and
`vocal_balance_db` (default 0) adds separated voice on top. Exported stems are for
inspection only and keep the separator's artefacts. Two-track input (real separate mics)
still uses per-source chains and `mastering.mix`. A RoFormer separator (BS-RoFormer
viperx 1297) took ~47 min per 4-min song on CPU vs ~6 for Demucs; not worth it here.

### Rules that keep the design sound

- **The engine has no UI dependencies.** `gui/` and `cli.py` only call
  `pipeline.process`. This is what lets us put the engine behind an API for mobile
  later. Never import Qt outside `gui/`.
- **Audio shape convention:** always `(channels, samples)` float32 at `WORK_SR`.
  Mono is `(1, n)`. Functions keep the input's channel count unless documented.
- **Every DSP stage measures before acting** and scales by a 0..1 amount. Hard caps
  (e.g. `MAX_BOOST_DB`, `TOLERANCE`, never a full noise gate) protect naturalness.
  Keep tuning constants as named module-level constants with a comment saying *why*.
- **Optional heavy deps are imported lazily** (`separation.py`). The app must run
  without PyTorch and degrade gracefully, with a note in `report["notes"]`.
- **Lengths are preserved:** no stage may shift or change timing (tests check
  alignment). Pitch shifting uses `stretch_factor=1.0`.
- **Honest A/B:** the reference file is loudness-matched with only a transparent
  limiter. Never compare against a quieter original.
- **Expert thresholds go through `thresholds.py`.** The engine keeps reading its
  module constants; `pipeline.process` wraps each job in `thresholds.applied(...)`, which
  sets the overrides and restores them (under a lock, so jobs with overrides never
  overlap). To expose a constant, add a `Threshold` (engine units, shown units via
  `scale`, a range that keeps it natural, a help text); never bind such a constant at
  import time (default arguments, `from x import CONST`, derived constants), or the
  override silently does nothing. `Settings.thresholds` holds only changed values.
- **UIs hand back strings.** Qt item data loses the `StrEnum` type, and the engine
  compares strategies with `is`; `Settings.__post_init__` normalises them. (Before that,
  the GUI crashed on every run in `to_dict`.)
- **Report what happened.** Anything detected or skipped goes into the report
  dict; the GUI and CLI render it.

### Pitch correction (the core feature)

`pitch.plan_corrections` runs pYIN, builds a note-number track relative to the
reference tuning, and finds note centres with a median filter. It splits where the
nearest target changes, then **re-merges pieces less than 0.6 st apart**, because
vibrato near a half-semitone otherwise splits one note into opposite-target halves.
Each stable note (≥80 ms, std ≤ 0.7 st) gets `strength × (target − trimmed mean)`.
Errors under `TOLERANCE` (8 cents) are left alone. **One constant shift per note, never
pieces inside a note.** Drift correction (100 ms pieces with their own shifts) was
removed: two Rubber Band renders never line up in phase, so each crossfade between them
blurs the timbre. At ~11 crossfades a second, held notes sounded like "singing in front
of a fan" (a timbre-jump peak at the piece rate in 43 of 68 held notes of the real take,
+3.9 dB median; none with one shift per note). Drift is still measured for the report.
Use the trimmed mean, not the median: pYIN quantises to 10 cents, vibrato dithers
that, and the mean is accurate.

**Targets follow the guitar (`PitchMode.HARMONY`, the default).** `analysis.harmony_map`
gives a smoothed chroma of the accompaniment over time plus the detected key.
`_choose_target` picks among semitones within `HARMONY_MAX_MOVE` (1 st) using a
linear cost: distance minus a bonus for chord weight now and a bonus for being in
key. Effects: a note more than ~35 cents from an off-chord neighbour goes to the
chord tone, and an out-of-key note is kept only if sung within ~18 cents (it's
intentional); otherwise it goes to the nearest in-key note. Why: the singer this
was tuned on sat 20–30 cents sharp of the guitar with wide scatter, so "nearest
semitone" locked ~35% of notes onto out-of-key neighbours (A# over an A chord).
Keep the cost linear: a squared distance cost never lets the chord win.

**Wrong notes (`melody.TargetStrategy`).** Pitch correction can't fix a *wrong* note
(singer sang a different note than the melody; a real user case was "high-low-high"
where the melody ascends). Three strategies are kept side by side because only
listening decides between them, so do not delete one without a listening test:
- `NEAREST`: moves at most 1 st and never changes the melody (the default).
- `PHRASE`: Viterbi over each phrase (notes separated by less than 0.6 s). Clashing notes may move ≤3 st
  to *in-key* notes; notes that fit the chord and key are anchors and never move. Transitions penalise leaps over a whole tone and moving against
  the guitar's pitch line (`HarmonyMap.line`, the CQT energy centroid) or against the sung direction.
- `REPETITION`: songs repeat. Each note's context (chords + sung pitch classes in ±2 s)
  is matched against the other notes ≥6 s away. Phrases are aligned by their
  neighbouring notes, with one match per repetition. When ≥2 repetitions agree (≥60%)
  on a different note within 3.5 st, the note moves there.
In every strategy, a note moved to a *different* note than the sung nearest semitone
(`round(pitch)`, not the nearest *allowed* note) gets the full shift. Strength only
scales fine-tuning, since 80% of a wrong-note fix still sounds wrong.

**Modulations.** The key is followed over time: `analysis.key_spans` scores 8 s windows
against the 12 key signatures (a relative major/minor pair is one state) and runs a
Viterbi with `KEY_CHANGE_COST` per change, so one chord leaning to a neighbouring key never
switches but a sustained modulation does. `HarmonyMap.key_at` gives the key at a note;
strict key mode, the in-key bonus and the background profile in `weights` all use it.
Real case: A major → A minor/C major at 3:19; with a single song-wide key, strict mode
dragged a third of the notes after it onto C# and G#. This only shows on full songs.
**Repeats in a new key** (`melody._modulated_repeat_targets`, PHRASE and REPETITION only).
The real ending repeated the chorus 3 st up (A → C major), but the singer went only ~1 st
up; snapping to the nearest C-major note left several notes a scale step low, which the
musician heard as a drop where the melody rises (3:27). Each phrase after a key change is
matched (sung contour, any transposition) against passages in another key signature, and
its notes take the earlier repetition's targets + the signature interval, in the octave
sung. The guitar must confirm the interval: with each note's mean removed (drones!), the
earlier accompaniment shifted by it must fit the new passage best of all 12 shifts. A
parallel-minor change has the same signature interval but fails this check.
Beware of drones: that guitarist lets the open A string ring through the whole song, so
A is the strongest chroma at almost every moment (with A#/G# beside it), even under C, F
or G chords. Chord weights near a drone note are unreliable.

**Leakage (ghost voice).** Demucs leaves part of the voice in the guitar stem (median
−12 dB on moved notes in the real take, as loud as the voice on a quarter of them). Moving
a note left the old wrong note behind in the guitar, which the musician heard as notes
sounding "different/artificial". `pitch.reclaim_leakage` moves the voice's harmonics (a
comb mask following the note's own pitch track) from the guitar stem into the vocal stem
before shifting, only on notes moved ≥ `LEAK_MIN_SHIFT` (0.5 st, i.e. moved to another
note: on fine-tuned notes the guitar often plays the same note). The stems still sum to
the same signal. The pipeline plans every strategy first, reclaims once for the union,
then renders. **Never take more than `LEAK_MAX_OF_VOICE` (half the voice's magnitude, per
bin).** Anything louder is the guitar (a chord tone, or guitar under a phantom note), and
moving it pitch-shifts the guitar, which sounds robotic. Real case: a held F over an F chord, moved
to E, took the guitar's F with it; and phantom notes took guitar 50 dB louder than the "voice".

**Phantom notes.** The separated vocal stem holds faint guitar bleed where nobody sings,
and pYIN finds notes in it (a third of all detected notes in the real take, 40–90 dB
below the singing). `detect_notes` drops notes more than `PHANTOM_DB` (30 dB) below the
loud notes. Real notes were all within 20 dB, so the gap is wide.

Lesson from the first real user: wrong notes are often sung *accurately*. The
singer's scale steps were a semitone too narrow (E–F–F#–G–A over A major instead of
E–F#–G#–A). The chord-aware default keeps an accurately sung out-of-key note as
"intentional", so it leaves exactly those errors. `PitchMode.KEY` (strict: in-key
targets only) fixes them. First listening test: the musician preferred PHRASE + strict
key over all six combinations; the default has not been switched yet.

`pitch.apply_corrections` renders every piece with its own **constant** Rubber Band
shift (plus `base_shift` for the A440 retune) and crossfades it in (20 ms sin² ramps
that sum to 1).

**Never use `pedalboard.time_stretch` with a per-sample `pitch_shift_in_semitones`
array.** Every change in value emits an artefact burst about 18 dB below the signal
(an audible chirp at every corrected note, which users heard as "repeating noise"),
and it applies changes about 87 ms early. `test_note_corrections_add_no_artifacts`
guards against this.

### Auto-EQ

Octave-band LTAS vs `TARGET_SPECTRUM`, moved `tone_match` of the way, capped at
−8/+6 dB. Bands that are nearly empty (`EMPTY_BAND_DB`) are never boosted. Two
re-measure passes correct for overlap between neighbouring filters. Phone takes are
typically ~10 dB boomy (200–400 Hz) and ~9 dB dull: small caps make the EQ a no-op.

**Boosts are limited by each band's signal-to-noise ratio** (`BOOST_MIN_SNR_DB` 30 → no
boost, full boost from `BOOST_FULL_SNR_DB` 45). A "dull" phone take's treble is mostly
hiss: the real take had only 20–28 dB between music and hiss in every band, and the
+6 dB treble boost raised the hiss ~9 dB ("sounds like an old recorder"). The SNR is
loud frames vs the quietest 2% (`analysis.band_snr_db`). Because the stem is
loudness-normalised afterwards, cutting boom lifts every other band: in the treble
(≥ `HISS_FROM_HZ`) the boost limit holds *relative to the overall level change*, so the
boom still goes but the hiss does not come up. Measure on a signal with real pauses (the
take or a real track), never on a separated stem: stems hide the noise floor.

## Diagnosing on real recordings

Unit tests can't hear. "Out of tune" means *relative to the guitar's chords*, not to
A440: check sung notes against `harmony_map` (in chord / in key / out of key). When a
user reports "still out of tune" or "noise", measure
on their file: cache the Demucs stems once, then compare stages using
`plan_corrections(output, 1.0)`. Its report gives per-note error and held-note
drift with the same note detection as the engine; don't hand-roll segmentation.
Check EQ decisions by printing the measured band levels against `TARGET_SPECTRUM`.

## Testing conventions

- Use synthetic signals from `tests/conftest.py` (`sung_notes`, `strummed_progression`),
  because they carry exact ground truth (pitch, tuning offset, key).
- Do **not** use Karplus-Strong for pitch-accurate guitars: integer delay lengths
  make it out of tune. Use the additive `plucked_chord`.
- Demucs does not recognise synthetic tones as voice. Separation quality can only
  be judged on real recordings (e.g. `librosa.ex("fishin")`, voice + guitar).
  Keep such checks out of the unit-test suite, or skip them when offline.
- Test behaviour (pitch moved toward target, loudness hit, lengths preserved,
  in-tune notes untouched), not implementation details.
- After any DSP change, also *listen* to a real take; numbers alone don't catch
  artefacts. Say so explicitly if you could not listen.

## Gotchas

- Constant-shift `pedalboard.time_stretch` keeps length and stays aligned to within
  about 2 ms, which is good enough for 20 ms crossfades. See above for why not to use
  per-sample arrays.
- pedalboard cannot decode AAC/.m4a on Linux. `audio_io.load` falls back to the
  ffmpeg bundled with `imageio-ffmpeg` (all OSes).
- Demucs `apply_model` callback dicts have no `audio_length`; compute progress from
  `segment_offset / length`. Use `shifts=0` for reproducible output.
- A near-silent separated vocal stem means "no voice found". The pipeline falls back
  to whole-take processing rather than boosting artefacts by 50 dB.
- `librosa.estimate_tuning` returns a fraction of a semitone in [-0.5, 0.5).
- PyTorch CPU wheels come from the `pytorch-cpu` index on Linux/Windows
  (`[tool.uv.sources]` in pyproject); macOS uses PyPI wheels.

- Installers bundle the Demucs model (`separation.bundled_model_dir`). After adding a
  dependency, rebuild and run `SongPolish --self-test`: PyInstaller misses data files and
  lazily imported modules silently, and only the frozen app shows it.

## Style

Python 3.12, type hints, dataclasses, `ruff` (line length 100). Match the
surrounding code: short docstrings that say *why*, named constants for every
magic number, and no comments that restate the code.
