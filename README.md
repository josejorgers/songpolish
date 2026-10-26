# SongPolish

**One-click, gentle polish for singer-songwriter recordings.**

You sing and play guitar, you record a take (on a phone, a laptop mic or a USB
interface), and you want it to sound *better*. Not "AI-generated", and not
auto-tuned into a robot. SongPolish does the boring engineering for you:

- 🎤 **Pitch correction that follows your chords.** The app listens to what
  the guitar plays at each moment and moves each sung note toward the matching
  chord or key note, not just the nearest semitone. Each note moves as a whole, so
  vibrato, slides, scoops and the natural life of held notes are kept, and notes
  already in tune are left alone.
- 🎼 **Follows key changes.** If the song modulates (say, A major to C major for the
  last chorus), the voice is tuned to the new key from that point on.
- 🎸 **Tuning reference fix.** If your guitar was a few cents off A=440, the
  whole song is retuned, and your voice is tuned *to your guitar* rather than
  to an abstract grid.
- 🧹 **Cleanup.** Hiss, hum and room noise are reduced, using a noise profile
  learnt from the quiet parts of your take.
- 🎚️ **Automatic tone and dynamics.** The EQ measures your recording and moves
  it part of the way toward a balanced studio tone (cut-biased and capped).
  It also de-esses the voice and applies gentle compression.
- 🌫️ **Space and mastering.** A small natural room, a voice/guitar balance,
  and a master at streaming loudness (-14 LUFS, -1 dBTP).
- 🎧 **Honest A/B.** The original is exported at the *same loudness* as the
  result, so you hear the real difference and not just "louder".

It works with **one take** (voice and guitar recorded together: the AI splits
them so the voice can be tuned) or with **two tracks** (separate vocal and
guitar files).

## Why another tool? (market check, Oct 2026)

| Tool | What it does well | Why it doesn't fit this use case |
|---|---|---|
| **Suno / Udio** | Generate or remix songs | Re-generates your music instead of keeping *your* performance |
| **iZotope Nectar 4 + Ozone 12** | Pro-grade vocal assistant and mastering | Plugins: needs a DAW, separate tracks, money and mixing knowledge |
| **Antares Auto-Tune / Melodyne** | Best-in-class pitch correction | Voice only, needs a DAW; Melodyne is manual note editing |
| **BandLab** (free) | AutoPitch, free mastering, mobile | A DAW workflow: you build the chain yourself; mastering is basic |
| **LANDR / eMastered** | Automatic mastering | Mastering only: no pitch, no per-source cleanup |
| **Moises** | Stem separation, practice tools | Separation and practice, not "polish my take" |

Every *piece* exists, but nothing takes a raw voice+guitar take and returns a
polished, still-natural version with no DAW and no mixing knowledge. That gap
is SongPolish. **If you're happy working in a DAW, BandLab (free) or Nectar +
Ozone (paid) are excellent alternatives.**

## Install

### Installer (no Python needed)

Download the file for your system from the repository's **Releases** page:

- **Windows**: `SongPolish-<version>-windows-setup.exe`, run it (no administrator rights needed).
- **macOS** (Apple Silicon): `SongPolish-<version>-macos-arm64.dmg`, drag SongPolish to Applications.
- **Linux**: `SongPolish-<version>-x86_64.AppImage`, make it executable and run it.

Everything is included, including the separation model, so it works offline. The
installers are not code-signed, so the first launch shows a warning; see
[packaging/README.md](packaging/README.md#unsigned-builds-first-launch) for the click-through.

### From source

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+. Works on Linux,
macOS and Windows.

```bash
git clone <this repo> songpolish && cd songpolish
uv sync --extra separation   # recommended: enables voice/guitar splitting (PyTorch, ~1 GB)
# or: uv sync                # lighter, two-track mode + whole-take cleanup only
```

The first single-take run downloads the separation model (~80 MB, cached).

## Use

### Desktop app

```bash
uv run songpolish-gui
```

1. Pick **Takes with voice and guitar together** and add one or more recordings
   (or drop them on the list), or **Separate tracks** and choose both files.
2. Choose where to **Save** the edited songs (default: `~/Downloads/polished`).
3. Pick a **Style**: *Natural* (barely there), *Balanced* (default) or
   *Polished* (tighter, more produced). Adjust sliders only if you want to.
   The combination the first musician preferred is **Wrong notes: Follow the
   phrase** + **Stick to the key**.
4. Optional: **▸ Advanced thresholds** opens the engine's tuned limits (one tab
   per stage: sung notes, wrong notes, key changes, ghost voice, EQ, tuning).
   Hover over each one for what it does; changed values are shown in bold and listed
   in the report. **Save setup… / Load setup…** keep a setup in a JSON file, and the
   app remembers the last one you used.
5. Press **Polish**. Songs are processed one after another (**Stop after this song**
   ends the batch). Each finished song appears under **Edited songs**: select it and
   compare **Original** and **Polished** (switching keeps the playback position).
   With *one version per wrong-note strategy*, each version is listed separately.

### Command line

```bash
uv run songpolish my_take.m4a                                # one take, balanced preset
uv run songpolish my_take.wav -p natural --reverb 0.1         # preset + override
uv run songpolish --vocal vox.wav --guitar gtr.wav -o out/    # separate tracks
uv run songpolish my_take.wav --pitch-mode key --pitch 0.9    # tighter, in-key correction
uv run songpolish my_take.wav --compare                       # 3 versions, one per wrong-note strategy
uv run songpolish --thresholds                                # list the expert thresholds
uv run songpolish my_take.wav --set tolerance_cents=15 --set key_change_cost=6
uv run songpolish --help
```

### Output files

| File | What it is |
|---|---|
| `<name>_polished.wav` | The final master (24-bit WAV, 44.1 kHz, stereo) |
| `<name>_original_matched.wav` | Your input at the same loudness, for A/B |
| `<name>_vocal_stem.wav`, `<name>_guitar_stem.wav` | The sources: processed in two-track mode; for a single take, the AI split (pitch-corrected voice). Split stems carry separation artefacts; the polished song does not use them except on corrected notes |
| `<name>_report.json` | What was detected and changed (tuning, key and key changes, notes, loudness) |

## Settings

| Setting | Range | Default | Meaning |
|---|---|---|---|
| Pitch correction | 0–1 | 0.8 | Fraction of each note's error that is removed (one constant shift per note: vibrato and the natural movement of held notes are kept) |
| Pitch mode | harmony / chromatic / key / off | harmony | `harmony` follows the guitar's chords and key (keeps accurately sung out-of-key notes); `chromatic` = nearest semitone; `key` = only in-key notes |
| Wrong notes | nearest / phrase / repetition | nearest | What to do with notes that clash with the chord: stay within a semitone; follow the phrase and the guitar (up to a minor third); or match how the other verses were sung. `--compare` renders all three |
| Noise cleanup | 0–1 | 0.5 | Never a full gate, which avoids "underwater" artefacts |
| Tone | 0–1 | 0.6 | How far the auto-EQ moves toward the reference tone |
| Evenness (compression) | 0–1 | 0.5 | Also drives the de-esser and master glue |
| Space (reverb) | 0–1 | 0.3 | Small room; voice gets more than guitar |
| Voice ↔ guitar | ±6 dB | 0 | Voice level relative to the guitar (on a single take, raising it adds some separated voice, with its artefacts) |
| Loudness | LUFS | -14 | -16 quiet/dynamic, -11 loud |

**Advanced thresholds** (desktop app: ▸ Advanced thresholds; CLI: `--set name=value`,
listed with `--thresholds`). These are the limits the engine was tuned with, such as how
close a note must be to count as in tune, how faint a "note" must be to count as guitar
bleed, how reluctant the key detection is to see a modulation, how much ghost voice is
taken from the guitar, and when the EQ may boost. Defaults were set on real recordings;
change one at a time to fix something specific you hear. Values outside a safe range are
refused. They are defined, with explanations, in `src/songpolish/thresholds.py`.

## Limitations (MVP)

- **Separation quality limits pitch correction on a single take.** The polished song
  is your recording plus only the correction, so separation artefacts can only appear
  on corrected notes. There, some guitar bleed gets pitch-shifted with the voice, and
  not all of the voice left in the guitar part is moved back. If you can, record voice and
  guitar on separate mics or tracks.
- Two-track mode assumes the files are **already time-aligned** (exported from
  the same start point).
- Pitch correction is designed for **solo melodic singing**, not harmonies or rap.
- Processing is offline (a 4-minute song takes about 4 minutes on a modern CPU, with separation).
- The automatic EQ targets are hand-tuned heuristics: they need ears and real recordings.

## Roadmap

- [ ] Pick the best wrong-note strategy from listening tests (or combine them)
- [ ] "Musician in the loop": send the extracted notes + chords (symbolic, not audio) to an
      LLM to propose the intended melody for clashing phrases, then apply it with the same engine
- [ ] Punch-in: re-sing one phrase and splice it into the take
- [ ] Prompt-driven settings ("warmer, less reverb, keep it raw") via an LLM → `Settings`
- [ ] Guitar polish beyond EQ and dynamics: string-noise and fret-squeak reduction
- [ ] Timing tightening (optional, off by default)
- [ ] Packaged installers (PyInstaller / Briefcase) for Windows, macOS and Linux
- [ ] Mobile: run the same engine behind a small API, or port the hot paths

## How it works

[notebooks/](notebooks/README.md) explains every step in practice, with sounds you can play and plots, for
readers who know programming and data science but nothing about audio. Run them with
`uv sync --extra separation --group notebooks && uv run jupyter lab notebooks/`.

## Development

See [CLAUDE.md](CLAUDE.md) for architecture and conventions.

```bash
uv sync --extra separation
uv run pytest          # synthetic-signal tests with known ground truth
uv run ruff check . && uv run ruff format .
```

## License

SongPolish is free software under the [GNU General Public License v3.0](LICENSE)
(GPL-3.0-only). It uses [pedalboard](https://github.com/spotify/pedalboard) (GPLv3, with the
Rubber Band pitch shifter), PySide6/Qt (LGPLv3), PyTorch and Demucs (BSD/MIT) and
other permissively licensed libraries; the installers bundle them under their own licenses.
