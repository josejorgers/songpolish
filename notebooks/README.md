# How SongPolish works, in practice

Eight notebooks that explain every step of the app, from "what is a sound file" to the full pipeline on a
real song. They assume basic programming, data science and machine learning, and **no knowledge of
audio or music**. Each one runs the project's own code, mostly on synthetic sounds whose exact pitch is
known, so every measurement can be checked against the truth. Press ▶ on the players to listen.

| # | Notebook | You will learn |
|---|---|---|
| 1 | [Digital sound](01_digital_sound.ipynb) | samples, sample rate, frequency and pitch, MIDI numbers and cents, harmonics, FFT and spectrograms, decibels, LUFS |
| 2 | [Music theory for machines](02_music_theory_for_machines.ipynb) | pitch classes, keys, chords, chroma features, key detection by template correlation, key changes with Viterbi, tuning reference |
| 3 | [Finding the sung notes](03_finding_the_sung_notes.ipynb) | pitch tracking (autocorrelation, YIN, pYIN + HMM), note segmentation, vibrato, trimmed means, phantom notes |
| 4 | [Choosing the right note](04_choosing_the_right_note.ipynb) | out-of-tune vs wrong notes, a linear cost with chord and key, strict key mode, PHRASE (Viterbi), REPETITION (k-NN vote), repeats in a new key |
| 5 | [Moving notes](05_moving_notes.ipynb) | resampling vs phase vocoder, formants, per-note rendering with crossfades, two real bugs (chirps, "fan" flutter) |
| 6 | [Separating voice and guitar](06_separating_voice_and_guitar.ipynb) | source separation as supervised ML (Demucs), masks and SDR, leakage and ghost notes, why the song is "recording + correction" |
| 7 | [Cleaning and mastering](07_cleaning_and_mastering.ipynb) | noise floor and SNR, spectral gating, EQ filters, automatic EQ and the "old recorder" lesson, de-esser, compressor, reverb, loudness mastering |
| 8 | [The whole pipeline on a real song](08_whole_pipeline_on_a_real_song.ipynb) | every stage on your own recording, with plots and what to check by ear |

## Running them

```bash
uv sync --extra separation --group notebooks   # once: the app, Demucs, and Jupyter
uv run jupyter lab notebooks/
```

Notebooks 1–7 take seconds each. Notebook 8 needs a recording (set `SONG` in its first cell); the
voice/guitar separation takes a few minutes the first time and is cached in `notebooks/.cache/`
(git-ignored). It writes its results to `~/Downloads/polished/notebook_demo/` and embeds no audio of your
song unless you set `PLAY = True`.

`nbtools.py` holds the shared helpers: synthetic voices and chords, the audio player and plots.
