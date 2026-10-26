"""End-to-end processing: the single entry point used by the CLI and the GUI.

Stages: load -> (separate) -> reference tuning -> pitch correction -> tone
chains -> mix -> master -> export. Each stage reports progress through a
callback so UIs can stay responsive; the engine itself has no UI dependencies.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from songpolish import analysis, mastering, pitch, separation, thresholds, tone
from songpolish.audio_io import WORK_SR, load, match_length, save, to_mono, to_stereo
from songpolish.melody import TargetStrategy
from songpolish.settings import PitchMode, Settings
from songpolish.tone import Source

ProgressFn = Callable[[str, float], None]

# Re-tune the whole take to A=440 only when it is audibly off.
TUNING_FIX_THRESHOLD_CENTS = 5.0
# A separated vocal stem quieter than this (LU below the take) means "no voice found".
MIN_VOCAL_STEM_LU = 20.0


@dataclass
class Inputs:
    """Either one take with voice + guitar together, or two aligned tracks."""

    mix: Path | None = None
    vocal: Path | None = None
    guitar: Path | None = None

    def validate(self) -> None:
        if self.mix and (self.vocal or self.guitar):
            raise ValueError(
                "Give either one mixed take or separate vocal/guitar tracks, not both."
            )
        if not self.mix and not (self.vocal and self.guitar):
            raise ValueError("Separate mode needs both a vocal and a guitar track.")

    @property
    def name(self) -> str:
        return Path(self.mix or self.vocal).stem


@dataclass
class Result:
    output: Path
    original_matched: Path  # input at the same loudness, for honest A/B listening
    stems: dict[str, Path] = field(default_factory=dict)
    report: dict = field(default_factory=dict)
    variants: dict[str, Path] = field(default_factory=dict)  # strategy -> master, if comparing


def _noop(stage: str, fraction: float) -> None:
    pass


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def with_correction(
    take: np.ndarray, vocal: np.ndarray, corrected: np.ndarray, vocal_balance_db: float = 0.0
) -> np.ndarray:
    """``take`` + (``corrected`` - ``vocal``), optionally with extra voice on top."""
    change = corrected - vocal
    if vocal_balance_db:
        change = change + (10 ** (vocal_balance_db / 20) - 1) * corrected
    change = to_mono(change) if take.shape[0] == 1 else to_stereo(change)
    take, change = match_length(take, change)
    return take + change


def _corrected_take(
    take: np.ndarray,
    vocal: np.ndarray,
    corrected: np.ndarray,
    settings: Settings,
    global_shift: float,
    sr: int,
) -> np.ndarray:
    """The take itself plus only the change the pitch correction made.

    Separation artefacts (a watery voice, a guitar that fades while the voice sings)
    would all be heard if the song were rebuilt from the stems. Here they only matter
    inside corrected notes; everywhere else the result is the recording.
    """
    mixed = with_correction(take, vocal, corrected, settings.vocal_balance_db)
    if abs(global_shift) >= pitch.MIN_SHIFT:
        mixed = pitch.shift_pitch(mixed, global_shift, sr)
    mixed = tone.process_source(mixed, Source.MIX, settings, sr)
    return mastering.add_space(mixed, 0.5 * settings.reverb, vocal=False, sr=sr)


def _master(mixed: np.ndarray, settings: Settings, sr: int) -> np.ndarray:
    return mastering.master(
        mixed, settings.target_lufs, settings.true_peak_db, glue=settings.compression, sr=sr
    )


def process(
    inputs: Inputs,
    settings: Settings,
    out_dir: Path,
    progress: ProgressFn = _noop,
    sr: int = WORK_SR,
    strategies: list[TargetStrategy] | None = None,
) -> Result:
    """Polish one recording. With ``strategies``, render one version per wrong-note
    strategy (sharing separation, analysis and the guitar chain) for side-by-side
    comparison: ``<name>_polished_<strategy>.wav``. ``settings.thresholds`` override
    engine thresholds for this job only."""
    inputs.validate()
    with thresholds.applied(settings.thresholds):
        return _process(inputs, settings, Path(out_dir), progress, sr, strategies)


def _process(
    inputs: Inputs,
    settings: Settings,
    out_dir: Path,
    progress: ProgressFn,
    sr: int,
    strategies: list[TargetStrategy] | None,
) -> Result:
    t0 = time.monotonic()
    report: dict = {"settings": settings.to_dict(), "notes": []}
    if custom := thresholds.changed(settings.thresholds):
        report["notes"].append("Custom thresholds: " + thresholds.describe(custom))

    progress("Loading audio", 0.0)
    if inputs.mix:
        original = load(inputs.mix, sr)
        vocal = guitar = None
    else:
        vocal, guitar = match_length(load(inputs.vocal, sr), load(inputs.guitar, sr))
        original = None
    reference = original if original is not None else vocal + guitar
    report["input_lufs"] = analysis.loudness_lufs(reference, sr)

    if original is not None and settings.separate:
        if separation.is_available():
            progress("Separating voice and guitar", 0.05)
            stems = separation.separate(
                original,
                sr,
                progress=lambda f: progress("Separating voice and guitar", 0.05 + 0.3 * f),
            )
            vocal_lufs = analysis.loudness_lufs(stems.vocals, sr)
            if vocal_lufs >= report["input_lufs"] - MIN_VOCAL_STEM_LU:
                vocal, guitar = stems.vocals, stems.accompaniment
                report["separated"] = True
            else:
                # Boosting a near-empty stem to mix level would only amplify artefacts.
                report["notes"].append(
                    "No clear voice found in the take; processed it as a whole "
                    "(no vocal pitch correction)."
                )
        else:
            report["notes"].append(
                "Voice/guitar separation is not installed; processed the take as a whole "
                "(no vocal pitch correction). Install with: uv sync --extra separation"
            )
    report.setdefault("separated", False)

    progress("Analysing tuning", 0.38)
    tuning_source = guitar if guitar is not None else original
    offset = analysis.tuning_offset_cents(tuning_source, sr)
    report["reference_tuning_cents"] = round(offset, 1)
    retune = settings.fix_reference_tuning and abs(offset) >= TUNING_FIX_THRESHOLD_CENTS
    global_shift = -offset / 100.0 if retune else 0.0
    report["retuned_to_a440"] = retune

    # name -> (final master, stems). One entry normally; one per strategy when comparing.
    renders: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    if guitar is None:
        # Whole-take path: one chain, no per-note correction possible.
        mixed = original
        if retune:
            progress("Retuning to A=440", 0.45)
            mixed = pitch.shift_pitch(mixed, global_shift, sr)
        progress("Polishing tone", 0.6)
        mixed = tone.process_source(mixed, Source.MIX, settings, sr)
        mixed = mastering.add_space(mixed, 0.5 * settings.reverb, vocal=False, sr=sr)
        renders[""] = (_master(mixed, settings, sr), {})
    else:
        key = harmony = None
        if settings.pitch_mode in (PitchMode.HARMONY, PitchMode.KEY):
            harmony = analysis.harmony_map(guitar, sr, tuning_cents=offset)
            report["key"] = harmony.key.name
            if len(harmony.keys) > 1:
                report["key_changes"] = [
                    f"{_clock(span.start / sr)} {span.key.name}" for span in harmony.keys
                ]
            if settings.pitch_mode is PitchMode.KEY:
                key = harmony.key
        strength = 0.0 if settings.pitch_mode is PitchMode.OFF else settings.pitch_correction

        progress("Finding sung notes", 0.42)
        notes = pitch.detect_notes(vocal, offset, sr) if strength > 0 else []
        names = [s.value for s in strategies] if strategies else [""]
        plans: dict[str, tuple[list[pitch.NoteShift], pitch.PitchReport]] = {}
        for name in names:
            strategy = TargetStrategy(name) if name else settings.note_targeting
            plans[name] = (
                pitch.plan_corrections(vocal, strength, offset, key, harmony, sr, strategy, notes)
                if strength > 0
                else ([], pitch.PitchReport())
            )
        # Any note some version moves: its voice must not stay behind in the guitar stem.
        all_shifts = [sh for shifts, _ in plans.values() for sh in shifts]
        vocal, guitar, reclaimed = pitch.reclaim_leakage(
            vocal, guitar, notes, all_shifts, offset, sr
        )
        report["leakage_notes_reclaimed"] = reclaimed
        separated = report["separated"]
        if not separated:
            if retune:
                guitar = pitch.shift_pitch(guitar, global_shift, sr)
            progress("Polishing guitar", 0.5)
            guitar = tone.process_source(guitar, Source.GUITAR, settings, sr)

        for k, name in enumerate(names):
            frac = 0.55 + 0.35 * k / len(names)
            progress(f"Correcting vocal pitch{f' ({name})' if name else ''}", frac)
            shifts, pitch_report = plans[name]
            if strength > 0:
                entry = {"pitch": asdict(pitch_report), "pitch_summary": pitch_report.summary()}
                if name:
                    report.setdefault("strategies", {})[name] = entry
                else:
                    report.update(entry)
            progress("Polishing and mixing", frac + 0.15 / len(names))
            if separated:
                corrected = pitch.apply_corrections(vocal, shifts, 0.0, sr) if shifts else vocal
                mixed = _corrected_take(original, vocal, corrected, settings, global_shift, sr)
                renders[name] = (_master(mixed, settings, sr), {"vocal": corrected})
                continue
            v = vocal
            if shifts or abs(global_shift) >= pitch.MIN_SHIFT:
                v = pitch.apply_corrections(vocal, shifts, global_shift, sr)
            v = tone.process_source(v, Source.VOCAL, settings, sr)
            mixed = mastering.mix(v, guitar, settings.vocal_balance_db, settings.reverb, sr)
            renders[name] = (_master(mixed, settings, sr), {"vocal": v, "guitar": guitar})
        if separated:
            renders[next(iter(renders))][1]["guitar"] = guitar

    final = next(iter(renders.values()))[0]
    report["output_lufs"] = round(analysis.loudness_lufs(final, sr), 1)
    report["output_peak_db"] = round(float(20 * np.log10(np.max(np.abs(final)) + 1e-9)), 1)

    progress("Saving", 0.95)
    # Louder always sounds "better", so the A/B reference gets the same loudness.
    reference = mastering.loudness_match(
        reference, report["output_lufs"], settings.true_peak_db, sr
    )

    base = inputs.name
    result = Result(
        output=out_dir / f"{base}_polished.wav",
        original_matched=save(out_dir / f"{base}_original_matched.wav", reference, sr),
    )
    for name, (audio, stems_out) in renders.items():
        suffix = f"_{name}" if name else ""
        path = save(out_dir / f"{base}_polished{suffix}.wav", audio, sr)
        if name:
            result.variants[name] = path
        for stem, data in stems_out.items():
            if stem == "guitar" and stem in result.stems:
                continue  # identical across strategies
            stem_suffix = suffix if stem == "vocal" else ""
            result.stems[stem + stem_suffix] = save(
                out_dir / f"{base}_{stem}_stem{stem_suffix}.wav", data, sr
            )
    result.output = next(iter(result.variants.values()), result.output)

    report["input_lufs"] = round(report["input_lufs"], 1)
    report["seconds"] = round(time.monotonic() - t0, 1)
    result.report = report
    (out_dir / f"{base}_report.json").write_text(json.dumps(report, indent=2))
    progress("Done", 1.0)
    return result
