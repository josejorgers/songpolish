"""Command-line interface: ``songpolish take.wav`` or ``songpolish --vocal v.wav --guitar g.wav``."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

from songpolish.melody import TargetStrategy
from songpolish.pipeline import Inputs, process
from songpolish.settings import DEFAULT_PRESET, PRESETS, PitchMode, preset


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="songpolish",
        description="Automatically and gently polish a voice + guitar recording.",
    )
    p.add_argument(
        "take", nargs="?", type=Path, help="one recording with voice and guitar together"
    )
    p.add_argument("--vocal", type=Path, help="separate vocal track (use with --guitar)")
    p.add_argument("--guitar", type=Path, help="separate guitar track (use with --vocal)")
    p.add_argument("-o", "--out", type=Path, help="output folder (default: next to the input)")
    p.add_argument("-p", "--preset", choices=list(PRESETS), default=DEFAULT_PRESET)
    p.add_argument("--pitch", type=float, metavar="0-1", help="pitch correction amount")
    p.add_argument("--pitch-mode", choices=[m.value for m in PitchMode])
    p.add_argument(
        "--targeting",
        choices=[t.value for t in TargetStrategy],
        help="how notes that clash with the chord are fixed",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="render one version per --targeting strategy, to compare by ear",
    )
    p.add_argument("--noise", type=float, metavar="0-1", help="noise reduction amount")
    p.add_argument("--tone", type=float, metavar="0-1", help="automatic EQ amount")
    p.add_argument("--compression", type=float, metavar="0-1")
    p.add_argument("--reverb", type=float, metavar="0-1")
    p.add_argument("--vocal-balance", type=float, metavar="dB", help="voice vs guitar level")
    p.add_argument("--lufs", type=float, help="target loudness (default from preset, e.g. -14)")
    p.add_argument("--keep-tuning", action="store_true", help="do not retune the take to A=440")
    p.add_argument("--no-separation", action="store_true", help="never split a single take")
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = {
        "pitch_correction": args.pitch,
        "pitch_mode": PitchMode(args.pitch_mode) if args.pitch_mode else None,
        "note_targeting": TargetStrategy(args.targeting) if args.targeting else None,
        "noise_reduction": args.noise,
        "tone_match": args.tone,
        "compression": args.compression,
        "reverb": args.reverb,
        "vocal_balance_db": args.vocal_balance,
        "target_lufs": args.lufs,
        "fix_reference_tuning": False if args.keep_tuning else None,
        "separate": False if args.no_separation else None,
    }
    valid = {f.name for f in fields(PRESETS[args.preset])}
    settings = preset(
        args.preset, **{k: v for k, v in overrides.items() if v is not None and k in valid}
    )

    inputs = Inputs(mix=args.take, vocal=args.vocal, guitar=args.guitar)
    try:
        inputs.validate()
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out_dir = args.out or Path(inputs.mix or inputs.vocal).parent / "polished"

    def progress(stage: str, fraction: float) -> None:
        if not args.json:
            print(f"\r[{fraction * 100:5.1f}%] {stage:<40}", end="", file=sys.stderr, flush=True)

    strategies = list(TargetStrategy) if args.compare else None
    result = process(inputs, settings, out_dir, progress, strategies=strategies)
    if args.json:
        print(json.dumps({"output": str(result.output), **result.report}, indent=2))
        return 0
    r = result.report
    print(file=sys.stderr)
    print(f"Polished:   {result.output}")
    print(f"A/B source: {result.original_matched}")
    print(
        f"Tuning:     {r['reference_tuning_cents']:+.1f} cents vs A440"
        + (" -> retuned" if r["retuned_to_a440"] else "")
    )
    if "pitch_summary" in r:
        print(f"Pitch:      {r['pitch_summary']}")
    for name, path in result.variants.items():
        print(f"Version '{name}': {path}")
        print(f"            {r['strategies'][name]['pitch_summary']}")
    if "key" in r:
        print(f"Key:        {r['key']}")
    if "key_changes" in r:
        print(f"Key changes: {', '.join(r['key_changes'])}")
    print(
        f"Loudness:   {r['input_lufs']} -> {r['output_lufs']} LUFS (peak {r['output_peak_db']} dBFS)"
    )
    for note in r["notes"]:
        print(f"Note:       {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
