"""User-facing settings and presets.

Every knob the user can touch lives here. Amounts are normalised to 0..1 so the
UI can map them straight to sliders; the DSP modules translate them into
concrete parameters. Keep the number of knobs small: the product promise is
"good result with minimal intervention".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import StrEnum

from songpolish.melody import TargetStrategy


class PitchMode(StrEnum):
    OFF = "off"
    # Tune each note to the chord the guitar is playing: the nearest semitone unless a
    # nearby chord tone fits clearly better. Needs an accompaniment (default).
    HARMONY = "harmony"
    # Snap each sung note toward the nearest semitone, ignoring the guitar.
    CHROMATIC = "chromatic"
    # Snap toward notes of the detected key. Tighter, but can "fix" intentional
    # out-of-key notes, so it is never the default.
    KEY = "key"


@dataclass(frozen=True)
class Settings:
    # 0 = untouched, 1 = fully on target. Vibrato and slides are always kept.
    pitch_correction: float = 0.8
    pitch_mode: PitchMode = PitchMode.HARMONY
    # How notes that clash with the chord are fixed: nearest / phrase / repetition.
    note_targeting: TargetStrategy = TargetStrategy.NEAREST
    # Shift the whole take to A=440 Hz if the guitar was tuned off-reference.
    fix_reference_tuning: bool = True
    noise_reduction: float = 0.5
    # How far the automatic EQ may move toward the reference tone (0..1).
    tone_match: float = 0.6
    compression: float = 0.5
    reverb: float = 0.3
    # Vocal level relative to the guitar, in dB (positive = voice louder). On a separated
    # single take, raising it adds separated voice on top of the take, so separation
    # artefacts come with it: 0 keeps the balance as recorded.
    vocal_balance_db: float = 0.0
    # Integrated loudness of the final master. -14 LUFS suits streaming.
    target_lufs: float = -14.0
    true_peak_db: float = -1.0
    # Separate a single mixed take into voice + guitar (needs the `separation` extra).
    separate: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pitch_mode"] = self.pitch_mode.value
        d["note_targeting"] = self.note_targeting.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Settings:
        data = dict(data)
        if "pitch_mode" in data:
            data["pitch_mode"] = PitchMode(data["pitch_mode"])
        if "note_targeting" in data:
            data["note_targeting"] = TargetStrategy(data["note_targeting"])
        return cls(**data)


PRESETS: dict[str, Settings] = {
    # Barely there: cleanup and loudness, the performance stays as recorded.
    "natural": Settings(
        pitch_correction=0.55,
        noise_reduction=0.35,
        tone_match=0.4,
        compression=0.3,
        reverb=0.2,
    ),
    "balanced": Settings(),
    # Radio-ready demo: tighter pitch, more compression and space.
    "polished": Settings(
        pitch_correction=0.95,
        noise_reduction=0.65,
        tone_match=0.8,
        compression=0.7,
        reverb=0.4,
        target_lufs=-12.0,
    ),
}

DEFAULT_PRESET = "balanced"


def preset(name: str, **overrides) -> Settings:
    try:
        base = PRESETS[name]
    except KeyError:
        raise ValueError(f"Unknown preset {name!r}. Choose from: {', '.join(PRESETS)}") from None
    return replace(base, **overrides)
