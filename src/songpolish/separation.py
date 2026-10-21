"""Optional source separation of a single voice+guitar take (Demucs).

Imported lazily: the engine must keep working without PyTorch installed, it just
falls back to whole-mix processing (no vocal pitch correction).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from songpolish.audio_io import WORK_SR, to_stereo

MODEL_NAME = "htdemucs"


@dataclass
class Stems:
    vocals: np.ndarray  # (2, n)
    accompaniment: np.ndarray  # (2, n) everything that is not voice: the guitar


def is_available() -> bool:
    try:
        import demucs.apply  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


@lru_cache(maxsize=1)
def _model():
    from demucs.pretrained import get_model

    model = get_model(MODEL_NAME)  # downloads ~80 MB on first use
    model.eval()
    return model


def separate(
    audio: np.ndarray,
    sr: int = WORK_SR,
    progress: Callable[[float], None] | None = None,
) -> Stems:
    import torch
    from demucs.apply import apply_model

    model = _model()
    if sr != model.samplerate:
        raise ValueError(f"Demucs expects {model.samplerate} Hz, got {sr}")

    mix = torch.from_numpy(to_stereo(audio).astype(np.float32))
    # Demucs is trained on normalised input; undo it afterwards.
    ref = mix.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    mix_n = (mix - mean) / std

    length = mix.shape[-1]

    def _cb(info: dict) -> None:
        # Keys provided by demucs: model_idx_in_bag, models, segment_offset, state.
        if progress and info.get("state") == "end":
            models = max(1, info.get("models", 1))
            done = info.get("model_idx_in_bag", 0) + info.get("segment_offset", 0) / length
            progress(min(1.0, done / models))

    with torch.no_grad():
        # shifts=0: no random time-shift augmentation, so results are reproducible.
        out = apply_model(
            model, mix_n[None], shifts=0, device="cpu", split=True, overlap=0.25, callback=_cb
        )
    out = (out[0] * std + mean).numpy()

    idx = model.sources.index("vocals")
    vocals = out[idx]
    accompaniment = out.sum(axis=0) - vocals
    return Stems(vocals=vocals.astype(np.float32), accompaniment=accompaniment.astype(np.float32))
