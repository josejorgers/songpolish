"""Loading and saving audio.

Internal convention used across the whole engine: float32 numpy arrays shaped
(channels, samples) at ``WORK_SR``. Mono is (1, samples).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from pedalboard.io import AudioFile

WORK_SR = 44100


def load(path: str | Path, sr: int = WORK_SR) -> np.ndarray:
    """Load any common format (wav, flac, mp3, ogg, m4a/aac where the OS supports it)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with AudioFile(str(path)).resampled_to(sr) as f:
            audio = f.read(f.frames)
    except Exception:
        # pedalboard cannot decode AAC/m4a on Linux (the usual phone format); ffmpeg can.
        audio = _ffmpeg_load(path, sr)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _ffmpeg_load(path: Path, sr: int) -> np.ndarray:
    import subprocess

    import imageio_ffmpeg  # bundles an ffmpeg binary for every desktop OS

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path), "-vn",
           "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(sr), "pipe:1"]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        msg = proc.stderr.decode(errors="replace").strip() or "no audio stream"
        raise ValueError(f"Could not read {path.name}: {msg}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32).reshape(-1, 2).T
    # Decoded as stereo; collapse back to mono when both channels are identical.
    return audio[:1] if np.array_equal(audio[0], audio[1]) else audio


def save(path: str | Path, audio: np.ndarray, sr: int = WORK_SR) -> Path:
    """Save as 24-bit WAV/FLAC (by extension) or anything pedalboard can encode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.atleast_2d(audio)
    if path.suffix.lower() in {".wav", ".flac"}:
        sf.write(str(path), audio.T, sr, subtype="PCM_24")
    else:
        with AudioFile(str(path), "w", sr, audio.shape[0]) as f:
            f.write(audio)
    return path


def to_mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=0, keepdims=True)


def to_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.shape[0] == 2:
        return audio
    return np.repeat(to_mono(audio), 2, axis=0)


def match_length(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero-pad the shorter of two (channels, samples) arrays."""
    n = max(a.shape[1], b.shape[1])
    pad = lambda x: np.pad(x, ((0, 0), (0, n - x.shape[1])))  # noqa: E731
    return pad(a), pad(b)
