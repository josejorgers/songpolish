# Installers

The desktop app is bundled with [PyInstaller](https://pyinstaller.org) (one folder, with
PyTorch, Demucs and its model, so separation works offline from the first run) and wrapped
per platform:

| Platform | File | Built on |
|---|---|---|
| Linux (x86-64, glibc 2.35+: Ubuntu 22.04+, Debian 12+, Fedora 36+) | `SongPolish-<v>-x86_64.AppImage` | ubuntu-22.04 |
| macOS (Apple Silicon, 12+) | `SongPolish-<v>-macos-arm64.dmg` | macos-14 |
| Windows (x64, 10/11) | `SongPolish-<v>-windows-setup.exe` (per-user, no admin) | windows-latest |

Each is ~1.2 GB unpacked (PyTorch alone is ~0.5 GB), roughly half that to download.

## Releasing

1. Bump `version` in `pyproject.toml`, run `uv lock`, commit.
2. `git tag v<version> && git push origin v<version>`.
3. `.github/workflows/release.yml` builds all three on GitHub's runners, runs the test
   suite and `SongPolish --self-test` on each installed build, and attaches the installers
   to a **draft** release. Check it, then publish it on GitHub.

The workflow can also be started by hand (Actions → Release → Run workflow): it then only
builds and tests, and the installers are kept as run artifacts for 14 days.

## Building locally (your own platform only)

```bash
uv sync --extra separation --group release
uv run python packaging/fetch_model.py          # Demucs model -> build/models/htdemucs
uv run pyinstaller --noconfirm --distpath dist --workpath build/pyinstaller packaging/songpolish.spec
QT_QPA_PLATFORM=offscreen dist/SongPolish/SongPolish --self-test [recording.wav out_dir]
packaging/linux/build_appimage.sh 0.1.0         # Linux; downloads appimagetool
```

`make_icons.py` redraws the icons (`packaging/{linux,windows,macos}/`), which are committed.

## Unsigned builds: first launch

The installers are not code-signed (that needs an Apple Developer account and a Windows
code-signing certificate). The system warns once:

- **macOS**: "SongPolish can't be opened / Apple could not verify…". Open it once from
  System Settings → Privacy & Security → *Open Anyway* (or right-click → Open on older
  macOS). Alternative: `xattr -dr com.apple.quarantine /Applications/SongPolish.app`.
- **Windows**: SmartScreen "Windows protected your PC" → *More info* → *Run anyway*.
- **Linux**: `chmod +x SongPolish-*.AppImage`, then run it. To get a menu entry, use an
  AppImage manager (e.g. Gear Lever or AppImageLauncher).

## Gotchas

- `torch/bin` must stay in the bundle (`torch_shm_manager` is checked at import); only
  `torch/test` is dropped.
- librosa needs its `.pyi` stubs (lazy_loader) and numba needs a writable cache
  (`NUMBA_CACHE_DIR`, set in `entry.py`).
- A windowed app has no stdout/stderr: output goes to `songpolish.log` in the app data
  folder (`~/.local/share/SongPolish`, `~/Library/Application Support/SongPolish`,
  `%LOCALAPPDATA%\SongPolish`).
- Bundled pedalboard (Rubber Band) is GPLv3, so the installers are GPLv3 software as a whole.
