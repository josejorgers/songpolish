# PyInstaller spec for the SongPolish desktop app (one folder; see packaging/README.md).
# Build: uv run python packaging/fetch_model.py && uv run pyinstaller packaging/songpolish.spec
import sys
from importlib.metadata import version
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, copy_metadata

ROOT = Path(SPECPATH).parent
MODEL = ROOT / "build" / "models" / "htdemucs"
if not (MODEL / "htdemucs.yaml").is_file():
    sys.exit(f"Model missing: run `uv run python packaging/fetch_model.py` first ({MODEL})")

datas = [(str(MODEL), "models/htdemucs")]
binaries = []
hiddenimports = []
for pkg in ("pedalboard", "demucs", "imageio_ffmpeg"):
    d, b, h = collect_all(pkg)
    datas, binaries, hiddenimports = datas + d, binaries + b, hiddenimports + h
datas += collect_data_files("librosa")  # lazy_loader reads its .pyi stubs at import
datas += copy_metadata("songpolish") + copy_metadata("librosa")
hiddenimports += ["songpolish.gui.app", "self_test", "sklearn.utils._typedefs"]

icon = {
    "win32": ROOT / "packaging" / "windows" / "songpolish.ico",
    "darwin": ROOT / "packaging" / "macos" / "songpolish.icns",
}.get(sys.platform)

a = Analysis(
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT / "src"), str(ROOT / "packaging")],
    datas=datas,
    binaries=binaries,
    hiddenimports=hiddenimports,
    # Not used by the app; some are pulled in by optional imports and weigh hundreds of MB.
    excludes=["matplotlib", "PIL", "sphn", "IPython", "jupyter", "notebook", "tkinter", "pytest",
              "torch.utils.tensorboard", "tensorboard", "PySide6.QtWebEngineCore",
              "PySide6.Qt3DCore", "PySide6.QtQuick3D", "PySide6.QtPdf"],
)
# torch's own test suite (~85 MB) is never used. (torch/bin is: torch_shm_manager.)
a.datas = [d for d in a.datas if not d[0].startswith("torch/test/")]
a.binaries = [b for b in a.binaries if not b[0].startswith("torch/test/")]
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SongPolish",
    console=False,
    icon=str(icon) if icon and icon.is_file() else None,
)
coll = COLLECT(exe, a.binaries, a.datas, name="SongPolish")
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="SongPolish.app",
        icon=str(icon) if icon and icon.is_file() else None,
        bundle_identifier="io.github.josejorgers.songpolish",
        version=version("songpolish"),
        info_plist={"NSHighResolutionCapable": True, "LSMinimumSystemVersion": "12.0"},
    )
