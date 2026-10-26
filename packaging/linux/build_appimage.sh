#!/usr/bin/env bash
# Wrap the PyInstaller folder (dist/SongPolish) into dist/SongPolish-<version>-x86_64.AppImage.
# Downloads appimagetool on first use; meant for CI (see .github/workflows/release.yml).
set -euo pipefail
cd "$(dirname "$0")/../.."
VERSION="${1:?usage: build_appimage.sh <version>}"
APPDIR=build/AppDir
TOOL=build/appimagetool-x86_64.AppImage

rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/lib" "$APPDIR/usr/share/applications" \
         "$APPDIR/usr/share/icons/hicolor/512x512/apps"
cp -a dist/SongPolish "$APPDIR/usr/lib/songpolish"
cp packaging/linux/songpolish.desktop "$APPDIR/songpolish.desktop"
cp packaging/linux/songpolish.desktop "$APPDIR/usr/share/applications/"
cp packaging/linux/songpolish.png "$APPDIR/songpolish.png"
cp packaging/linux/songpolish.png "$APPDIR/usr/share/icons/hicolor/512x512/apps/"
cat > "$APPDIR/AppRun" <<'RUN'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/lib/songpolish/SongPolish" "$@"
RUN
chmod +x "$APPDIR/AppRun"

if [ ! -x "$TOOL" ]; then
  curl -fsSL -o "$TOOL" \
    https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
  chmod +x "$TOOL"
fi
# --appimage-extract-and-run: works where FUSE is unavailable (CI runners).
ARCH=x86_64 "$TOOL" --appimage-extract-and-run --comp zstd "$APPDIR" \
  "dist/SongPolish-${VERSION}-x86_64.AppImage"
