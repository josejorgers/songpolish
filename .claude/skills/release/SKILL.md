---
name: release
description: Release a new SongPolish version with installers for Linux, macOS and Windows (version bump, tag, CI build, check the self-tests, draft release). Use when the user asks to release, publish, ship or build installers for SongPolish.
---

# Releasing SongPolish

A `v*` tag makes `.github/workflows/release.yml` build all three installers on GitHub's
runners. It runs the tests and `SongPolish --self-test` on each built app, then
attaches the installers to a **draft** release. A person publishes it; never publish it
yourself unless the user explicitly asks.

The packaging itself (spec, entry point, AppImage/dmg/Inno Setup) is described in
`packaging/README.md`.

## Steps

1. **Pick the version.** Ask the user if they did not say. Patch (`0.1.1`) for fixes,
   minor (`0.2.0`) for new features. It must match the tag without the `v`, or the
   workflow fails at "Version".

2. **Check the tree and the tests.**
   ```bash
   git status --short            # clean, on main, up to date with origin
   uv run ruff check . && uv run ruff format --check .
   uv run pytest -q
   ```
   If engine behaviour changed since the last release, the notebooks must be re-run
   (see CLAUDE.md).

3. **Build locally if packaging or dependencies changed** (new dependency, changes in
   `packaging/`, `separation.py` or the entry point). PyInstaller misses data files and
   lazy imports without an error; only the built app shows it:
   ```bash
   uv sync --extra separation --group release
   uv run python packaging/fetch_model.py
   uv run pyinstaller --noconfirm --distpath dist --workpath build/pyinstaller packaging/songpolish.spec
   QT_QPA_PLATFORM=offscreen dist/SongPolish/SongPolish --self-test
   ```
   For a real check, pass a voice + guitar recording plus an output folder (separation
   only finds a voice in real singing). Also check that every new file is tracked:
   `git status --short --ignored packaging/` (`*.spec` is ignored except
   `packaging/songpolish.spec`).

4. **Bump and commit.**
   ```bash
   # edit version = "X.Y.Z" in pyproject.toml
   uv lock
   git commit -am "Release X.Y.Z"   # end the message with the session's attribution lines
   git push origin main
   ```

5. **Tag and push the tag.** This starts the build.
   ```bash
   git tag -a vX.Y.Z -m "SongPolish X.Y.Z" && git push origin vX.Y.Z
   gh run list --repo josejorgers/songpolish --limit 1   # get the run id
   ```
   Wait with `gh run watch <id> --repo josejorgers/songpolish --exit-status` in the
   background; it takes ~15–25 min. Don't poll.

6. **Verify, don't trust the green tick.** Every platform must print
   `separation available: True …models/htdemucs` and `self-test: ok`:
   ```bash
   gh run view <id> --repo josejorgers/songpolish --log | grep -E "window: ok|separation available|self-test:"
   gh release view vX.Y.Z --repo josejorgers/songpolish --json isDraft,assets \
     -q '.isDraft, (.assets[]|.name+" "+((.size/1048576|floor)|tostring)+" MB")'
   ```
   Expect three assets of roughly 350–460 MB: `SongPolish-X.Y.Z-x86_64.AppImage`,
   `-macos-arm64.dmg`, `-windows-setup.exe`. A big jump in size usually means a new
   dependency pulled in something heavy; check the bundle (`du -sh dist/SongPolish/_internal/*`).

7. **Hand over.** Give the user the draft's link. Remind them to try one installer, then
   publish it on GitHub (Edit → Publish release). The installers are unsigned, so the
   first launch shows a macOS/Windows warning (click-through in `packaging/README.md`).
   Report honestly what was checked: the CI self-test uses a synthetic take, so pitch
   correction on real singing was only checked where you ran it yourself.

## If a build fails

- `gh run view <id> --log-failed` shows the failing step. "Tests" failing on one OS is
  a real portability bug: fix it, don't skip the test.
- Fix it on main, then move the tag, **only while no release has been published**
  (a draft may be deleted first with `gh release delete vX.Y.Z --yes`):
  ```bash
  git push origin :refs/tags/vX.Y.Z && git tag -d vX.Y.Z
  git tag -a vX.Y.Z -m "SongPolish X.Y.Z" && git push origin vX.Y.Z
  ```
  After a release has been published, never move its tag; release a new patch version.
- Past failures: the spec file was not committed (the `*.spec` ignore rule); `torch/bin`
  removed from the bundle (`torch_shm_manager` is required at import).

## Things that are not automated

- No code signing or notarisation (it would need an Apple Developer account and a
  Windows certificate as repository secrets).
- macOS builds are Apple Silicon only (PyTorch has no current Intel-macOS wheels).
- The repo is public and GPL-3.0-only (the bundled pedalboard/Rubber Band is GPLv3); the
  release must stay GPL, with the source available.
