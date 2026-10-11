#!/usr/bin/env bash
#
# Build a native, installable application for the current platform.
#
#   macOS   -> OpenForge.app  (Drag to /Applications, double-click to run)
#   Windows -> OpenForge.exe  (built into dist/; wrap in an installer separately)
#   Linux   -> OpenForge      (built into dist/; package as AppImage/deb separately)
#
# Usage: ./scripts/build-app.sh [--release]
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROFILE_FLAG="${1:---release}"
PROFILE="debug"
[[ "$PROFILE_FLAG" == "--release" ]] && PROFILE="release"

echo "==> Building openforge-app ($PROFILE)"
if [[ "$PROFILE" == "release" ]]; then
  cargo build --release -p openforge-app
  BIN="target/release/openforge"
else
  cargo build -p openforge-app
  BIN="target/debug/openforge"
fi

mkdir -p dist
rm -rf dist/OpenForge.app dist/OpenForge.exe dist/OpenForge

case "$(uname -s)" in
  Darwin)
    APP="dist/OpenForge.app"
    echo "==> Assembling $APP"
    mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
    cp "$BIN" "$APP/Contents/MacOS/OpenForge"
    cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>OpenForge</string>
    <key>CFBundleDisplayName</key><string>OpenForge</string>
    <key>CFBundleIdentifier</key><string>com.openforge.app</string>
    <key>CFBundleVersion</key><string>0.1.0</string>
    <key>CFBundleShortVersionString</key><string>0.1.0</string>
    <key>CFBundleExecutable</key><string>OpenForge</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>LSMinimumSystemVersion</key><string>10.15</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSPrincipalClass</key><string>NSApplication</string>
</dict>
</plist>
PLIST
    # Sign ad-hoc so Gatekeeper allows local launches.
    codesign --force --sign - "$APP" 2>/dev/null || echo "    (ad-hoc codesign skipped)"
    echo "==> Done: $APP"
    echo "    Install:  cp -R \"$APP\" /Applications/"
    echo "    Or run:   open \"$APP\""
    ;;

  MINGW*|MSYS*|CYGWIN*)
    OUT="dist/OpenForge.exe"
    cp "$BIN" "$OUT"
    echo "==> Done: $OUT"
    echo "    (Wrap in an installer such as Inno Setup or MSIX for distribution.)"
    ;;

  *)
    OUT="dist/OpenForge"
    cp "$BIN" "$OUT"
    echo "==> Done: $OUT"
    echo "    (Package as an AppImage or .deb for distribution.)"
    ;;
esac
