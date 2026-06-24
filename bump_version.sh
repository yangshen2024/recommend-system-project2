#!/bin/bash
# Usage:
#   ./bump_version.sh          → bump patch (1.0.0 → 1.0.1)
#   ./bump_version.sh minor    → bump minor (1.0.0 → 1.1.0)
#   ./bump_version.sh major    → bump major (1.0.0 → 2.0.0)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERSION_FILE="$SCRIPT_DIR/VERSION"

if [ ! -f "$VERSION_FILE" ]; then
  echo "1.0.0" > "$VERSION_FILE"
  echo "Created VERSION file with 1.0.0"
  exit 0
fi

OLD_VERSION=$(cat "$VERSION_FILE" | tr -d '[:space:]')
IFS='.' read -r MAJOR MINOR PATCH <<< "$OLD_VERSION"

case "${1:-patch}" in
  major)
    MAJOR=$((MAJOR + 1))
    MINOR=0
    PATCH=0
    ;;
  minor)
    MINOR=$((MINOR + 1))
    PATCH=0
    ;;
  patch|*)
    PATCH=$((PATCH + 1))
    ;;
esac

NEW_VERSION="${MAJOR}.${MINOR}.${PATCH}"
echo "$NEW_VERSION" > "$VERSION_FILE"
echo "✅ Version bumped: $OLD_VERSION → $NEW_VERSION"
