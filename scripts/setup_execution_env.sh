#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SOURCE="$ROOT/data/external/abides-ng"
REVISION=ac7a2a7bf6e406cd2bbed887e22c5a5a790bb242
PYTHON_BIN=${PYTHON:-python3}

mkdir -p "$ROOT/data/external"
if [ ! -d "$SOURCE/.git" ]; then
    git clone https://github.com/GabrieleDiCorato/abides-ng.git "$SOURCE"
fi

git -C "$SOURCE" fetch --quiet origin "$REVISION"
git -C "$SOURCE" checkout --quiet --detach "$REVISION"
"$PYTHON_BIN" -m pip install -e "$SOURCE/abides-core" -e "$SOURCE/abides-markets" 'gymnasium>=1.0,<2'

printf 'ABIDES-NG pinned at %s\n' "$(git -C "$SOURCE" rev-parse HEAD)"
printf 'Python: %s\n' "$PYTHON_BIN"
