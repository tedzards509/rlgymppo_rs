#!/usr/bin/env bash
set -euo pipefail

if ! cargo pkgid -p annie-train >/dev/null 2>&1; then
  echo "Error: annie-train workspace member not found." >&2
  exit 1
fi

LIBTORCH_DIR="${LIBTORCH:-$HOME/.libtorch/libtorch-rocm-6.4-2_9}"

if [[ ! -d "$LIBTORCH_DIR" ]]; then
  echo "Error: No directory found at $LIBTORCH_DIR" >&2
  exit 1
fi

LIBTORCH="$LIBTORCH_DIR" \
LD_LIBRARY_PATH="$LIBTORCH_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
LIBTORCH_BYPASS_VERSION_CHECK=1 \
cargo run -p annie-train \
  --no-default-features \
  --features torch,tui \
  -- "$@"