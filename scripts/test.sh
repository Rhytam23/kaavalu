#!/usr/bin/env bash
# Run the test suite. Torch parity tests skip automatically when torch is not installed.
#   bash scripts/test.sh            all tests
#   bash scripts/test.sh -k tiling  pytest arguments pass through
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
python -m pytest -q "$@"
