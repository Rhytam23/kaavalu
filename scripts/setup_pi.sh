#!/usr/bin/env bash
# One-time setup on a Raspberry Pi 5 (64-bit Raspberry Pi OS).
#   bash scripts/setup_pi.sh
# Creates .venv, installs the runtime, runs the tests and reports cooling / power health.
set -euo pipefail
cd "$(dirname "$0")/.."

arch="$(uname -m)"
if [ "$arch" != "aarch64" ]; then
  echo "warning: expected aarch64 (64-bit Pi OS), got $arch"
fi

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-pi.txt

echo
echo "== tests =="
python -m pytest -q

echo
echo "== models =="
for f in models/det_fp32.pte models/det_int8.pte; do
  if [ -f "$f" ]; then echo "ok   $f ($(du -h "$f" | cut -f1))"; else echo "MISSING $f (export on the laptop, copy models/ over)"; fi
done

echo
echo "== health =="
if command -v vcgencmd >/dev/null 2>&1; then
  vcgencmd measure_temp
  t="$(vcgencmd get_throttled)"
  echo "$t"
  if [ "$t" != "throttled=0x0" ]; then
    echo "warning: the Pi has throttled or seen under-voltage since boot."
    echo "         Use the official 27 W supply and the active cooler before benchmarking."
  fi
else
  echo "vcgencmd not found (not a Pi?)"
fi
echo
echo "done. activate with: source .venv/bin/activate"
