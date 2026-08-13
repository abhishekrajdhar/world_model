#!/bin/bash
# Wait for the cross-arch SMOKE kernel, verify it actually passed, then launch the full run.
# Guarded on the "SMOKE OK" marker: a kernel can exit 0 while the science is wrong, so we
# check the log rather than trusting COMPLETE.
set -u
cd /Users/abhishekrajdhardubey/Documents/world_model
SLUG=abhishekrajdhardubey/rih-cross-arch

while true; do
  S=$(kaggle kernels status "$SLUG" 2>&1 | tail -1)
  case "$S" in *COMPLETE*|*ERROR*|*CANCEL*) break;; esac
  sleep 40
done
echo "smoke finished: $S"

case "$S" in
  *ERROR*|*CANCEL*)
    echo "SMOKE FAILED - not launching the full run"; exit 1;;
esac

rm -rf /tmp/casmoke && mkdir -p /tmp/casmoke
kaggle kernels output "$SLUG" -p /tmp/casmoke >/dev/null 2>&1
./.venv/bin/python - <<'PY' > /tmp/casmoke/summary.txt
import json, glob
txt = []
for f in glob.glob('/tmp/casmoke/*.log'):
    for e in json.load(open(f)):
        t = e.get('data', '').rstrip()
        if t and 'eta 0:00' not in t and '━' not in t:
            txt.append(t)
out = "\n".join(txt)
print(out)   # full log: the marker can precede trailing warnings
PY
cat /tmp/casmoke/summary.txt

if grep -q "SMOKE OK" /tmp/casmoke/summary.txt; then
  echo "=== SMOKE PASSED -> launching full cross-arch run ==="
  ./.venv/bin/python scripts/kaggle_ood.py pusharch 2>&1 | tail -3
else
  echo "=== SMOKE DID NOT PASS (no 'SMOKE OK' marker) - full run NOT launched ==="
  exit 1
fi
