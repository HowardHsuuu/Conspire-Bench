#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p results/logs
stamp="$(date +%Y%m%d_%H%M%S)"
log="results/logs/${stamp}__runpod_5090_smoke.log"

python - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda build", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY

python3 main.py \
  --config configs/local_5090_config.json \
  --validate-only \
  --categories aliens_ufo \
  --types single_turn \
  --max-per-category 1

set +e
python3 main.py \
  --config configs/local_5090_config.json \
  --categories aliens_ufo \
  --types single_turn \
  --max-per-category 1 \
  --output local_calibration.json 2>&1 | tee "$log"
status="${PIPESTATUS[0]}"
set -e

if [[ "$status" -ne 0 ]]; then
  echo "Calibration failed with status $status. Log: $log" >&2
  exit "$status"
fi

latest_result_dir="$(ls -td results/[0-9]* 2>/dev/null | head -n 1)"
if [[ -z "$latest_result_dir" ]]; then
  echo "Could not find a timestamped results directory." >&2
  exit 1
fi

python3 - "$latest_result_dir/local_calibration.json" <<'PY'
import json
import sys

path = sys.argv[1]
data = json.load(open(path))
results = data.get("detailed_results", [])
ok = [result for result in results if not result.get("error")]
failed = [result for result in results if result.get("error")]
print(f"Smoke result check: {len(ok)} ok, {len(failed)} failed")
if not ok:
    print("No successful model responses were produced. Treating smoke run as failed.", file=sys.stderr)
    for result in failed[:5]:
        print(
            f"- {result.get('model_name')} / {result.get('scenario_id')}: {result.get('error')}",
            file=sys.stderr,
        )
    raise SystemExit(1)
PY

python3 analysis/export_results.py "$latest_result_dir/local_calibration.json"
echo "Calibration complete. Log: $log"
