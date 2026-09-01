#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "$PROJECT_ROOT"

mkdir -p results/logs
stamp="$(date +%Y%m%d_%H%M%S)"
log="results/logs/${stamp}__local_v3_smoke.log"

"$PYTHON_BIN" - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda build", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY

"$PYTHON_BIN" main.py \
  --config configs/experiment_v3_local_smoke.json \
  --dataset Conspire-Bench-v3.json \
  --validate-only \
  --scenario-ids v3_weather_cloud_seeding_single_001 \
  --context-variants neutral_none

set +e
"$PYTHON_BIN" main.py \
  --config configs/experiment_v3_local_smoke.json \
  --dataset Conspire-Bench-v3.json \
  --scenario-ids v3_weather_cloud_seeding_single_001 \
  --context-variants neutral_none \
  --execution-mode phased \
  --output local_v3_smoke.json 2>&1 | tee "$log"
status="${PIPESTATUS[0]}"
set -e

if [[ "$status" -ne 0 ]]; then
  echo "Calibration failed with status $status. Log: $log" >&2
  exit "$status"
fi

result_path="$(sed -n 's/^Full results saved to: //p' "$log" | tail -n 1)"
if [[ -z "$result_path" || ! -f "$result_path" ]]; then
  echo "Could not resolve this run's result file from $log." >&2
  exit 1
fi

"$PYTHON_BIN" - "$result_path" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    data = json.load(handle)
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

"$PYTHON_BIN" analysis/export_results.py "$result_path"
echo "Calibration complete. Log: $log"
