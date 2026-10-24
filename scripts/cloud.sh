#!/usr/bin/env bash
# Run Driving on the shared cloud worker with its own interpreter.
set -euo pipefail

driving_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
driving_action="${1:-preflight}"
if [[ $# -gt 0 ]]; then shift; fi

if [[ -n "${DRIVING_PYTHON:-}" ]]; then
    driving_python="$DRIVING_PYTHON"
elif [[ -x "$driving_root/.venv/bin/python" ]]; then
    driving_python="$driving_root/.venv/bin/python"
elif [[ -x /workspace/yike-top2-venv/bin/python ]]; then
    driving_python=/workspace/yike-top2-venv/bin/python
elif [[ "$driving_action" == setup ]]; then
    python3.12 -m venv "$driving_root/.venv"
    driving_python="$driving_root/.venv/bin/python"
    "$driving_python" -m pip install -r "$driving_root/requirements-cpu-lock.txt"
else
    printf 'Run bash scripts/cloud.sh setup first.\n' >&2
    exit 2
fi

if [[ "$driving_action" == setup ]]; then
    "$driving_python" - "$driving_root" <<'PY'
import importlib.metadata
import json
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
pins = {}
for line in (root / "requirements-cpu-lock.txt").read_text().splitlines():
    match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", line)
    if match:
        pins[match[1]] = match[2]
differences = {}
for name, wanted in pins.items():
    try:
        actual = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        actual = None
    if actual != wanted:
        differences[name] = {"expected": wanted, "actual": actual}
ready = sys.version_info[:2] == (3, 12) and not differences
record = {
    "kind": "Driving CPU environment on existing shared worker",
    "project_root": str(root), "python": sys.executable,
    "python_version": sys.version.split()[0], "cpu_ready": ready,
    "pinned_packages_checked": len(pins), "dependency_differences": differences,
    "DWTS_environment_modified": False,
    "new_platform_environment_created": False,
    "network_policy_modified": False,
    "data_or_GPU_experiment_claimed": False,
}
destination = root / "output/cloud/runtime.json"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
if not ready:
    raise SystemExit("CPU environment differs from lock; select/create a separate Driving venv")
PY
    exec "$driving_python" "$driving_root/scripts/driving_preflight.py" \
        --data-root "$driving_root/dataset/nuscenes" \
        --model-dir "$driving_root/models/Cosmos-Reason1-7B" \
        --output "$driving_root/output/cloud/preflight.json"
fi

if [[ "$driving_action" == --help || "$driving_action" == help ]]; then
    printf 'Usage: bash scripts/cloud.sh setup | <make-target> [VARIABLE=value ...]\n'
    printf 'Examples: preflight; demo OUTPUT=output/cloud-demo; real-validate OUTPUT=output/real-mini\n'
    exit 0
fi
exec make --no-print-directory -C "$driving_root" "PYTHON=$driving_python" \
    "$driving_action" "$@"
