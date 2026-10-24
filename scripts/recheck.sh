#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ -f .env.local ]]; then
  set -a
  source .env.local
  set +a
fi
exec .venv/bin/python -m real2sim_workbench.cli recheck "$@"
