#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ -f .env.local ]]; then
  set -a
  source .env.local
  set +a
fi
if [[ ! -x .venv/bin/python ]]; then
  echo '请先运行 scripts/setup.sh 创建本项目环境。' >&2
  exit 1
fi
exec .venv/bin/python -m real2sim_workbench.cli serve "$@"
