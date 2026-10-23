#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
command -v uv >/dev/null || { echo '请先安装 uv: https://docs.astral.sh/uv/'; exit 1; }
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python "${R2S_CORE_PYTHON:-3.12}" .venv
fi
uv pip install --python .venv/bin/python -r environments/core.lock.txt
uv pip install --python .venv/bin/python -e '.[dev]'
echo '核心环境已安装；按 .env.example 配置模型环境，再运行 scripts/run.sh。'
