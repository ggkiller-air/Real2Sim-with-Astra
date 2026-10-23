#!/usr/bin/env bash
# Source acquisition only. Checkpoint access and heavy runtime installation are separate.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
mkdir -p vendor
repo_dir=vendor/sam-3d-objects
revision=f91db411c50efee93d8db7aeb323885650f6f722
if [[ ! -d "$repo_dir/.git" ]]; then
  git clone https://github.com/facebookresearch/sam-3d-objects.git "$repo_dir"
fi
if [[ -n "$(git -C "$repo_dir" status --porcelain)" ]]; then
  echo 'SAM 3D 仓库有本地修改，未切换版本。' >&2
  exit 1
fi
git -C "$repo_dir" checkout --detach "$revision"
echo '官方源码已准备。环境与权重说明见 docs/models.md。'
