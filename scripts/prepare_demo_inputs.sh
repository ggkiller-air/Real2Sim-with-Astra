#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
mkdir -p data/image data/depth
if [[ ! -e data/image/tabletop.jpg ]]; then
  cp examples/rgbd_tabletop/capture/image_external_rgb.jpg data/image/tabletop.jpg
fi
if [[ ! -e data/depth/tabletop ]]; then
  cp -R examples/rgbd_tabletop/capture data/depth/tabletop
fi
printf '%s\n' '示例已准备：data/image/tabletop.jpg 和 data/depth/tabletop/（已有文件不会覆盖）'
