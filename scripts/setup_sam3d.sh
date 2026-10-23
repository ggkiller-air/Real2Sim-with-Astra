#!/usr/bin/env bash
# Isolated Linux/CUDA 12.1 runtime. Does not modify the SAM3 or core environments.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
command -v uv >/dev/null
command -v conda >/dev/null
mkdir -p vendor runs
checkout() {
  local url="$1" destination="$2" revision="$3"
  if [[ ! -d "$destination/.git" ]]; then git clone "$url" "$destination"; fi
  if [[ "$(git -C "$destination" rev-parse HEAD)" != "$revision" ]]; then
    [[ -z "$(git -C "$destination" status --porcelain)" ]] || { echo "Local changes in $destination; preserve them before changing revision."; exit 1; }
    git -C "$destination" fetch origin "$revision"
    git -C "$destination" checkout "$revision"
  fi
}
checkout https://github.com/facebookresearch/sam-3d-objects.git vendor/sam-3d-objects f91db411c50efee93d8db7aeb323885650f6f722
checkout https://github.com/facebookresearch/pytorch3d.git vendor/pytorch3d 75ebeeaea0908c5527e7b1e305fbc7681382db47
checkout https://github.com/microsoft/MoGe.git vendor/MoGe a8c37341bc0325ca99b9d57981cc3bb2bd3e255b
checkout https://github.com/EasternJournalist/utils3d.git vendor/utils3d 3913c65d81e05e47b9f367250cf8c0f7462a0900
[[ -x .venv-sam3d/bin/python ]] || uv venv --python 3.11 .venv-sam3d
uv pip install --python .venv-sam3d/bin/python torch==2.5.1 torchvision==0.20.1 xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv-sam3d/bin/python --no-deps -r environments/sam3d-runtime.lock.txt
uv pip install --python .venv-sam3d/bin/python --no-deps ./vendor/MoGe ./vendor/utils3d https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121/kaolin-0.17.0-cp311-cp311-linux_x86_64.whl
if [[ ! -x vendor/cuda121/bin/nvcc ]]; then
  conda create -y -p "$project_dir/vendor/cuda121" --override-channels -c nvidia -c conda-forge cuda-nvcc=12.1.105 cuda-cudart-dev=12.1.105 cuda-cccl=12.1.109 gxx_linux-64=12
fi
.venv-sam3d/bin/python - <<'PY'
from pathlib import Path
root=Path.cwd();cuda=root/'vendor/cuda121';nvidia=root/'.venv-sam3d/lib/python3.11/site-packages/nvidia'
for folder in nvidia.glob('*/include'):
    for file in folder.iterdir():
        dest=cuda/'include'/file.name
        if not dest.exists() and not dest.is_symlink():dest.symlink_to(file)
dest=cuda/'lib/libcudart.so'
if dest.is_symlink() and not dest.exists():dest.unlink()
if not dest.exists():dest.symlink_to(nvidia/'cuda_runtime/lib/libcudart.so.12')
PY
export CUDA_HOME="$project_dir/vendor/cuda121" CONDA_PREFIX="$project_dir/vendor/cuda121"
export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-cc" CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++"
export MAX_JOBS="${MAX_JOBS:-2}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}" FORCE_CUDA=1
uv pip install --python .venv-sam3d/bin/python --no-build-isolation --no-deps ./vendor/pytorch3d
.venv-sam3d/bin/python vendor/sam-3d-objects/patching/hydra
PYTHONPATH="$project_dir/vendor/sam-3d-objects:$project_dir/vendor/sam-3d-objects/notebook" LIDRA_SKIP_INIT=true .venv-sam3d/bin/python -c 'from inference import Inference; print("Import check passed. Run a real masked-image inference before claiming qualification.")'
echo 'Set R2S_SAM3D_PYTHON, R2S_SAM3D_REPO, R2S_SAM3D_CONFIG in .env.local; weights use scripts/download_sam3d_modelscope.py.'
