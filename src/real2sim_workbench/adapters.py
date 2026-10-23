"""Subprocess boundary: GPU model dependencies never enter the web server."""
import json
import os
import subprocess
import sys
from pathlib import Path


def settings():
    return {k: os.environ.get(k, "") for k in (
        "R2S_SAM3_PYTHON", "R2S_SAM3_CHECKPOINT", "R2S_SAM3D_PYTHON", "R2S_SAM3D_REPO", "R2S_SAM3D_CONFIG")}


def status():
    s = settings()
    return {
        "sam3": {"configured": all(Path(s[k]).is_file() for k in ("R2S_SAM3_PYTHON", "R2S_SAM3_CHECKPOINT") if s[k]) and all(s[k] for k in ("R2S_SAM3_PYTHON", "R2S_SAM3_CHECKPOINT")), "description": "本地 SAM3 · 文字候选 / 框与正负点"},
        "sam3d": {"configured": all(s[k] and Path(s[k]).exists() for k in ("R2S_SAM3D_PYTHON", "R2S_SAM3D_REPO", "R2S_SAM3D_CONFIG")), "description": "需要官方权重访问权限及独立运行环境；此适配器尚未实模验证"},
        "engine": "MuJoCo", "special_structures": "annotation_only",
    }


def run_worker(kind, request, folder, timeout=1200):
    s = settings()
    if not status()[kind]["configured"]:
        raise ValueError(f"{kind} 未配置；请查看环境设置")
    from .store import atomic_json
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    request_file = folder / "request.json"
    result_file = folder / "result.json"
    atomic_json(request_file, request)
    worker = Path(__file__).with_name("workers") / ("sam3_worker.py" if kind == "sam3" else "sam3d.py")
    interpreter = s["R2S_SAM3_PYTHON" if kind == "sam3" else "R2S_SAM3D_PYTHON"]
    with (folder / "worker.log").open("w") as log:
        try:
            result = subprocess.run([interpreter, str(worker), str(request_file), str(result_file)],
                                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout, env=os.environ.copy())
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"{kind} 超时，日志已保存") from exc
    if result.returncode:
        raise RuntimeError(f"{kind} 运行失败；查看 {folder.name}/worker.log")
    return json.loads(result_file.read_text())
