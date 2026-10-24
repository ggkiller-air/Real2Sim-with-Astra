"""Subprocess boundary: GPU model dependencies never enter the web server."""
import json
import os
import subprocess
import sys
from pathlib import Path


def settings():
    return {k: os.environ.get(k, "") for k in (
        "R2S_SAM3_PYTHON", "R2S_SAM3_CHECKPOINT", "R2S_SAM3D_PYTHON", "R2S_SAM3D_REPO", "R2S_SAM3D_CONFIG")}


def verified_weights_present(config):
    if not config or not Path(config).is_file():
        return False
    root = Path(config).resolve().parent
    try:
        report = json.loads((root / "download_verification.json").read_text())
        return bool(report.get("complete") and report.get("files") and all(
            (root / row["path"]).resolve().is_relative_to(root)
            and (root / row["path"]).stat().st_size == row["bytes"]
            for row in report["files"]
        ))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def runtime_qualification(settings):
    if not settings["R2S_SAM3D_REPO"]:return False
    report=Path(settings["R2S_SAM3D_REPO"]).resolve().parent.parent/"runs/sam3d-runtime-qualification.json"
    try:
        value=json.loads(report.read_text())
        return bool(value.get("real_inference_pass") and value.get("python")==settings["R2S_SAM3D_PYTHON"]
                    and value.get("config")==settings["R2S_SAM3D_CONFIG"])
    except (OSError,ValueError):return False


def status():
    s = settings()
    return {
        "sam3": {"configured": all(Path(s[k]).is_file() for k in ("R2S_SAM3_PYTHON", "R2S_SAM3_CHECKPOINT") if s[k]) and all(s[k] for k in ("R2S_SAM3_PYTHON", "R2S_SAM3_CHECKPOINT")), "description": "本地 SAM3 · 文字候选 / 框与正负点"},
        "sam3d": {"real_inference_verified":runtime_qualification(s), "weights_present": verified_weights_present(s["R2S_SAM3D_CONFIG"]), "configured": all(s[k] and Path(s[k]).exists() for k in ("R2S_SAM3D_PYTHON", "R2S_SAM3D_REPO", "R2S_SAM3D_CONFIG")), "description": "需要独立推理环境；权重就绪不代表已完成真实推理验证"},
        "engine": "MuJoCo", "special_structures": "annotation_only",
    }


def worker_environment(environ=None):
    """Normalize the common desktop SOCKS alias for HTTPX/Requests.

    Preserve proxy host, credentials and all other settings; never change the
    parent shell or silently disable routing. Explicit SOCKS4/5 URLs stay intact.
    """
    env=dict(os.environ if environ is None else environ)
    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
        value=env.get(key,'')
        if value.lower().startswith('socks://'):
            env[key]='socks5://'+value[len('socks://'):]
    return env


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
                                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout, env=worker_environment())
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"{kind} 超时；日志：{folder / 'worker.log'}") from exc
    if result.returncode:
        raise RuntimeError(f"{kind} 运行失败；查看 {folder / 'worker.log'}")
    return json.loads(result_file.read_text())
