import hashlib
import io
import json
import os
import re
import threading
import uuid
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError
from .models import SceneRecord


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def path(self, sid):
        if not re.fullmatch(r"[a-f0-9]{16}", sid):
            raise ValueError("非法项目 ID")
        path = self.root / sid
        if not path.is_dir():
            raise FileNotFoundError(sid)
        return path

    def asset(self, sid, rel):
        base = self.path(sid)
        path = (base / rel).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            raise FileNotFoundError(rel)
        return path

    def load(self, sid):
        return SceneRecord.model_validate_json((self.path(sid) / "scene.json").read_text())

    def save(self, scene, expected=None):
        scene = SceneRecord.model_validate(scene.model_dump())
        with self.lock:
            dest = self.path(scene.id) / "scene.json"
            if dest.exists():
                current = self.load(scene.id)
                if expected is not None and current.revision != expected:
                    raise RuntimeError("项目已更新，请刷新后重试")
                scene.revision = current.revision + 1
            atomic_json(dest, scene.model_dump())
        return scene

    def create(self, content, name):
        if len(content) > 25 * 1024 * 1024:
            raise ValueError("图片最大 25 MB")
        try:
            im = ImageOps.exif_transpose(Image.open(io.BytesIO(content)))
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError("无法读取图片，请使用有效的 PNG/JPEG/WebP") from exc
        if im.width * im.height > 24_000_000:
            raise ValueError("图片最大 2400 万像素")
        im = im.convert("RGB")
        sid = uuid.uuid4().hex[:16]
        folder = self.root / sid
        (folder / "input").mkdir(parents=True)
        im.save(folder / "input/source.png")
        scene = SceneRecord(id=sid, name=name[:100] or "未命名场景", image_size=list(im.size),
                            image_sha256=hashlib.sha256((folder / "input/source.png").read_bytes()).hexdigest())
        return self.save(scene)

    def list(self):
        return [self.load(p.parent.name) for p in sorted(self.root.glob("*/scene.json"), key=lambda p: p.stat().st_mtime, reverse=True)]
