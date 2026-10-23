import io
import json
import os
import threading
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field
from .models import Record, SceneRecord, ObjectRecord
from .store import Store, atomic_json
from . import adapters, geometry, physics


class SegmentRequest(Record):
    mode: str = "text"
    labels: list[str] = Field(default_factory=list, max_length=12)
    box: list[float] | None = None
    points: list[list[float]] = Field(default_factory=list, max_length=50)
    object_id: str | None = None
    label: str = Field(default="选中物体", max_length=100)


class ActionRequest(Record):
    seconds: float = Field(default=3, ge=.1, le=30)


def create_app(root=None):
    store = Store(root or os.environ.get("R2S_DATA_DIR", "runs"))
    app = FastAPI(title="Real2Sim with Astra", version="0.1.0")
    app.state.store = store
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="r2s")
    job_lock = threading.RLock()
    jobs = {}
    busy = set()

    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin and origin != str(request.base_url).rstrip("/"):
            return JSONResponse({"detail": "仅允许当前本地界面提交修改"}, status_code=403)
        return await call_next(request)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(FileNotFoundError)
    async def missing(request, exc):
        return JSONResponse({"detail": "文件或项目不存在"}, status_code=404)

    @app.exception_handler(RuntimeError)
    async def conflict(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    def available(sid):
        store.path(sid)
        if sid in busy:
            raise RuntimeError("当前项目正在计算，完成后可继续编辑")

    def submit(sid, kind, operation):
        with job_lock:
            available(sid)
            jid = uuid.uuid4().hex[:16]
            folder = store.path(sid)/"jobs"/jid
            folder.mkdir(parents=True)
            job = {"id": jid, "scene_id": sid, "kind": kind, "state": "queued", "error": None}
            jobs[jid] = job
            busy.add(sid)
            atomic_json(folder/"job.json", job)

        def run():
            job["state"] = "running"
            atomic_json(folder/"job.json", job)
            try:
                scene = store.load(sid)
                revision = scene.revision
                operation(scene, folder)
                # Any geometry/layout change invalidates earlier validation/export.
                if kind not in ("validate", "export"):
                    scene.reports.pop("validation", None)
                    scene.reports.pop("export", None)
                store.save(scene, expected=revision)
                job["state"] = "done"
            except Exception as exc:
                job["state"] = "failed"
                job["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                atomic_json(folder/"job.json", job)
                with job_lock:
                    busy.discard(sid)
        executor.submit(run)
        return dict(job)

    @app.get("/api/status")
    def status():
        return adapters.status()

    @app.get("/api/projects")
    def projects():
        return [{"id": s.id, "name": s.name, "objects": len(s.objects)} for s in store.list()]

    @app.post("/api/projects")
    async def upload(file: UploadFile = File(...)):
        data = await file.read(25*1024*1024+1)
        return store.create(data, Path(file.filename or "场景").stem)

    @app.get("/api/projects/{sid}")
    def get(sid: str):
        scene = store.load(sid).model_dump()
        active = next((dict(j) for j in jobs.values() if j["scene_id"] == sid and j["state"] in ("running", "queued")), None)
        return {"scene": scene, "job": active}

    @app.put("/api/projects/{sid}")
    def update(sid: str, scene: SceneRecord):
        with job_lock:
            available(sid)
            original = store.load(sid)
            if scene.id != sid or scene.image != original.image or scene.image_sha256 != original.image_sha256 or scene.image_size != original.image_size:
                raise ValueError("不能修改原图身份信息")
            original_items = {o.id: o for o in original.objects}
            if set(original_items) != {o.id for o in scene.objects}:
                raise ValueError("此接口仅编辑现有物体；取消勾选可排除物体")
            for obj in scene.objects:
                old = original_items[obj.id]
                if (obj.mesh, obj.mask, obj.geometry, obj.bbox) != (old.mesh, old.mask, old.geometry, old.bbox):
                    raise ValueError("几何与分割请使用专用操作修改")
            geometry.homography(scene.table, scene.image_size)
            scene.reports = {}  # previous artifacts remain in run history, but are stale
            return store.save(scene, expected=scene.revision)

    @app.get("/api/projects/{sid}/files/{rel:path}")
    def file(sid: str, rel: str):
        return FileResponse(store.asset(sid, rel))

    @app.get("/api/jobs/{jid}")
    def job(jid: str):
        if jid not in jobs:
            raise HTTPException(404, "任务不在当前服务中；重启前的记录保存在项目 jobs 目录")
        return dict(jobs[jid])

    @app.post("/api/projects/{sid}/segment")
    def segment(sid: str, req: SegmentRequest):
        if not adapters.status()["sam3"]["configured"]:
            raise ValueError("SAM3 未配置")
        scene = store.load(sid)
        w, h = scene.image_size
        if req.mode not in ("text", "instance"):
            raise ValueError("未知分割模式")
        if req.mode == "text" and (not req.labels or any(not v.strip() or len(v)>100 for v in req.labels)):
            raise ValueError("请提供 1–12 个物体描述")
        if req.mode == "instance" and not req.box and not req.points:
            raise ValueError("请框选物体或添加提示点")
        if req.box and (len(req.box) != 4 or not (0 <= req.box[0] < req.box[2] <= w and 0 <= req.box[1] < req.box[3] <= h)):
            raise ValueError("框选坐标超出图片")
        if any(len(p) != 3 or not (0 <= p[0] < w and 0 <= p[1] < h and p[2] in (0,1)) for p in req.points):
            raise ValueError("提示点无效")
        if req.object_id and (req.mode != "instance" or req.object_id not in {o.id for o in scene.objects}):
            raise ValueError("修正目标无效")

        def operation(scene, folder):
            request = req.model_dump()
            request["image"] = str(store.asset(sid, scene.image))
            result = adapters.run_worker("sam3", request, folder)
            for row in result["objects"]:
                mask = str((folder/row["mask"]).relative_to(store.path(sid)))
                if req.object_id:
                    obj = next(o for o in scene.objects if o.id == req.object_id)
                    obj.mask, obj.bbox, obj.score = mask, row["bbox"], row["score"]
                    # Imported meshes are independent; previously generated mesh would be stale.
                    if obj.geometry == "sam3d":
                        obj.geometry, obj.mesh = "none", None
                else:
                    scene.objects.append(ObjectRecord(id="o_"+uuid.uuid4().hex[:12], name=row["name"],
                                                      bbox=row["bbox"], score=row["score"], mask=mask))
            scene.reports["segmentation"] = {"backend": "official_sam3", "candidate_count": len(result["objects"]), "job": folder.name}
        return submit(sid, "segment", operation)

    @app.post("/api/projects/{sid}/region")
    def manual_region(sid: str, req: SegmentRequest):
        with job_lock:
            available(sid)
            scene = store.load(sid)
            w, h = scene.image_size
            if not req.box or len(req.box) != 4 or not (0 <= req.box[0] < req.box[2] <= w and 0 <= req.box[1] < req.box[3] <= h):
                raise ValueError("请先框选有效区域")
            scene.objects.append(ObjectRecord(id="o_"+uuid.uuid4().hex[:12], name=req.label,
                                              bbox=req.box, segmentation_source="manual_box"))
            scene.reports = {}
            return store.save(scene, expected=scene.revision)

    @app.post("/api/projects/{sid}/objects/{oid}/mesh")
    async def mesh(sid: str, oid: str, file: UploadFile = File(...)):
        data = await file.read(30*1024*1024+1)
        ext = Path(file.filename or "").suffix.lower()
        if ext not in (".obj", ".glb", ".stl", ".ply") or len(data) > 30*1024*1024:
            raise ValueError("请使用 30 MB 以内的 OBJ / GLB / STL / PLY")
        # OBJ external material references are stripped before parsing.
        if ext == ".obj":
            data = b"\n".join(line for line in data.splitlines() if not line.lstrip().startswith((b"mtllib ", b"usemtl ")))
        with job_lock:
            available(sid)
            scene = store.load(sid)
            obj = next((o for o in scene.objects if o.id == oid), None)
            if obj is None:
                raise FileNotFoundError(oid)
            folder = store.path(sid)/"assets"/oid/uuid.uuid4().hex[:8]
            folder.mkdir(parents=True)
            source = folder/("source"+ext)
            source.write_bytes(data)
            original = geometry.import_mesh(source, folder/"normalized.obj")
            obj.mesh = str((folder/"normalized.obj").relative_to(store.path(sid)))
            obj.geometry = "imported"
            # Preserve user-selected width; derive proportions, not metric scale.
            obj.size = [obj.size[0]*v/original[0] for v in original]
            ObjectRecord.model_validate(obj.model_dump())
            scene.reports = {}
            return store.save(scene, expected=scene.revision)

    @app.post("/api/projects/{sid}/actions/{action}")
    def action(sid: str, action: str, req: ActionRequest):
        if action not in ("proxy", "initialize", "optimize", "validate", "export", "reconstruct"):
            raise HTTPException(404, "未知操作")
        if action == "reconstruct" and not adapters.status()["sam3d"]["configured"]:
            raise ValueError("SAM 3D Objects 尚无可用环境或权重；不会自动替换为占位模型")

        def operation(scene, folder):
            if not any(o.selected for o in scene.objects):
                raise ValueError("请先选择物体")
            if action == "proxy":
                for obj in scene.objects:
                    if obj.selected and obj.geometry == "none":
                        obj.geometry = "proxy"
                scene.reports["layout"] = geometry.initialize(scene)
            elif action == "initialize":
                scene.reports["layout"] = geometry.initialize(scene)
            elif action == "optimize":
                if any(o.selected and o.geometry == "none" for o in scene.objects):
                    raise ValueError("选中物体需要先具有几何")
                scene.reports["optimization"] = geometry.optimize(scene)
            elif action == "reconstruct":
                objects = [{"id": o.id, "mask": str(store.asset(sid, o.mask))} for o in scene.objects if o.selected and o.mask]
                if len(objects) != sum(o.selected for o in scene.objects):
                    raise ValueError("所有选中物体都需要分割掩码")
                result = adapters.run_worker("sam3d", {"image": str(store.asset(sid, scene.image)), "objects": objects, "seed": 42}, folder, timeout=3600)
                for row in result["objects"]:
                    obj = next(o for o in scene.objects if o.id == row["id"])
                    dest = folder/(obj.id+".obj")
                    original = geometry.import_mesh(folder/row["mesh"], dest)
                    obj.mesh, obj.geometry = str(dest.relative_to(store.path(sid))), "sam3d"
                    obj.size = [obj.size[0]*v/original[0] for v in original]
                scene.reports["reconstruction"] = {"job": folder.name, "raw_layout_applied": False,
                                                     "note": "原始模型姿态保存在 result.json；坐标变换尚未实模验证"}
            else:
                output = folder/"simulation"
                xml = physics.export_scene(scene, store.path(sid), output)
                scene.reports["export"] = {"path": str(output.relative_to(store.path(sid))), "source_revision": scene.revision}
                if action == "validate":
                    scene.reports["validation"] = physics.validate_scene(scene, xml, output, req.seconds)
        return submit(sid, action, operation)

    @app.get("/api/projects/{sid}/download")
    def download(sid: str):
        scene = store.load(sid)
        exported = scene.reports.get("export")
        if not exported:
            raise ValueError("请先导出当前版本")
        folder = store.path(sid)/exported["path"]
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            for item in folder.rglob("*"):
                if item.is_file():
                    archive.write(item, item.relative_to(folder))
        return Response(stream.getvalue(), media_type="application/zip", headers={"Content-Disposition": 'attachment; filename="mujoco-scene.zip"'})

    web = Path(__file__).with_name("web")
    app.mount("/static", StaticFiles(directory=web), name="static")

    @app.get("/")
    def index():
        return FileResponse(web/"index.html")

    return app
