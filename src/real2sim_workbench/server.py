import io
import json
import os
import shlex
import shutil
import threading
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field
from .models import Record, SceneRecord, ObjectRecord
from .store import Store, atomic_json
from . import adapters, geometry, physics
from .segmentation import AUTO_LABELS, load_mask, duplicate


class SegmentRequest(Record):
    mode: str = "text"
    labels: list[str] = Field(default_factory=list, max_length=12)
    box: list[float] | None = None
    points: list[list[float]] = Field(default_factory=list, max_length=50)
    object_id: str | None = None
    preview: bool = False
    label: str = Field(default="选中物体", max_length=100)


class SelectionAcceptRequest(Record):
    revision: int


class ImportRequest(Record):
    kind: str
    name: str


class GenerateRequest(Record):
    revision: int
    target: Literal["mujoco","isaac"] = "mujoco"
    seconds: float = Field(default=10, ge=.1, le=30)


class ActionRequest(Record):
    seconds: float = Field(default=3, ge=.1, le=30)
    object_ids: list[str] | None = Field(default=None, max_length=20)


def create_app(root=None, input_root=None):
    store = Store(root or os.environ.get("R2S_DATA_DIR", "runs"))
    from .inputs import InputLibrary
    library = InputLibrary(input_root or os.environ.get("R2S_INPUT_DIR") or Path(__file__).resolve().parents[2]/"data")
    app = FastAPI(title="Real2Sim with Astra", version="0.1.0")
    app.state.store = store
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="r2s")
    job_lock = threading.RLock()
    jobs = {}
    busy = set()
    from .previews import PreviewManager
    previews=PreviewManager(store)

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

    def submit(sid, kind, operation, manages_save=False):
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
                if not manages_save:
                    # Any geometry/layout change invalidates earlier validation/export.
                    if kind not in ("validate", "export", "capture", "review", "selection_preview", "asset_refine"):
                        scene.reports.pop("validation", None)
                        scene.reports.pop("export", None)
                        scene.reports.pop("views",None)
                        scene.reports.pop("astra_review",None)
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

    @app.get("/api/inputs")
    def inputs():
        return library.catalog()

    @app.post("/api/inputs/import")
    def import_input(req: ImportRequest):
        from .pipeline import prepare_input
        try:
            source = library.resolve(req.kind, req.name)
            _, scene = prepare_input(source, store.root)
        except (KeyError, TypeError) as exc:
            raise ValueError("采集元数据格式不完整，请参考 examples/rgbd_tabletop/capture") from exc
        return scene

    @app.get("/api/projects")
    def projects():
        return [{"id": s.id, "name": s.name, "objects": len(s.objects)} for s in store.list()]

    @app.post("/api/projects")
    async def upload(file: UploadFile = File(...)):
        data = await file.read(25*1024*1024+1)
        scene = store.create(data, Path(file.filename or "场景").stem)
        scene.preprocessing['input_path'] = 'single_image'
        # Store the validated, normalized image; filenames cannot escape the library.
        dest = library.root/'image'/(scene.id+'.png')
        shutil.copy2(store.asset(scene.id, scene.image), dest)
        scene.preprocessing['library_input'] = str(dest)
        return store.save(scene)

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
            if scene.preprocessing != original.preprocessing:
                raise ValueError("自动背景与排除区域不能由编辑接口修改")
            original_items = {o.id: o for o in original.objects}
            if set(original_items) != {o.id for o in scene.objects}:
                raise ValueError("此接口仅编辑现有物体；取消勾选可排除物体")
            for obj in scene.objects:
                old = original_items[obj.id]
                if any(getattr(obj,k)!=getattr(old,k) for k in ("mesh","mask","geometry","bbox","visual_texture","collision_meshes","reconstruction")):
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
        value=dict(jobs[jid])
        progress=store.path(value['scene_id'])/'jobs'/jid/'progress.json'
        if progress.is_file():
            try:value['progress']=json.loads(progress.read_text())
            except (ValueError,OSError):pass
        return value

    @app.post("/api/projects/{sid}/segment")
    def segment(sid: str, req: SegmentRequest):
        if not adapters.status()["sam3"]["configured"]:
            raise ValueError("SAM3 未配置")
        scene = store.load(sid)
        w, h = scene.image_size
        if req.mode not in ("text", "instance", "auto"):
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
            if req.mode == "auto":
                request["labels"] = AUTO_LABELS + ["carrot", "apple"]
            request["exclusion_masks"] = [str(store.asset(sid, scene.preprocessing["exclusion_mask"]))] if scene.preprocessing.get("exclusion_mask") else []
            result = adapters.run_worker("sam3", request, folder)
            if req.preview:
                if not result["objects"]:
                    raise ValueError("未找到有效物体，请增加正点或用负点排除机械臂/背景")
                row = max(result["objects"], key=lambda r:r["score"])
                scene.reports["selection_preview"] = dict(row, mask=str((folder/row["mask"]).relative_to(store.path(sid))), object_id=req.object_id)
                return
            background = result.get("background", {})
            if background:
                for key in ("exclusion_mask", "table_mask"):
                    if background.get(key):
                        scene.preprocessing[key] = str((folder/background[key]).relative_to(store.path(sid)))
                scene.preprocessing["exclude_labels"] = ["robot arm", "robot gripper"]
                scene.preprocessing["background"] = "simplified_support_plane"
                # A visible mask bounding rectangle is only an image-plane approximation.
                if background.get("table_bbox") and scene.table.calibration_source == "assumed":
                    x1,y1,x2,y2 = background["table_bbox"]
                    scene.table.corners = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
                    scene.preprocessing["table_mapping"] = "detected_mask_bounds_assumed_not_metric_calibration"
            existing_masks = [load_mask(store.asset(sid,o.mask)) for o in scene.objects if o.mask]
            added = 0
            rows = result["objects"]
            if req.mode == "auto":
                from .pipeline import select_rows
                rows = select_rows(rows, folder)
            for row in rows:
                if not req.object_id and duplicate(load_mask(folder/row["mask"]), existing_masks):
                    continue
                added += 1
                mask = str((folder/row["mask"]).relative_to(store.path(sid)))
                if req.object_id:
                    obj = next(o for o in scene.objects if o.id == req.object_id)
                    obj.mask, obj.bbox, obj.score = mask, row["bbox"], row["score"]
                    # Imported meshes are independent; previously generated mesh would be stale.
                    if obj.geometry == "sam3d":
                        obj.geometry, obj.mesh = "none", None
                        obj.visual_texture=None;obj.collision_meshes=[];obj.reconstruction={};obj.orientation=[1,0,0,0]
                else:
                    scene.objects.append(ObjectRecord(id="o_"+uuid.uuid4().hex[:12], name=row["name"],
                                                      bbox=row["bbox"], score=row["score"], mask=mask))
            scene.reports["segmentation"] = {"backend": "official_sam3", "candidate_count": len(result["objects"]), "added_count": added, "job": folder.name}
        return submit(sid, "selection_preview" if req.preview else "segment", operation)

    @app.delete("/api/projects/{sid}/selection/preview")
    def discard_selection(sid: str, req: SelectionAcceptRequest):
        with job_lock:
            available(sid)
            scene=store.load(sid)
            if scene.revision!=req.revision:raise RuntimeError("项目已更新，请刷新后重试")
            scene.reports.pop("selection_preview",None)
            return store.save(scene,expected=req.revision)

    @app.post("/api/projects/{sid}/selection/accept")
    def accept_selection(sid: str, req: SelectionAcceptRequest):
        with job_lock:
            available(sid)
            scene = store.load(sid)
            if scene.revision != req.revision:
                raise RuntimeError("项目已更新，请重新预览选择")
            row = scene.reports.get("selection_preview")
            if not row:
                raise ValueError("请先预览分割结果")
            obj = next((o for o in scene.objects if o.id == row.get("object_id")), None)
            if obj is None:
                mask = load_mask(store.asset(sid,row["mask"]))
                obj = next((o for o in scene.objects if o.mask and duplicate(mask,[load_mask(store.asset(sid,o.mask))])), None)
            if obj is None:
                obj = ObjectRecord(id="o_"+uuid.uuid4().hex[:12],name=row["name"],bbox=row["bbox"])
                scene.objects.append(obj)
            elif obj.geometry == "sam3d":
                obj.geometry, obj.mesh = "none", None
                obj.visual_texture=None;obj.collision_meshes=[];obj.reconstruction={};obj.orientation=[1,0,0,0]
            obj.mask, obj.bbox, obj.score = row["mask"], row["bbox"], row["score"]
            obj.selected = True
            if obj.support != "table" and not next(o for o in scene.objects if o.id == obj.support).selected:
                obj.support = "table"
            scene.reports = {}
            return store.save(scene, expected=req.revision)

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
            obj.visual_texture=None;obj.collision_meshes=[];obj.reconstruction={};obj.orientation=[1,0,0,0]
            # Preserve user-selected width; derive proportions, not metric scale.
            obj.size = [obj.size[0]*v/original[0] for v in original]
            ObjectRecord.model_validate(obj.model_dump())
            scene.reports = {}
            return store.save(scene, expected=scene.revision)

    @app.post("/api/projects/{sid}/generate")
    def generate(sid: str, req: GenerateRequest):
        from .pipeline import generate_selected
        with job_lock:
            available(sid)
            scene = store.load(sid)
            if scene.revision != req.revision:
                raise RuntimeError("项目已更新，请刷新后重试")
            if not adapters.status()['sam3d']['configured']:
                raise ValueError("SAM 3D 环境未配置；请完成模型设置后生成")
            selected = [o for o in scene.objects if o.selected]
            if not selected or any(not o.mask for o in selected):
                raise ValueError("请先用 SAM3 分割并保留至少一个物体")
            for obj in selected:store.asset(sid, obj.mask)
            def operation(scene, folder):
                def progress(stage):atomic_json(folder/'progress.json', {'stage':stage})
                from .pipeline import generate_target
                generate_target(store,scene,folder,target=req.target,seconds=req.seconds,progress=progress)
            return submit(sid, 'generate', operation, manages_save=True)

    @app.post("/api/projects/{sid}/recheck")
    def recheck(sid: str, req: GenerateRequest):
        from .pipeline import recheck_existing
        with job_lock:
            available(sid)
            scene=store.load(sid)
            if scene.revision!=req.revision:raise RuntimeError("项目已更新，请刷新后重试")
            if not any(o.selected for o in scene.objects) or any(o.selected and o.geometry=='none' for o in scene.objects):
                raise ValueError("请先生成资产；复检不会创建占位物体")
            def operation(scene,folder):
                recheck_existing(store,scene,folder,req.seconds,
                                 progress=lambda stage:atomic_json(folder/'progress.json',{'stage':stage}))
            return submit(sid,'recheck',operation,manages_save=True)

    @app.post("/api/projects/{sid}/layout")
    def auto_layout(sid: str, req: GenerateRequest):
        from .pipeline import recheck_existing
        with job_lock:
            available(sid)
            scene=store.load(sid)
            if scene.revision!=req.revision:raise RuntimeError("项目已更新，请刷新后重试")
            if not any(o.selected for o in scene.objects) or any(o.selected and o.geometry=='none' for o in scene.objects):
                raise ValueError("请先生成资产；复检不会创建占位物体")
            def operation(scene,folder):
                recheck_existing(store,scene,folder,req.seconds,auto_layout=True,
                                 progress=lambda stage:atomic_json(folder/'progress.json',{'stage':stage}))
            return submit(sid,'layout',operation,manages_save=True)

    @app.post("/api/projects/{sid}/table/fit")
    def fit_table(sid: str, req: SelectionAcceptRequest):
        import numpy as np
        with job_lock:
            available(sid)
            scene=store.load(sid)
            if scene.revision!=req.revision:raise RuntimeError("项目已更新，请刷新后重试")
            chosen=[o for o in scene.objects if o.selected and o.geometry!='none']
            if chosen:
                lower=np.min([np.asarray(o.position[:2])-geometry.half_extent(o)[:2] for o in chosen],axis=0)-.06
                upper=np.max([np.asarray(o.position[:2])+geometry.half_extent(o)[:2] for o in chosen],axis=0)+.06
                scene.table.width,scene.table.depth=map(float,upper-lower)
                scene.table.center_xy=((lower+upper)/2).tolist()
            else:
                scene.table.width,scene.table.depth=1.2,.8;scene.table.center_xy=[0.,0.]
            scene.table.calibration_source='user'
            scene.reports={}
            return store.save(scene,expected=req.revision)

    @app.get("/api/projects/{sid}/result")
    def result(sid: str):
        scene = store.load(sid)
        exported = scene.reports.get('export')
        xml = store.asset(sid, exported['path']+'/scene.xml') if exported else None
        repo = Path(__file__).resolve().parents[2]
        command = (f"cd {shlex.quote(str(repo))} && .venv/bin/python -m real2sim_workbench.cli view {shlex.quote(str(xml))}" if xml else None)
        isaac=scene.reports.get('isaac')
        usd=store.asset(sid,isaac['path']+'/scene.usdc') if isaac else None
        return {'project':str(store.path(sid)), 'xml':str(xml) if xml else None,'usd':str(usd) if usd else None,
                'view_command':command, 'reports':scene.reports}

    @app.post("/api/projects/{sid}/preview/{engine}")
    def preview(sid: str,engine: Literal['mujoco','isaac'],req: SelectionAcceptRequest):
        with job_lock:
            available(sid);scene=store.load(sid)
            if scene.revision!=req.revision:raise RuntimeError('项目已更新，请刷新后重试')
            return previews.launch(scene,engine)

    @app.get("/api/projects/{sid}/preview/{engine}")
    def preview_status(sid: str,engine: Literal['mujoco','isaac']):
        store.path(sid)
        return previews.status(sid,engine)

    @app.post("/api/projects/{sid}/actions/{action}")
    def action(sid: str, action: str, req: ActionRequest):
        if action not in ("proxy", "initialize", "optimize", "validate", "export", "reconstruct", "settle", "capture", "review", "asset_refine"):
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
                scene.reports["optimization"] = (physics.optimize_contacts(scene,store.path(sid),folder/"contacts")
                    if all(o.geometry=="sam3d" for o in scene.objects if o.selected) else geometry.optimize(scene))
            elif action == "asset_refine":
                from .asset_refinement import refine_assets
                refine_assets(scene,store.path(sid),folder,req.object_ids)
            elif action == "review":
                from .review import review_and_refine
                review_and_refine(scene,store.path(sid),folder)
            elif action == "settle":
                scene.reports["settling"]=physics.settle(scene,store.path(sid),folder/"settling")
            elif action == "capture":
                xml=physics.export_scene(scene,store.path(sid),folder/"simulation")
                report=physics.capture_views(scene,xml,folder/"views",reference=store.path(sid)/scene.image)
                report["path"]=str((folder/"views").relative_to(store.path(sid)))
                scene.reports["views"]=report
            elif action == "reconstruct":
                objects = [{"id": o.id, "mask": str(store.asset(sid, o.mask))} for o in scene.objects if o.selected and o.mask]
                if len(objects) != sum(o.selected for o in scene.objects):
                    raise ValueError("所有选中物体都需要分割掩码")
                request = {"image": str(store.asset(sid, scene.image)), "objects": objects, "seed": 42}
                if scene.preprocessing.get('depth_registration'):
                    request['registered_depth'] = str(store.asset(sid, scene.preprocessing['depth_registration']))
                result = adapters.run_worker("sam3d", request, folder, timeout=3600)
                from .reconstruction import assemble
                assemble(scene,store.path(sid),folder,result)
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
