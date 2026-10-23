import os
os.environ.setdefault("MUJOCO_GL", "egl")
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from .geometry import support_order
from .store import atomic_json


def numbers(values):
    return " ".join(f"{float(v):.9g}" for v in values)


def body_quaternion(obj):
    from scipy.spatial.transform import Rotation
    q=np.asarray(obj.orientation)
    base=Rotation.from_quat(q[[1,2,3,0]])
    q=(Rotation.from_euler('z',obj.yaw)*base).as_quat()
    return q[[3,0,1,2]]


def export_scene(scene, folder, output):
    """Export ordinary rigid bodies. Special annotations are never silently applied."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "assets").mkdir(exist_ok=True)
    objects = support_order(scene)
    if not objects:
        raise ValueError("请至少选择一个物体")
    if any(o.geometry == "none" for o in objects):
        raise ValueError("有物体尚无几何；请导入网格或明确生成占位几何")
    root = ET.Element("mujoco", model="Real2Sim with Astra")
    ET.SubElement(root, "compiler", angle="radian", meshdir="assets", texturedir="assets", autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81", integrator="implicitfast", iterations="80", cone="elliptic")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="960", offheight="720")
    ET.SubElement(visual,"headlight",ambient="0.35 0.35 0.35",diffuse="0.65 0.65 0.65",specular="0.1 0.1 0.1")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -1 3", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(world, "geom", name="table", type="box", size=numbers([scene.table.width/2, scene.table.depth/2, .025]),
                  pos="0 0 -0.025", rgba=numbers(scene.preprocessing.get("table_rgba",[.04,.43,.44,1])), friction="0.5 0.005 0.0001")
    ET.SubElement(world, "geom", name="floor", type="plane", size="5 5 .1", pos="0 0 -0.8", rgba="0.16 0.19 0.22 1")
    camera=scene.preprocessing.get("camera")
    if camera:
        ET.SubElement(world,"camera",name="source",pos=numbers(camera["position"]),xyaxes=numbers(camera["xyaxes"]),fovy=str(camera["fovy"]))
    warnings = []
    for index, obj in enumerate(objects):
        body = ET.SubElement(world, "body", name=obj.id, pos=numbers(obj.position),
                             quat=numbers(body_quaternion(obj)))
        ET.SubElement(body, "freejoint")
        color = [(.23 + index*.31)% .65 + .15, (.38+index*.17)%.65+.15, (.55+index*.23)%.65+.15, 1]
        params = dict(name=obj.id+"_geom", mass=str(obj.mass), friction=numbers([obj.friction, .005, .0001]), rgba=numbers(color))
        if obj.geometry in ("imported", "sam3d"):
            source = (Path(folder)/obj.mesh).resolve()
            if not source.is_relative_to(Path(folder).resolve()) or not source.is_file():
                raise ValueError("网格文件不存在或路径非法")
            shutil.copy2(source, output / "assets" / f"{obj.id}.obj")
            ET.SubElement(asset, "mesh", name=obj.id, file=f"{obj.id}.obj", scale=numbers(obj.size))
            if obj.visual_texture:
                texture_source = (Path(folder)/obj.visual_texture).resolve()
                if not texture_source.is_relative_to(Path(folder).resolve()) or not texture_source.is_file():
                    raise ValueError("纹理文件不存在或路径非法")
                shutil.copy2(texture_source, output / "assets" / f"{obj.id}.png")
                ET.SubElement(asset, "texture", name=obj.id+"_tex", type="2d", file=f"{obj.id}.png")
                ET.SubElement(asset, "material", name=obj.id+"_mat", texture=obj.id+"_tex", rgba="1 1 1 1", specular="0.15", shininess="0.2")
            if obj.collision_meshes:
                visual_params = {"material":obj.id+"_mat"} if obj.visual_texture else {"rgba":numbers(color)}
                ET.SubElement(body, "geom", name=obj.id+"_visual", type="mesh", mesh=obj.id,
                              contype="0", conaffinity="0", mass="0", group="2", **visual_params)
                import trimesh
                parts=[]
                for i, relative in enumerate(obj.collision_meshes):
                    source=(Path(folder)/relative).resolve()
                    if not source.is_relative_to(Path(folder).resolve()) or not source.is_file():
                        raise ValueError("碰撞网格不存在或路径非法")
                    name=f"{obj.id}_collision_{i}"
                    shutil.copy2(source,output/"assets"/(name+".obj"))
                    volume=abs(trimesh.load(source,force="mesh").volume)
                    parts.append((name,max(volume,1e-12)))
                    ET.SubElement(asset,"mesh",name=name,file=name+".obj",scale=numbers(obj.size))
                total=sum(v for _,v in parts)
                for name,volume in parts:
                    ET.SubElement(body,"geom",name=name,type="mesh",mesh=name,mass=str(obj.mass*volume/total),
                                  friction=numbers([obj.friction,.005,.0001]),rgba="0 0 0 0",group="3")
                warnings.append(f"{obj.name}: {len(parts)} 个近似凸体碰撞；质量按凸体体积分配")
            else:
                if obj.visual_texture:
                    params.pop("rgba")
                    params["material"]=obj.id+"_mat"
                ET.SubElement(body, "geom", type="mesh", mesh=obj.id, **params)
                warnings.append(f"{obj.name}: 使用单个凸包碰撞，凹洞和容器内腔未保留")
        else:
            ET.SubElement(body, "geom", type="box", size=numbers(np.array(obj.size)/2), **params)
            warnings.append(f"{obj.name}: 人工尺寸占位盒，非照片生成资产")
        if obj.annotation != "rigid":
            warnings.append(f"{obj.name}: {obj.annotation} 仅标注，本版仍按刚体处理")
    ET.indent(root)
    ET.ElementTree(root).write(output / "scene.xml", encoding="utf-8", xml_declaration=True)
    atomic_json(output / "scene_record.json", scene.model_dump())
    atomic_json(output / "export_report.json", {"warnings": warnings, "engine": "mujoco", "source_revision": scene.revision,
                                                "visual_match_validated": False, "special_structures_applied": False})
    return output / "scene.xml"


def validate_scene(scene, xml, output, seconds=3.0):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    initial = data.xpos.copy()
    initial_q = data.xquat.copy()
    initial_contacts = [{"a": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom1)),
                         "b": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom2)),
                         "penetration_m": max(0., -float(c.dist))} for c in data.contact]
    steps = round(seconds/model.opt.timestep)
    max_depth = 0.
    sampled = 0
    finite = True
    for _ in range(steps):
        mujoco.mj_step(model, data)
        sampled += 1
        finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        max_depth = max(max_depth, max((-float(c.dist) for c in data.contact), default=0))
        if not finite or any(w.number for w in data.warning):
            break
    # Refresh derived body transforms at the final integrated state.
    mujoco.mj_forward(model, data)
    active_contacts = set()
    table_bodies=set()
    table_gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"table")
    for contact in data.contact:
        if contact.dist <= .001:
            a, b = [int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
            active_contacts.add(frozenset((a, b)))
            if contact.geom1==table_gid:table_bodies.add(b)
            if contact.geom2==table_gid:table_bodies.add(a)
    objects = []
    for obj in scene.objects:
        if not obj.selected:
            continue
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj.id)
        parent = 0 if obj.support == "table" else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj.support)
        angle = 2*math.acos(float(np.clip(abs(np.dot(initial_q[bid], data.xquat[bid])), 0, 1)))
        drift = float(np.linalg.norm(data.xpos[bid]-initial[bid]))
        joint = model.body_jntadr[bid]
        dof = model.jnt_dofadr[joint]
        supported = frozenset((bid, parent)) in active_contacts
        # World includes the floor: require the intended table height as well.
        if obj.support == "table":
            supported = bid in table_bodies
        speed = float(np.linalg.norm(data.qvel[dof:dof+3]))
        angular = float(np.linalg.norm(data.qvel[dof+3:dof+6]))
        objects.append({"id": obj.id, "name": obj.name, "translation_m": drift, "rotation_deg": math.degrees(angle),
                        "final_position": data.xpos[bid].tolist(), "final_speed_m_s": speed,
                        "final_angular_speed_rad_s": angular, "expected_support_contact": bool(supported),
                        "stable": bool(drift <= .01 and angle <= math.radians(10) and speed <= .02 and angular <= .2 and supported)})
    warnings = [{"index": i, "count": int(w.number)} for i, w in enumerate(data.warning) if w.number]
    initial_depth = max((c["penetration_m"] for c in initial_contacts), default=0.)
    report = {"source_revision": scene.revision, "mujoco_version": mujoco.__version__, "requested_seconds": seconds,
              "simulated_seconds": float(data.time), "contact_samples": sampled, "expected_steps": steps,
              "finite": finite, "warnings": warnings, "initial_max_penetration_m": initial_depth,
              "max_penetration_m": max_depth, "initial_contacts": initial_contacts, "objects": objects,
              "thresholds": {"initial_penetration_m": .002, "drift_m": .01, "rotation_deg": 10, "speed_m_s": .02, "angular_speed_rad_s": .2},
              "visual_match_validated": False,
              "limitations": ["刚体与凸包近似", "短时重力检查，不证明真实物理属性、抓取或任务成功"]}
    report["physics_check_pass"] = bool(finite and not warnings and sampled == steps and
                                        abs(data.time-seconds) < model.opt.timestep and initial_depth <= .002 and all(o["stable"] for o in objects))
    atomic_json(Path(output)/"validation.json", report)
    return report


def settle(scene, folder, output, seconds=3):
    """Accept gravity settling as a recorded pose proposal; validate separately."""
    import mujoco
    xml=export_scene(scene,folder,output)
    model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    old={o.id:{'position':list(o.position),'orientation':list(o.orientation),'yaw':o.yaw} for o in scene.objects if o.selected}
    for _ in range(round(seconds/model.opt.timestep)):
        mujoco.mj_step(model,data)
        if not np.isfinite(data.qpos).all() or any(w.number for w in data.warning):
            raise ValueError('落桌预处理数值异常，未应用姿态')
    mujoco.mj_forward(model,data)
    changes=[]
    for obj in scene.objects:
        if not obj.selected or obj.locked:continue
        bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,obj.id)
        position=data.xpos[bid].copy()
        delta=float(np.linalg.norm(position-np.asarray(obj.position)))
        if delta>.15 or position[2]<-.01:
            raise ValueError(f'{obj.name} 落桌移动过大或掉落，需修正布局')
        obj.position=position.tolist();obj.orientation=data.xquat[bid].tolist();obj.yaw=0
        changes.append({'id':obj.id,'before':old[obj.id],'after':{'position':obj.position,'orientation':obj.orientation},'translation_m':delta})
    report={'method':'gravity_settle_pose_proposal','simulated_seconds':float(data.time),'changes':changes,
            'note':'This proposes initial poses; a separate unchanged-threshold stability check follows.'}
    atomic_json(Path(output)/'settling.json',report)
    return report


def capture_views(scene,xml,output,reference=None):
    import os
    os.environ.setdefault('MUJOCO_GL','egl')
    import mujoco
    from PIL import Image,ImageDraw
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    renderer=mujoco.Renderer(model,height=480,width=640)
    option=mujoco.MjvOption();option.geomgroup[3]=0
    views=[]
    if reference is not None:
        Image.open(reference).convert('RGB').resize((640,480)).save(output/'reference.png')
        views.append('reference.png')
    try:
        if scene.preprocessing.get('camera'):
            renderer.update_scene(data,camera='source',scene_option=option)
            path=output/'source_view.png';Image.fromarray(renderer.render()).save(path);views.append(path.name)
        center=np.mean([o.position for o in scene.objects if o.selected],axis=0)
        distance=max(scene.table.width,scene.table.depth)*1.35
        for name,azimuth,elevation in [('overview',90,-35),('top',90,-89),('side',0,-15),('opposite',270,-25)]:
            cam=mujoco.MjvCamera();cam.lookat[:]=center;cam.distance=distance;cam.azimuth=azimuth;cam.elevation=elevation
            renderer.update_scene(data,camera=cam,scene_option=option)
            path=output/(name+'.png');Image.fromarray(renderer.render()).save(path);views.append(path.name)
    finally:renderer.close()
    sheet=Image.new('RGB',(1280,3*510),'#e8ece7');draw=ImageDraw.Draw(sheet)
    for i,name in enumerate(views):
        x=(i%2)*640;y=(i//2)*510
        sheet.paste(Image.open(output/name),(x,y+30));draw.text((x+10,y+8),name,fill='black')
    sheet.save(output/'views.png')
    report={'views':views,'contact_state':'initial exported pose','engine':'MuJoCo native renderer'}
    atomic_json(output/'views.json',report)
    return report


def silhouette_metrics(scene, project, xml):
    """Visible object silhouette IoU from the native reference camera (not RGB)."""
    import os
    os.environ.setdefault('MUJOCO_GL','egl')
    import mujoco
    from PIL import Image
    if not scene.preprocessing.get('camera'):return {'available':False}
    width,height=scene.image_size
    model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    renderer=mujoco.Renderer(model,height=height,width=width)
    option=mujoco.MjvOption();option.geomgroup[3]=0
    try:
        renderer.enable_segmentation_rendering();renderer.update_scene(data,camera='source',scene_option=option)
        segmentation=renderer.render().copy()
    finally:renderer.close()
    rows=[]
    for obj in scene.objects:
        if not obj.selected or not obj.mask:continue
        bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,obj.id)
        geoms=np.where(model.geom_bodyid==bid)[0]
        predicted=np.isin(segmentation[:,:,0],geoms)&(segmentation[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
        expected=np.asarray(Image.open(Path(project)/obj.mask).convert('L'))>127
        union=np.logical_or(predicted,expected).sum()
        iou=float(np.logical_and(predicted,expected).sum()/union) if union else 0.
        def centroid(mask):
            yy,xx=np.where(mask)
            return [float(xx.mean()),float(yy.mean())] if len(xx) else None
        rows.append({'id':obj.id,'iou':iou,'rendered_pixels':int(predicted.sum()),'reference_pixels':int(expected.sum()),
                     'rendered_centroid':centroid(predicted),'reference_centroid':centroid(expected)})
    return {'available':True,'objects':rows,'mean_iou':float(np.mean([r['iou'] for r in rows])) if rows else 0.,
            'limitation':'Visible silhouette only; excludes RGB appearance, hidden geometry and metric accuracy.'}


def optimize_contacts(scene,folder,output):
    """Bounded native-contact optimization; retain the best observed proposal."""
    import mujoco
    from scipy.optimize import minimize
    from .geometry import half_extent
    xml=export_scene(scene,folder,output)
    model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model)
    objects={mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,o.id):o for o in scene.objects if o.selected}
    moving=[(bid,o) for bid,o in objects.items() if not o.locked]
    if not moving:return {'method':'native_contact_search','changed':False}
    addresses=[model.jnt_qposadr[model.body_jntadr[bid]] for bid,_ in moving]
    anchors=np.array([o.position[:2] for _,o in moving])
    best=[float('inf'),anchors.ravel().copy(),0.]
    def objective(flat):
        xy=flat.reshape(-1,2)
        for q,pos in zip(addresses,xy):data.qpos[q:q+2]=pos
        mujoco.mj_forward(model,data)
        depths={}
        for contact in data.contact:
            a,b=[int(model.geom_bodyid[g]) for g in (contact.geom1,contact.geom2)]
            if a not in objects or b not in objects or a==b:continue
            key=tuple(sorted((a,b)))
            depths[key]=max(depths.get(key,0.),max(0.,-float(contact.dist)))
        max_depth=max(depths.values(),default=0.)
        loss=float(np.square(xy-anchors).sum())
        loss+=10000*sum(max(0.,d-.0003)**2 for d in depths.values())
        for (_,obj),pos in zip(moving,xy):
            ext=half_extent(obj)[:2]
            overflow=np.maximum(np.abs(pos)+ext-[scene.table.width/2,scene.table.depth/2],0)
            loss+=10000*float(overflow@overflow)
        if loss<best[0]:best[:]=[loss,flat.copy(),max_depth]
        return loss
    baseline=objective(anchors.ravel());initial_depth=best[2]
    solved=minimize(objective,anchors.ravel(),method='Powell',
                    bounds=[(v-.12,v+.12) for v in anchors.ravel()],
                    options={'maxiter':35,'maxfev':3500,'xtol':1e-5,'ftol':1e-6})
    for (_,obj),xy in zip(moving,best[1].reshape(-1,2)):obj.position[:2]=xy.tolist()
    return {'method':'native_convex_contact_bounded_powell','before_penetration_m':initial_depth,
            'after_penetration_m':best[2],'objective_before':baseline,'objective_after':best[0],
            'evaluations':int(solved.nfev),'converged':bool(solved.success),'max_axis_translation_m':.12,
            'changes':[{'id':o.id,'delta_xy':(np.asarray(o.position[:2])-anchor).tolist()} for (_,o),anchor in zip(moving,anchors)]}
