import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from .geometry import support_order
from .store import atomic_json


def numbers(values):
    return " ".join(f"{float(v):.9g}" for v in values)


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
    ET.SubElement(root, "compiler", angle="radian", meshdir="assets", autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81", integrator="implicitfast", iterations="80", cone="elliptic")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="960", offheight="720")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -1 3", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(world, "geom", name="table", type="box", size=numbers([scene.table.width/2, scene.table.depth/2, .025]),
                  pos="0 0 -0.025", rgba="0.72 0.74 0.7 1", friction="0.5 0.005 0.0001")
    ET.SubElement(world, "geom", name="floor", type="plane", size="5 5 .1", pos="0 0 -0.8", rgba="0.16 0.19 0.22 1")
    warnings = []
    for index, obj in enumerate(objects):
        body = ET.SubElement(world, "body", name=obj.id, pos=numbers(obj.position),
                             quat=numbers([math.cos(obj.yaw/2), 0, 0, math.sin(obj.yaw/2)]))
        ET.SubElement(body, "freejoint")
        color = [(.23 + index*.31)% .65 + .15, (.38+index*.17)%.65+.15, (.55+index*.23)%.65+.15, 1]
        params = dict(name=obj.id+"_geom", mass=str(obj.mass), friction=numbers([obj.friction, .005, .0001]), rgba=numbers(color))
        if obj.geometry in ("imported", "sam3d"):
            source = (Path(folder)/obj.mesh).resolve()
            if not source.is_relative_to(Path(folder).resolve()) or not source.is_file():
                raise ValueError("网格文件不存在或路径非法")
            shutil.copy2(source, output / "assets" / f"{obj.id}.obj")
            ET.SubElement(asset, "mesh", name=obj.id, file=f"{obj.id}.obj", scale=numbers(obj.size))
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
    for contact in data.contact:
        if contact.dist <= .001:
            a, b = [int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
            active_contacts.add(frozenset((a, b)))
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
            supported = supported and data.xpos[bid, 2] >= obj.size[2]/2-.005
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
