"""Isaac Sim export from the shared, reviewed scene (isolated NVIDIA runtime)."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
import trimesh
from . import physics
from .store import atomic_json


def runtime_python():
    configured=os.environ.get('R2S_ISAAC_PYTHON')
    candidates=[Path(configured)] if configured else [Path(__file__).resolve().parents[3]/'.tools/isaacsim-6.1.0/python_compat.sh']
    for path in candidates:
        if path.is_file():return path.resolve()
    raise ValueError('Isaac Sim 未配置：请设置 R2S_ISAAC_PYTHON 为 Isaac 的 python.sh 路径')


def worker_path():return Path(__file__).with_name('isaac_worker.py')


def scene_fingerprint(scene):
    from .models import SceneRecord
    scene=SceneRecord.model_validate(scene.model_dump())
    payload={'objects':[o.model_dump() for o in scene.objects if o.selected],'table':scene.table.model_dump(),'camera':scene.preprocessing.get('camera'),'image_sha256':scene.image_sha256}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def prepare_manifest(scene,project,output):
    """Bake asset scale once, preserve reviewed CoACD parts, mass and inertial frame."""
    import mujoco
    project,output=Path(project),Path(output);output.mkdir(parents=True,exist_ok=True)
    xml=physics.export_scene(scene,project,output/'mujoco_reference')
    model=mujoco.MjModel.from_xml_path(str(xml))
    manifest={'version':1,'fingerprint':scene_fingerprint(scene),'scene_id':scene.id,'table':scene.table.model_dump(),
              'table_rgba':scene.preprocessing.get('table_rgba',[.04,.43,.44,1]),'camera':scene.preprocessing.get('camera'),
              'image_size':scene.image_size,'objects':[],'source_revision':scene.revision}
    def read_mesh(rel,scale):
        path=(project/rel).resolve()
        if not path.is_relative_to(project.resolve()) or not path.is_file():raise ValueError('Invalid asset path')
        mesh=trimesh.load(path,force='mesh',process=False)
        uv=getattr(mesh.visual,'uv',None)
        return {'vertices':(mesh.vertices*np.asarray(scale)).tolist(),'faces':mesh.faces.tolist(),
                'uv':uv.tolist() if uv is not None else None}
    for o in scene.objects:
        if not o.selected:continue
        if not o.mesh:raise ValueError('Isaac 导出需要真实网格资产')
        bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,o.id)
        row={'id':o.id,'name':o.name,'support':o.support,'position':o.position,'orientation':physics.body_quaternion(o).tolist(),
             'mass':float(model.body_mass[bid]),'friction':o.friction,'inertia':model.body_inertia[bid].tolist(),
             'center_of_mass':model.body_ipos[bid].tolist(),'principal_axes':model.body_iquat[bid].tolist(),
             'visual':read_mesh(o.mesh,o.size),'collisions':[read_mesh(p,o.size) for p in o.collision_meshes or [o.mesh]],
             'annotation':o.annotation,'texture':None}
        if o.visual_texture:
            source=(project/o.visual_texture).resolve()
            if not source.is_relative_to(project.resolve()):raise ValueError('Invalid texture path')
            dest=output/'textures'/f'{o.id}.png';dest.parent.mkdir(exist_ok=True);shutil.copy2(source,dest)
            row['texture']=str(dest.relative_to(output))
        manifest['objects'].append(row)
    atomic_json(output/'manifest.json',manifest)
    return output/'manifest.json'


def export_isaac(store,scene,output,seconds=10,progress=None):
    runtime=runtime_python();output=Path(output).resolve();project=store.path(scene.id)
    if progress:progress('正在导出 Isaac Sim 材质、碰撞与物理属性')
    manifest=prepare_manifest(scene,project,output)
    args=[str(runtime),str(worker_path()),'--manifest',str(manifest),'--seconds',str(seconds)]
    env=os.environ.copy();env.pop('MUJOCO_GL',None)
    if progress:progress('Isaac Sim 正在加载场景并独立验证物理')
    with (output/'worker.log').open('w') as log:
        result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=900)
    if result.returncode or not all((output/name).is_file() for name in ('validation.json','scene.usdc','preview.png')):
        raise RuntimeError('Isaac Sim 导出或验证失败，查看 '+str(output/'worker.log'))
    report=json.loads((output/'validation.json').read_text())
    scene.reports['isaac']={'path':str(output.relative_to(project)),'fingerprint':scene_fingerprint(scene),
                           'route':'shared_scene_direct_usd','validation':report}
    scene=store.save(scene)
    atomic_json(output/'scene_record.json',scene.model_dump())
    return scene
