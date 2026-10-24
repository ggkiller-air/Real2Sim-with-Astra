import io,json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import trimesh
from fastapi.testclient import TestClient
from real2sim_workbench.store import Store,atomic_json
from real2sim_workbench.models import ObjectRecord
from real2sim_workbench import isaac,pipeline
from real2sim_workbench.previews import PreviewManager
from real2sim_workbench.server import create_app


def fixture(tmp_path):
    store=Store(tmp_path/'runs');b=io.BytesIO();Image.new('RGB',(320,240)).save(b,format='PNG');s=store.create(b.getvalue(),'test');p=store.path(s.id)
    mesh=trimesh.creation.box();mesh.export(p/'block.obj')
    s.objects=[ObjectRecord(id='o_000000000001',name='block',bbox=[10,10,100,100],geometry='imported',mesh='block.obj',collision_meshes=['block.obj'],size=[.1,.2,.3],position=[0,0,.15],mass=.25,friction=.65)]
    return store,s,p


def test_manifest_preserves_metric_mesh_mass_inertia_and_parts(tmp_path):
    store,s,p=fixture(tmp_path);path=isaac.prepare_manifest(s,p,p/'isaac');data=json.loads(path.read_text());o=data['objects'][0]
    assert o['mass']==pytest.approx(.25) and o['friction']==.65
    np.testing.assert_allclose(np.ptp(o['visual']['vertices'],axis=0),[.1,.2,.3])
    assert len(o['collisions'])==1 and o['support']=='table'
    assert all(i>0 for i in o['inertia'])
    assert data['fingerprint']==isaac.scene_fingerprint(s)


def test_fingerprint_ignores_reports_but_not_user_input(tmp_path):
    _,s,_=fixture(tmp_path);original=isaac.scene_fingerprint(s);s.revision+=1;s.reports={'isaac':{'path':'foo'}}
    assert original==isaac.scene_fingerprint(s)
    s.objects[0].mass=.5;assert original!=isaac.scene_fingerprint(s)


def test_isaac_reuses_only_matching_reviewed_scene(tmp_path,monkeypatch):
    store,s,p=fixture(tmp_path);s.reports={'export':{'path':'mujoco'},'final_visual_review':{'actual_model_called':True}}
    atomic_json(p/'mujoco/scene_record.json',s.model_dump());calls=[]
    monkeypatch.setattr(isaac,'runtime_python',lambda:Path('/runtime/python.sh'))
    monkeypatch.setattr(isaac,'export_isaac',lambda *args:(calls.append('isaac') or args[1]))
    monkeypatch.setattr(pipeline,'generate_selected',lambda *args,**kwargs:(calls.append('full') or args[1]))
    pipeline.generate_target(store,s,p/'job',target='isaac');assert calls==['isaac']
    s.objects[0].mass=.3;calls.clear();pipeline.generate_target(store,s,p/'job2',target='isaac');assert calls==['full','isaac']


def test_preview_launch_deduplicates_and_uses_fixed_command(tmp_path,monkeypatch):
    store,s,p=fixture(tmp_path);s.reports={'export':{'path':'out'}};(p/'out').mkdir();(p/'out/scene.xml').write_text('<mujoco/>')
    monkeypatch.setenv('DISPLAY',':0');calls=[]
    class Process:
        pid=123
        def poll(self):return None
    def spawn(command,**kwargs):calls.append((command,kwargs));return Process()
    monkeypatch.setattr('real2sim_workbench.previews.subprocess.Popen',spawn)
    previews=PreviewManager(store)
    assert previews.launch(s,'mujoco')['state']=='starting';previews.launch(s,'mujoco');assert len(calls)==1
    cmd,kwargs=calls[0];assert cmd[1].endswith('preview_worker.py') and kwargs['env']['MUJOCO_GL']=='glfw'
    assert 'shell' not in kwargs
    ready=previews.processes[(s.id,'mujoco')]['ready'];ready.write_text('{}');assert previews.status(s.id,'mujoco')['state']=='ready'
    with pytest.raises(ValueError):previews.launch(s,'arbitrary-command')


def test_preview_api_revision_origin_and_backend_validation(tmp_path,monkeypatch):
    store,s,p=fixture(tmp_path);s=store.save(s)
    monkeypatch.setattr(PreviewManager,'launch',lambda self,scene,engine:{'state':'ready','engine':engine})
    with TestClient(create_app(store.root,input_root=tmp_path/'data')) as client:
        url=f'/api/projects/{s.id}/preview/mujoco'
        assert client.post(url,json={'revision':s.revision-1}).status_code==409
        assert client.post(url,json={'revision':s.revision},headers={'Origin':'https://untrusted.example'}).status_code==403
        assert client.post(url,json={'revision':s.revision}).json()['state']=='ready'
        assert client.post(url.replace('mujoco','shell'),json={'revision':s.revision}).status_code==422


def test_simple_ui_has_only_target_generation_controls():
    text=(Path(__file__).parents[1]/'src/real2sim_workbench/web/index.html').read_text()
    for old in ['id="autoLayout"','id="editTable"','id="showViews"','id="showAssets"','自动背景 · 排除机械臂']:
        assert old not in text
    for required in ['id="buildScene"','id="buildIsaac"','id="openMujoco"','id="openIsaac"']:
        assert required in text
