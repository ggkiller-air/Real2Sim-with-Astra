import io
import time
import zipfile
import uuid
from pathlib import Path
import numpy as np
import pytest
import trimesh
from PIL import Image
from fastapi.testclient import TestClient
from pydantic import ValidationError
from real2sim_workbench.models import SceneRecord, ObjectRecord
from real2sim_workbench.store import Store
from real2sim_workbench.server import create_app
from real2sim_workbench.geometry import initialize, optimize, homography
from real2sim_workbench.physics import export_scene, validate_scene


def image_bytes():
    f = io.BytesIO()
    Image.new('RGB', (640, 480), '#dde2d5').save(f, format='PNG')
    return f.getvalue()


def object_record(**kwargs):
    return ObjectRecord(id='o_'+uuid.uuid4().hex[:12], name='block', bbox=[200, 100, 300, 240], **kwargs)


def scene_pair(tmp_path, overlap=False):
    store=Store(tmp_path/'runs')
    scene=store.create(image_bytes(), 'test')
    scene.objects=[object_record(geometry='proxy', position=[-.15,0,.0502]),
                   object_record(geometry='proxy', position=[-.15 if overlap else .15,0,.0502])]
    store.save(scene)
    return store,scene


def wait(client, job):
    for _ in range(300):
        current=client.get('/api/jobs/'+job['id']).json()
        if current['state'] not in ('running','queued'):
            return current
        time.sleep(.02)
    raise AssertionError('job did not finish')


def test_scene_validation_rejects_cycles_and_missing_support():
    a=object_record(); b=object_record(support=a.id); a.support=b.id
    with pytest.raises(ValidationError, match='循环'):
        SceneRecord(id='a'*16,name='test',image_sha256='x',image_size=[640,480],objects=[a,b])
    with pytest.raises(ValidationError):
        object_record(mass=-1)
    with pytest.raises(ValidationError):
        object_record(position=[float('nan'),0,0])


def test_image_coordinates_and_scale(tmp_path):
    store,scene=scene_pair(tmp_path)
    matrix=homography(scene.table,scene.image_size)
    p=matrix@[320,240,1]
    np.testing.assert_allclose(p[:2]/p[2],[0,0],atol=1e-9)
    scene.table.corners=[[0,0],[640,480],[640,0],[0,480]]
    with pytest.raises(ValueError,match='交叉'):
        homography(scene.table,scene.image_size)


def test_support_initialization(tmp_path):
    store,scene=scene_pair(tmp_path)
    scene.objects[1].support=scene.objects[0].id
    initialize(scene)
    a,b=scene.objects
    assert b.position[2]==pytest.approx(a.position[2]+a.size[2]/2+b.size[2]/2+.0002)
    assert b.position[:2]==a.position[:2]


def test_native_mujoco_stable_and_initial_overlap_failure(tmp_path):
    store,scene=scene_pair(tmp_path)
    xml=export_scene(scene,store.path(scene.id),tmp_path/'stable')
    report=validate_scene(scene,xml,tmp_path/'stable',seconds=1)
    assert report['physics_check_pass']
    assert report['contact_samples']==500
    scene.objects[1].position=scene.objects[0].position.copy()
    xml=export_scene(scene,store.path(scene.id),tmp_path/'overlap')
    report=validate_scene(scene,xml,tmp_path/'overlap',seconds=.2)
    assert report['initial_max_penetration_m']>.01
    assert not report['physics_check_pass']


def test_optimizer_preserves_mass_and_locked_pose(tmp_path):
    store,scene=scene_pair(tmp_path,overlap=True)
    scene.objects[0].locked=True
    original=scene.objects[0].model_dump()
    report=optimize(scene)
    assert report['objective_after']<report['objective_before']
    assert scene.objects[0].model_dump()==original
    assert scene.objects[1].mass==.1
    assert np.linalg.norm(np.array(scene.objects[1].position[:2])-scene.objects[0].position[:2])>.07


def test_api_real_export_mesh_import_and_stale_result(tmp_path,monkeypatch):
    monkeypatch.delenv('R2S_SAM3D_CONFIG',raising=False)
    app=create_app(tmp_path/'runs')
    with TestClient(app) as client:
        assert client.get('/').status_code==200
        scene=client.post('/api/projects',files={'file':('photo.png',image_bytes(),'image/png')}).json()
        sid=scene['id'];store=app.state.store
        record=store.load(sid);record.objects=[object_record(geometry='proxy',position=[0,0,.0502])];store.save(record)
        oid=record.objects[0].id
        # Import a mesh through the actual upload/normalization path.
        mesh=trimesh.creation.box(extents=[1,2,1]).export(file_type='obj').encode()
        response=client.post(f'/api/projects/{sid}/objects/{oid}/mesh',files={'file':('block.obj',mesh)})
        assert response.status_code==200,response.text
        assert response.json()['objects'][0]['size']==pytest.approx([.08,.16,.08])
        record=store.load(sid);record.objects[0].position[2]=.0402;store.save(record)
        job=client.post(f'/api/projects/{sid}/actions/validate',json={'seconds':1}).json()
        assert wait(client,job)['state']=='done'
        updated=client.get(f'/api/projects/{sid}').json()['scene']
        assert updated['reports']['validation']['physics_check_pass']
        blob=client.get(f'/api/projects/{sid}/download')
        with zipfile.ZipFile(io.BytesIO(blob.content)) as z:
            assert 'scene.xml' in z.namelist()
            assert f'assets/{oid}.obj' in z.namelist()
        updated['objects'][0]['mass']=.2
        response=client.put(f'/api/projects/{sid}',json=updated)
        assert response.status_code==200
        assert response.json()['reports']=={}
        assert client.get(f'/api/projects/{sid}/download').status_code==422
        assert client.put(f'/api/projects/{sid}',json=updated).status_code==409
        assert client.post(f'/api/projects/{sid}/actions/reconstruct',json={}).status_code==422
        assert client.post(f'/api/projects/{sid}/actions/export',json={},headers={'origin':'https://unrelated.example'}).status_code==403


def test_store_rejects_path_escape(tmp_path):
    store,scene=scene_pair(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.asset(scene.id,'../../secret')
    with pytest.raises(ValueError):
        store.path('../')


def test_manual_region_without_models(tmp_path):
    app=create_app(tmp_path/'runs')
    with TestClient(app) as client:
        scene=client.post('/api/projects',files={'file':('photo.png',image_bytes(),'image/png')}).json()
        path='/api/projects/'+scene['id']
        r=client.post(path+'/region',json={'box':[100,100,200,200],'label':'manual'})
        assert r.status_code==200
        assert r.json()['objects'][0]['segmentation_source']=='manual_box'
        assert r.json()['objects'][0]['mask'] is None
        job=client.post(path+'/actions/proxy',json={}).json()
        assert wait(client,job)['state']=='done'
        job=client.post(path+'/actions/validate',json={'seconds':1}).json()
        assert wait(client,job)['state']=='done'
        assert client.get(path).json()['scene']['reports']['validation']['physics_check_pass']


def test_candidate_dedup_and_robot_exclusion(tmp_path):
    from real2sim_workbench.segmentation import clean_candidates
    masks=[]
    for name,rect in [('cup',(10,10,25,25)),('tool',(10,10,25,25)),('arm',(40,10,70,50))]:
        mask=np.zeros((100,100),dtype=np.uint8); x,y,x2,y2=rect;mask[y:y2,x:x2]=255
        Image.fromarray(mask).save(tmp_path/f'{name}.png')
        masks.append({'name':name,'mask':f'{name}.png','score':.9})
    exclusion=np.zeros((100,100),dtype=bool);exclusion[10:50,40:70]=True
    assert [r['name'] for r in clean_candidates(masks,tmp_path,exclusion)]==['cup']


def test_auto_segmentation_preserves_removed_objects_and_background(tmp_path,monkeypatch):
    from real2sim_workbench import adapters
    monkeypatch.setattr(adapters,'status',lambda:{'sam3':{'configured':True}})
    def worker(kind,request,folder):
        assert request['mode']=='auto'
        assert 'hammer' in request['labels']
        mask=np.zeros((480,640),dtype=np.uint8);mask[300:340,200:240]=255
        Image.fromarray(mask).save(folder/'object.png')
        Image.fromarray(np.zeros_like(mask)).save(folder/'robot.png')
        return {'objects':[{'name':'hammer','mask':'object.png','score':.9,'bbox':[200,300,240,340]}],
                'background':{'exclusion_mask':'robot.png','table_bbox':[0,250,640,480]}}
    monkeypatch.setattr(adapters,'run_worker',worker)
    with TestClient(create_app(tmp_path/'runs')) as client:
        s=client.post('/api/projects',files={'file':('photo.png',image_bytes())}).json()
        path='/api/projects/'+s['id']
        assert wait(client,client.post(path+'/segment',json={'mode':'auto'}).json())['state']=='done'
        s=client.get(path).json()['scene'];assert len(s['objects'])==1
        assert s['table']['calibration_source']=='assumed'
        assert s['preprocessing']['exclude_labels']==['robot arm','robot gripper']
        s['objects'][0]['selected']=False
        assert client.put(path,json=s).status_code==200
        assert wait(client,client.post(path+'/segment',json={'mode':'auto'}).json())['state']=='done'
        s=client.get(path).json()['scene']
        assert len(s['objects'])==1 and not s['objects'][0]['selected']
        assert s['reports']['segmentation']['added_count']==0


def test_downloaded_weights_do_not_enable_unconfigured_runtime(tmp_path,monkeypatch):
    import json
    from real2sim_workbench.adapters import status
    config=tmp_path/'pipeline.yaml';config.write_text('config')
    monkeypatch.setenv('R2S_SAM3D_CONFIG',str(config))
    monkeypatch.delenv('R2S_SAM3D_PYTHON',raising=False)
    assert not status()['sam3d']['weights_present']
    weights=tmp_path/'model.ckpt';weights.write_bytes(b'123')
    (tmp_path/'download_verification.json').write_text(json.dumps({'complete':True,'files':[{'path':'model.ckpt','bytes':3}]}))
    assert status()['sam3d']['weights_present']
    assert not status()['sam3d']['configured']
    weights.write_bytes(b'')
    assert not status()['sam3d']['weights_present']
