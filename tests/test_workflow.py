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


def test_point_preview_accept_and_stale_revision(tmp_path,monkeypatch):
    from real2sim_workbench import adapters
    monkeypatch.setattr(adapters,'status',lambda:{'sam3':{'configured':True}})
    def worker(kind,request,folder):
        assert request['points']==[[250.,180.,1.],[400.,80.,0.]]
        mask=np.zeros((480,640),dtype=np.uint8);mask[100:240,200:300]=255
        Image.fromarray(mask).save(folder/'mask.png')
        return {'objects':[{'name':'hammer','mask':'mask.png','score':.95,'bbox':[200,100,300,240]}]}
    monkeypatch.setattr(adapters,'run_worker',worker)
    with TestClient(create_app(tmp_path/'runs')) as client:
        s=client.post('/api/projects',files={'file':('photo.png',image_bytes())}).json()
        path='/api/projects/'+s['id']
        job=client.post(path+'/segment',json={'mode':'instance','preview':True,'points':[[250,180,1],[400,80,0]]}).json()
        assert wait(client,job)['state']=='done'
        s=client.get(path).json()['scene'];assert not s['objects'];assert s['reports']['selection_preview']['score']==.95
        assert client.post(path+'/selection/accept',json={'revision':s['revision']-1}).status_code==409
        accepted=client.post(path+'/selection/accept',json={'revision':s['revision']})
        assert accepted.status_code==200
        s=accepted.json();assert len(s['objects'])==1 and s['objects'][0]['selected']
        assert 'selection_preview' not in s['reports']


def test_colored_mesh_with_separate_collision_loads_in_mujoco(tmp_path):
    from real2sim_workbench.geometry import prepare_visual_collision
    import mujoco
    store,scene=scene_pair(tmp_path)
    scene.objects=scene.objects[:1]
    obj=scene.objects[0]
    mesh=trimesh.creation.box(extents=[.12,.08,.04])
    mesh.visual.vertex_colors=np.tile([200,30,20,255],(len(mesh.vertices),1))
    prep=prepare_visual_collision(mesh,store.path(scene.id)/'colored.obj')
    obj.mesh='colored.obj';obj.geometry='sam3d';obj.size=prep['size'];obj.position=[0,0,.0202]
    obj.visual_texture=prep['texture'].name;obj.collision_meshes=[p.name for p in prep['collision']]
    xml=export_scene(scene,store.path(scene.id),tmp_path/'colored_sim')
    model=mujoco.MjModel.from_xml_path(str(xml))
    bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,obj.id)
    gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,obj.id+'_visual')
    assert model.ntex==1
    assert model.geom_contype[gid]==0 and model.geom_conaffinity[gid]==0
    assert model.body_mass[bid]==pytest.approx(obj.mass)
    assert validate_scene(scene,xml,tmp_path/'colored_sim',seconds=.5)['physics_check_pass']


def test_support_frame_robust_to_depth_outliers():
    from real2sim_workbench.reconstruction import fit_support_frame
    xs,ys=np.meshgrid(np.linspace(-.6,.6,100),np.linspace(-.3,.3,80))
    points=np.stack([xs,ys*.7-.3,ys+1.5],axis=-1)
    points[:5,:5]=[20,20,20]
    origin,basis,scale,report=fit_support_frame(points,np.ones(xs.shape,bool),1.2)
    assert np.linalg.det(basis)==pytest.approx(1)
    assert report['inlier_fraction']>.95
    assert np.std(((points[10:]-origin)@basis)[...,2])<1e-6
    assert 0.8<scale<1.2


def test_native_contact_optimizer_actually_separates_overlapping_bodies(tmp_path):
    from real2sim_workbench.physics import optimize_contacts
    store,scene=scene_pair(tmp_path,overlap=True)
    masses=[o.mass for o in scene.objects]
    report=optimize_contacts(scene,store.path(scene.id),tmp_path/'contacts')
    assert report['before_penetration_m']>.01
    assert report['after_penetration_m']<.002
    assert [o.mass for o in scene.objects]==masses
    xml=export_scene(scene,store.path(scene.id),tmp_path/'verified')
    assert validate_scene(scene,xml,tmp_path/'verified',seconds=1)['physics_check_pass']


def test_astra_rejects_out_of_bounds_model_proposal(tmp_path,monkeypatch):
    from real2sim_workbench import review
    from types import SimpleNamespace
    import json
    store,scene=scene_pair(tmp_path)
    out=tmp_path/'review';views=tmp_path/'views';views.mkdir()
    monkeypatch.setattr(review.shutil,'which',lambda _: '/mock/codex')
    def run(args,**kwargs):
        assert args[args.index('--model')+1]=='gpt-6-astra'
        assert args[args.index('--sandbox')+1]=='read-only'
        answer={'assessment':'test','visual_match':'partial','limitations':[],'pose_search_ids':[],
                'adjustments':[{'id':scene.objects[0].id,'dx_m':100,'dy_m':0,'yaw_deg':0,'reason':'invalid'}]}
        (out/'answer.json').write_text(json.dumps(answer))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(review.subprocess,'run',run)
    original=scene.model_dump()
    with pytest.raises(ValueError,match='超出保守边界'):
        review.astra_review(scene,store.path(scene.id),views,out)
    assert scene.model_dump()==original


def test_asset_recipes_reject_unbounded_values_and_duplicate_ids():
    from real2sim_workbench.asset_refinement import PARAMS,check_recipes
    import copy
    recipe={'id':'a','template':'button_box','button_color':'red','parameters':{k:(lo+hi)/2 for k,(lo,hi) in PARAMS.items()}}
    check_recipes({'assets':[recipe]},['a'])
    bad=copy.deepcopy(recipe);bad['parameters']['head_length']=float('nan')
    with pytest.raises(ValueError,match='bounds'):check_recipes({'assets':[bad]},['a'])
    bad=copy.deepcopy(recipe);bad['parameters']['housing_height']=.85;bad['parameters']['cap_height']=.15
    with pytest.raises(ValueError,match='stem'):check_recipes({'assets':[bad]},['a'])
    with pytest.raises(ValueError,match='IDs'):check_recipes({'assets':[recipe,recipe]},['a','b'])


def test_parametric_asset_geometry_is_closed_and_frame_roundtrips():
    from real2sim_workbench.asset_refinement import PARAMS,build_template,canonical_frame
    params={k:(lo+hi)/2 for k,(lo,hi) in PARAMS.items()}
    for template,extents in [('hammer',[.5,.1,.12]),('button_box',[.14,.13,.18])]:
        mesh=build_template({'template':template,'parameters':params,'button_color':'red'},extents)
        assert mesh.is_watertight and mesh.volume>0 and np.isfinite(mesh.vertices).all()
        assert mesh.extents[0]==pytest.approx(extents[0],rel=.01)
        transform,_=canonical_frame(mesh)
        assert np.linalg.det(transform[:3,:3])==pytest.approx(1.)
        copy=mesh.copy();copy.apply_transform(transform);copy.apply_transform(np.linalg.inv(transform))
        np.testing.assert_allclose(copy.vertices,mesh.vertices,atol=1e-10)


def test_asset_only_action_preserves_active_geometry_and_scene_reports(tmp_path,monkeypatch):
    from real2sim_workbench import asset_refinement
    store,scene=scene_pair(tmp_path)
    scene.reports={'export':{'path':'original'},'validation':{'physics_check_pass':True},'views':{'path':'before'},'astra_review':{'accepted':True}}
    store.save(scene)
    original=store.load(scene.id)
    def refine(candidate,project,folder,ids):
        assert ids==[scene.objects[0].id]
        candidate.reports['asset_refinement']={'stage':'asset_only','active_objects_unchanged':True}
    monkeypatch.setattr(asset_refinement,'refine_assets',refine)
    with TestClient(create_app(store.root)) as client:
        job=client.post('/api/projects/'+scene.id+'/actions/asset_refine',json={'object_ids':[scene.objects[0].id]}).json()
        assert wait(client,job)['state']=='done'
    current=store.load(scene.id)
    assert current.objects==original.objects
    for key,value in original.reports.items():assert current.reports[key]==value
    assert current.reports['asset_refinement']['stage']=='asset_only'
