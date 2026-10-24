"""GUI route contracts. Model inference is stubbed; calibration uses real sample data."""
import io
import json
import shutil
import threading
import time
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from real2sim_workbench import pipeline
from real2sim_workbench.models import ObjectRecord
from real2sim_workbench.server import create_app
from real2sim_workbench.store import atomic_json

EXAMPLE=Path(__file__).resolve().parents[1]/'examples/rgbd_tabletop/capture'


def wait(client,job):
    for _ in range(300):
        state=client.get('/api/jobs/'+job['id']).json()
        if state['state'] not in ('queued','running'):return state
        time.sleep(.01)
    raise AssertionError('job timeout')


@pytest.mark.parametrize('kind',['image','depth'])
def test_import_registers_depth_before_interactive_masks(tmp_path,monkeypatch,kind):
    library=tmp_path/'data';(library/'image').mkdir(parents=True);(library/'depth').mkdir()
    shutil.copy2(EXAMPLE/'image_external_rgb.jpg',library/'image/table.jpg')
    shutil.copytree(EXAMPLE,library/'depth/capture')
    app=create_app(tmp_path/'runs',library);client=TestClient(app)
    assert client.get('/api/inputs').json()[kind]==(['table.jpg'] if kind=='image' else ['capture'])
    scene=client.post('/api/inputs/import',json={'kind':kind,'name':'table.jpg' if kind=='image' else 'capture'}).json()
    assert scene['preprocessing']['input_path']==('rgbd' if kind=='depth' else 'single_image')
    project=app.state.store.path(scene['id'])
    if kind=='depth':
        sensor=np.load(project/scene['preprocessing']['depth_registration'])
        assert sensor['depth_m'].shape==(480,640)
        assert (project/'input/raw_source.png').exists()
        np.testing.assert_array_equal(np.asarray(Image.open(project/scene['image'])),np.asarray(Image.open(project/'input/rgbd/rectified_rgb.png')))
    monkeypatch.setattr(pipeline.adapters,'status',lambda:{'sam3':{'configured':True}})
    def segment(worker,request,folder,**kwargs):
        assert request['image']==str(project/scene['image'])
        mask=np.zeros((480,640),dtype=np.uint8);mask[100:150,200:250]=255
        Image.fromarray(mask).save(folder/'mask.png')
        return {'objects':[{'name':'object','mask':'mask.png','bbox':[200,100,250,150],'score':.95}]}
    monkeypatch.setattr(pipeline.adapters,'run_worker',segment)
    job=client.post('/api/projects/'+scene['id']+'/segment',json={'mode':'instance','points':[[220,120,1]]}).json()
    assert wait(client,job)['state']=='done'
    obj=app.state.store.load(scene['id']).objects[0]
    assert obj.bbox==[200,100,250,150]
    mask=np.asarray(Image.open(project/obj.mask))
    assert mask[120,220]==255 and mask[99,200]==0


def test_input_library_rejects_escape_and_uploads_are_saved(tmp_path):
    app=create_app(tmp_path/'runs',tmp_path/'data');client=TestClient(app)
    content=io.BytesIO();Image.new('RGB',(20,20)).save(content,format='PNG')
    response=client.post('/api/projects',files={'file':('../../unsafe.png',content.getvalue(),'image/png')})
    assert response.status_code==200
    scene=response.json()
    assert (tmp_path/'data/image'/f"{scene['id']}.png").is_file()
    assert client.post('/api/inputs/import',json={'kind':'image','name':'../../unsafe.png'}).status_code==422
    shutil.copytree(EXAMPLE,tmp_path/'data/depth/bad')
    meta_path=tmp_path/'data/depth/bad/capture_metadata.json'
    meta=json.loads(meta_path.read_text());meta['d455_depth']['file']='../../outside.png';atomic_json(meta_path,meta)
    assert client.post('/api/inputs/import',json={'kind':'depth','name':'bad'}).status_code==422


@pytest.mark.parametrize('depth',[False,True])
def test_generate_preserves_selection_routes_depth_locks_edits_and_retains_failed_quality(tmp_path,monkeypatch,depth):
    app=create_app(tmp_path/'runs',tmp_path/'data');client=TestClient(app);store=app.state.store
    content=io.BytesIO();Image.new('RGB',(20,20)).save(content,format='PNG');scene=store.create(content.getvalue(),'test')
    project=store.path(scene.id);Image.new('L',(20,20),255).save(project/'mask.png')
    scene.objects=[ObjectRecord(id='o_000000000001',name='hammer',bbox=[0,0,10,10],mask='mask.png',mass=.25,annotation='hinge'),
                   ObjectRecord(id='o_000000000002',name='excluded',bbox=[10,10,20,20],mask='mask.png',selected=False)]
    if depth:
        scene.preprocessing['depth_registration']='sensor.npz';np.savez(project/'sensor.npz',depth_m=np.ones((20,20)))
    scene=store.save(scene)
    started=threading.Event();release=threading.Event();requests=[]
    monkeypatch.setattr(pipeline.adapters,'status',lambda:{'sam3d':{'configured':True}})
    def worker(kind,request,folder,**kwargs):
        requests.append(request);started.set();assert release.wait(5);return {}
    monkeypatch.setattr(pipeline.adapters,'run_worker',worker)
    monkeypatch.setattr(pipeline,'assemble',lambda *args:None)
    def finish(store,scene,refine,seconds,**kwargs):
        assert refine is True and seconds==10
        assert scene.objects[0].mass==.25 and scene.objects[0].annotation=='hinge'
        output=kwargs['output'];output.mkdir(parents=True);(output/'scene.xml').write_text('<mujoco/>')
        scene.reports={'validation':{'physics_check_pass':False},'export':{'path':str(output.relative_to(project))},'workflow_quality':{'status':'needs_review'}}
        return store.save(scene)
    monkeypatch.setattr(pipeline,'finish_scene',finish)
    route='/api/projects/'+scene.id
    assert client.post(route+'/generate',json={'revision':scene.revision-1}).status_code==409
    job=client.post(route+'/generate',json={'revision':scene.revision}).json()
    assert started.wait(2)
    assert client.put(route,json=scene.model_dump()).status_code==409
    assert client.post(route+'/generate',json={'revision':scene.revision}).status_code==409
    release.set();assert wait(client,job)['state']=='done'
    assert [row['id'] for row in requests[0]['objects']]==['o_000000000001']
    assert ('registered_depth' in requests[0]) is depth
    result=client.get(route+'/result').json()
    assert result['reports']['validation']['physics_check_pass'] is False
    assert 'cli view' in result['view_command'] and Path(result['xml']).exists()
    previous_xml=result['xml'];previous_revision=store.load(scene.id).revision
    second=client.post(route+'/generate',json={'revision':previous_revision}).json()
    assert wait(client,second)['state']=='done'
    assert client.get(route+'/result').json()['xml']!=previous_xml
    assert Path(previous_xml).exists()


def test_generation_failure_clears_stale_result_and_unlocks_project(tmp_path,monkeypatch):
    app=create_app(tmp_path/'runs',tmp_path/'data');client=TestClient(app);store=app.state.store
    content=io.BytesIO();Image.new('RGB',(20,20)).save(content,format='PNG')
    scene=store.create(content.getvalue(),'failure');project=store.path(scene.id)
    Image.new('L',(20,20),255).save(project/'mask.png')
    scene.objects=[ObjectRecord(id='o_000000000001',name='target',mask='mask.png',bbox=[0,0,20,20])]
    (project/'old').mkdir();(project/'old/scene.xml').write_text('<mujoco/>')
    scene.reports={'export':{'path':'old'},'validation':{'physics_check_pass':True}}
    scene=store.save(scene)
    monkeypatch.setattr(pipeline.adapters,'status',lambda:{'sam3d':{'configured':True}})
    def failure(*args,**kwargs):raise RuntimeError('fixture worker failed; see worker.log')
    monkeypatch.setattr(pipeline.adapters,'run_worker',failure)
    route='/api/projects/'+scene.id
    job=client.post(route+'/generate',json={'revision':scene.revision}).json()
    result=wait(client,job)
    assert result['state']=='failed' and 'worker.log' in result['error']
    assert client.get(route+'/result').json()['xml'] is None
    assert (project/'old/scene.xml').is_file()
    assert client.put(route,json=store.load(scene.id).model_dump()).status_code==200
