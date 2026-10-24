import io
from pathlib import Path

from PIL import Image
import pytest

from real2sim_workbench.adapters import worker_environment
from real2sim_workbench.models import ObjectRecord
from real2sim_workbench.store import Store
from real2sim_workbench import pipeline


def test_proxy_alias_normalization_preserves_routing_and_parent():
    env={'ALL_PROXY':'socks://user:secret@localhost:7897/',
         'https_proxy':'SOCKS://localhost:7897',
         'HTTP_PROXY':'http://localhost:8888','NO_PROXY':'localhost,127.0.0.1',
         'all_proxy':'socks5h://localhost:7898','OTHER':'untouched'}
    actual=worker_environment(env)
    assert actual['ALL_PROXY']=='socks5://user:secret@localhost:7897/'
    assert actual['https_proxy']=='socks5://localhost:7897'
    for name in ['HTTP_PROXY','NO_PROXY','all_proxy','OTHER']:assert actual[name]==env[name]
    assert env['ALL_PROXY'].startswith('socks://')


def test_resume_reuses_masks_preserves_failure_and_forwards_depth(tmp_path,monkeypatch):
    image=io.BytesIO();Image.new('RGB',(20,20)).save(image,format='PNG')
    store=Store(tmp_path/'runs');scene=store.create(image.getvalue(),'resume')
    project=store.path(scene.id)
    Image.new('L',(20,20),255).save(project/'input/mask.png')
    (project/'input/depth.npz').write_bytes(b'fixture')
    scene.preprocessing['depth_registration']='input/depth.npz'
    scene.objects=[ObjectRecord(id='o_000000000001',name='target',mask='input/mask.png',bbox=[0,0,20,20])]
    store.save(scene)
    failed=project/'jobs/reconstruct';failed.mkdir(parents=True);(failed/'worker.log').write_text('original failure')
    def worker(kind,request,job,**kwargs):
        assert kind=='sam3d'  # No repeated SAM3 segmentation.
        assert request['objects']==[{'id':scene.objects[0].id,'mask':str(project/'input/mask.png')}]
        assert request['registered_depth']==str(project/'input/depth.npz')
        assert job.name.startswith('reconstruct_retry_')
        return {}
    monkeypatch.setattr(pipeline.adapters,'run_worker',worker)
    monkeypatch.setattr(pipeline,'assemble',lambda *args:None)
    monkeypatch.setattr(pipeline,'finish_scene',lambda store,scene,*args:scene)
    result=pipeline.resume_reconstruction(project)
    assert result.objects[0].id==scene.objects[0].id
    assert (failed/'worker.log').read_text()=='original failure'
    scene.objects[0].geometry='proxy';store.save(scene)
    with pytest.raises(ValueError,match='no assembled assets'):
        pipeline.resume_reconstruction(project)
