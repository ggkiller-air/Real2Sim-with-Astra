import io
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pytest
from PIL import Image,ImageDraw
from fastapi.testclient import TestClient
from real2sim_workbench import physics,rendering,pipeline
from real2sim_workbench.models import ObjectRecord
from real2sim_workbench.store import Store
from real2sim_workbench.server import create_app


def fixture_scene(tmp_path,size):
    source=io.BytesIO();Image.new('RGB',size,'#708090').save(source,format='PNG')
    store=Store(tmp_path/'runs');scene=store.create(source.getvalue(),'dimension regression');project=store.path(scene.id)
    scene.objects=[ObjectRecord(id='o_000000000001',name='block',bbox=[0,0,*size],geometry='proxy',size=[.1,.1,.1],position=[0,0,.0502],mask='mask.png')]
    scene.preprocessing['camera']={'position':[0,0,1],'xyaxes':[1,0,0,0,1,0],'fovy':60}
    w,h=size;f=h/2/np.tan(np.pi/6);half=.05*f/(1-.1002)
    mask=Image.new('L',size);ImageDraw.Draw(mask).rectangle([round(w/2-half),round(h/2-half),round(w/2+half),round(h/2+half)],fill=255);mask.save(project/'mask.png')
    return store,scene,project


@pytest.mark.parametrize('size',[(1080,1440),(1920,1080),(4000,600)])
def test_native_reference_render_large_portrait_landscape_and_historical_xml(tmp_path,size):
    store,scene,project=fixture_scene(tmp_path,size)
    xml=physics.export_scene(scene,project,project/'simulation')
    root=ET.parse(xml);global_=root.find('visual/global');rw,rh=rendering.render_size(size)
    assert int(global_.get('offwidth'))>=rw and int(global_.get('offheight'))>=rh
    # Reproduce the old export exactly: checks must also work on already saved XML.
    global_.set('offwidth','960');global_.set('offheight','720');root.write(xml)
    result=physics.silhouette_metrics(scene,project,xml)
    assert result['mean_iou']>.9
    assert max(result['render_size'])<=1600
    np.testing.assert_allclose(result['objects'][0]['rendered_centroid'],np.array(size)/2,atol=2)
    assert Image.open(project/'mask.png').size==size
    assert scene.image_size==list(size)
    # Known synthetic optical Z of the box's upper face, in metres.
    from real2sim_workbench.rgbd import depth_agreement
    scene.preprocessing['depth_registration']='sensor.npz'
    np.savez(project/'sensor.npz',depth_m=np.full((size[1],size[0]),.8998),observed_valid=np.ones((size[1],size[0]),dtype=bool))
    depth=depth_agreement(scene,project,xml)
    assert depth['objects'][0]['coverage']>.9
    assert depth['objects'][0]['median_abs_z_error_m']<.001


def test_portrait_diagnostic_sheet_preserves_aspect_ratio(tmp_path):
    store,scene,project=fixture_scene(tmp_path,(1080,1440))
    xml=physics.export_scene(scene,project,project/'simulation')
    report=physics.capture_views(scene,xml,project/'views',project/scene.image)
    reference=np.asarray(Image.open(project/'views/reference.png'))
    content=np.any(reference!=[232,236,231],axis=2);ys,xs=np.where(content)
    assert (xs.max()-xs.min()+1,ys.max()-ys.min()+1)==(360,480)
    source=np.asarray(Image.open(project/'views/source_view.png'))
    assert np.all(source[:,:140]==[232,236,231])
    assert 'aspect preserved' in report['reference_layout']


def test_table_edits_export_exactly_and_fit_is_revision_guarded(tmp_path):
    store,scene,project=fixture_scene(tmp_path,(640,480))
    scene.table.center_xy=[.3,-.2];scene.table.width=2.;scene.table.depth=1.;scene.table.calibration_source='user'
    xml=physics.export_scene(scene,project,project/'simulation')
    table=ET.parse(xml).find("worldbody/geom[@name='table']")
    assert list(map(float,table.get('pos').split()))==[.3,-.2,-.025]
    assert list(map(float,table.get('size').split()))==[1.,.5,.025]
    from real2sim_workbench.geometry import homography
    xy=homography(scene.table,scene.image_size)@np.array([320,240,1])
    np.testing.assert_allclose(xy[:2]/xy[2],[.3,-.2],atol=1e-8)
    scene=store.save(scene)
    with TestClient(create_app(store.root,input_root=tmp_path/'data')) as client:
        route='/api/projects/'+scene.id+'/table/fit'
        assert client.post(route,json={'revision':scene.revision-1}).status_code==409
        result=client.post(route,json={'revision':scene.revision}).json()
        assert result['table']['calibration_source']=='user'
        assert result['table']['center_xy']==[0.,0.]
        assert result['table']['width']==pytest.approx(.22)
        assert result['objects']==scene.model_dump()['objects']


def test_recheck_preserves_assets_and_skips_reconstruction(tmp_path,monkeypatch):
    store,scene,project=fixture_scene(tmp_path,(1080,1440))
    scene.reports={'export':{'path':'old'},'validation':{'physics_check_pass':True}}
    original=scene.model_dump()['objects']
    def finish(store,scene,**kwargs):
        assert kwargs['reuse_assets'] is True and kwargs['fit_depth_poses'] is False
        assert kwargs['optimize_layout'] is False
        assert scene.model_dump()['objects']==original
        assert 'validation' not in scene.reports and 'export' not in scene.reports
        return scene
    monkeypatch.setattr(pipeline,'finish_scene',finish)
    def forbidden(*args,**kwargs):raise AssertionError('Must not rebuild assets')
    monkeypatch.setattr(pipeline.adapters,'run_worker',forbidden)
    pipeline.recheck_existing(store,scene,project/'jobs/recheck')
