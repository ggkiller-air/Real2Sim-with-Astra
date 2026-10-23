import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from real2sim_workbench.rgbd import deproject,project,zbuffer,register_capture,metric_support_frame


def test_depth_projection_roundtrip_and_zero_invalid():
    intr={'width':20,'height':16,'fx':30.,'fy':31.,'ppx':9.2,'ppy':7.3,'model':'distortion.inverse_brown_conrady','coeffs':[-.055,.068,-.00007,.0008,-.022]}
    depth=np.full((16,20),1.3);points=deproject(depth,intr);uv=project(points.reshape(-1,3),intr)
    yy,xx=np.indices(depth.shape)
    np.testing.assert_allclose(uv,np.c_[xx.ravel(),yy.ravel()],atol=1e-5)
    depth[0,0]=0;assert np.all(deproject(depth,intr)[0,0]==0)


def test_zbuffer_uses_closest_surface_and_rejects_behind_camera():
    K=np.array([[10,0,5],[0,10,5],[0,0,1.]])
    result=zbuffer(np.array([[0,0,2],[0,0,1],[0,0,-1]]),K,(10,10))
    assert result[5,5]==1
    assert result[0,0]==0


def test_metric_plane_scale_not_normalized_to_assumed_table_width():
    yy,xx=np.mgrid[-.2:.2:30j,-.3:.3:40j];points=np.stack([xx,yy,np.full_like(xx,1.2)],axis=-1)
    origin,basis,report=metric_support_frame(points,np.ones(xx.shape,bool))
    assert report['meters_per_model_unit']==1
    assert report['rms_m']<1e-8
    assert np.linalg.det(basis)==pytest.approx(1.)
    transformed=(points-origin)@basis
    assert np.ptp(transformed[:,:,0])==pytest.approx(.6)


def test_example_registration_uses_current_intrinsics_and_tracks_holes(tmp_path):
    folder=Path(__file__).parents[1]/'examples/rgbd_tabletop/capture'
    report=register_capture(folder,tmp_path)
    sensor=np.load(tmp_path/'registered_depth.npz')
    assert .9<report['registered_valid_fraction']<1
    assert not report['historical_robot_extrinsics_used_for_metric_layout']
    assert np.all(sensor['depth_m'][~sensor['observed_valid']]==0)
    assert np.isfinite(sensor['pointmap']).all()
    assert sensor['K'][0,0]==pytest.approx(386.4453125*640/416)
    assert np.max(sensor['hole_distance_px'])>0


def test_native_intrinsic_camera_projects_offcenter_principal_point():
    import os
    os.environ.setdefault('MUJOCO_GL','egl')
    import mujoco
    K=np.array([[594.,0,279.],[0,594.,196.],[0,0,1.]])
    # Same principalpixel convention as the exporter, tested by actual rasterization.
    xml=f'<mujoco><visual><global offwidth="640" offheight="480"/></visual><worldbody><camera name="source" sensorsize=".064 .048" resolution="640 480" focalpixel="594 594" principalpixel="{320-K[0,2]} {240-K[1,2]}"/><geom type="sphere" size=".008" pos=".05 -.02 -1"/></worldbody></mujoco>'
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    with mujoco.Renderer(m,width=640,height=480) as r:
        r.enable_segmentation_rendering();r.update_scene(d,camera='source');seg=r.render();ys,xs=np.where(seg[:,:,0]>=0)
    np.testing.assert_allclose([xs.mean(),ys.mean()],[279+594*.05-.5,196+594*.02-.5],atol=.5)


def test_selection_rejects_negative_coordinates(tmp_path):
    from real2sim_workbench.pipeline import select_rows
    Image.fromarray(np.full((20,20),255,dtype=np.uint8)).save(tmp_path/'mask.png')
    with pytest.raises(ValueError,match='outside'):
        select_rows([{'mask':'mask.png','score':1}],tmp_path,{'objects':[{'name':'bad','point':[-1,10]}]})


@pytest.mark.parametrize('metric',[False,True])
def test_build_routes_sensor_geometry_only_for_depth(tmp_path,monkeypatch,metric):
    from real2sim_workbench import pipeline
    from real2sim_workbench.store import Store
    import io
    image=io.BytesIO();Image.new('RGB',(20,20)).save(image,format='PNG')
    store=Store(tmp_path/'runs');scene=store.create(image.getvalue(),'route test')
    if metric:scene.preprocessing['depth_registration']='input/rgbd/registered_depth.npz'
    monkeypatch.setattr(pipeline,'create_input',lambda *args:(store,scene))
    requests=[]
    def worker(kind,request,job,**kwargs):
        requests.append((kind,request));return {}
    monkeypatch.setattr(pipeline.adapters,'run_worker',worker)
    monkeypatch.setattr(pipeline,'assemble',lambda *args:None)
    monkeypatch.setattr(pipeline,'finish_scene',lambda store,scene,*args:scene)
    pipeline.build(tmp_path)
    assert requests[0][0]=='sam3d'
    assert ('registered_depth' in requests[0][1]) is metric
    if metric:assert requests[0][1]['registered_depth'].endswith('registered_depth.npz')


def test_depth_quality_does_not_hide_low_overlap_or_bad_depth():
    from real2sim_workbench.pipeline import assess_depth_quality
    result=assess_depth_quality({'objects':[
        {'id':'a','name':'low overlap','median_abs_z_error_m':.001,'coverage':.2},
        {'id':'b','name':'depth offset','median_abs_z_error_m':.04,'coverage':.99},
        {'id':'c','name':'aligned','median_abs_z_error_m':.003,'coverage':.95}]})
    assert result['status']=='needs_review'
    assert {o['id'] for o in result['flagged_objects']}=={'a','b'}
