import io
import numpy as np
import pytest
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation
from real2sim_workbench import relations,physics
from real2sim_workbench.models import ObjectRecord
from real2sim_workbench.store import Store


def fixture(tmp_path):
    b=io.BytesIO();Image.new('RGB',(320,240)).save(b,format='PNG');store=Store(tmp_path);s=store.create(b.getvalue(),'shelf');p=store.path(s.id)
    # Broad intermediate shelf and a taller side wall; the latter must not be
    # selected by a blanket parent AABB/max-height heuristic.
    shelf=trimesh.creation.box(extents=[.6,.4,.04]);shelf.apply_translation([0,0,.32])
    wall=trimesh.creation.box(extents=[.03,.4,.65]);wall.apply_translation([-.285,0,.325])
    right=trimesh.creation.box(extents=[.03,.4,.32]);right.apply_translation([.285,0,.16])
    mesh=trimesh.util.concatenate([shelf,wall,right]);mesh.export(p/'shelf.obj')
    a=ObjectRecord(id='o_000000000001',name='shelf',geometry='imported',mesh='shelf.obj',bbox=[0,0,200,200],size=[1,1,1],position=[0,0,0])
    c=ObjectRecord(id='o_000000000002',name='cup',geometry='proxy',bbox=[20,20,50,60],size=[.06,.06,.1],position=[0,0,.05],mass=.17,friction=.6)
    s.objects=[a,c];return s,p


def plan(s):
    return {'objects':[{'id':o.id,'role':o.name,'support':'table' if i==0 else s.objects[0].id,'upright':False,'surface_index':-1 if i==0 else 1,'dx_m':0,'dy_m':0,'yaw_deg':0,'reason':'reference'} for i,o in enumerate(s.objects)]}


def test_intermediate_shelf_support_and_native_contact(tmp_path):
    s,p=fixture(tmp_path);proposal=relations.propose_layout(s,p,plan(s),p/'proposal')
    assert proposal.objects[1].position[2]==pytest.approx(.3906,abs=.001)
    assert proposal.objects[1].position[2]<.65
    assert proposal.objects[1].mass==.17 and proposal.objects[1].friction==.6
    # Use exact constituent box collision meshes to preserve the open space.
    for i,(size,center) in enumerate([([.6,.4,.04],[0,0,.32]),([.03,.4,.65],[-.285,0,.325]),([.03,.4,.32],[.285,0,.16])]):
        m=trimesh.creation.box(extents=size);m.apply_translation(center);m.export(p/f'part{i}.obj')
    proposal.objects[0].collision_meshes=['part0.obj','part1.obj','part2.obj']
    physics.settle(proposal,p,p/'settle')
    xml=physics.export_scene(proposal,p,p/'checked');result=physics.validate_scene(proposal,xml,p/'checked',seconds=1)
    assert result['physics_check_pass']
    assert all(o['expected_support_contact'] for o in result['objects'])


def test_leveling_recovers_tilt_without_changing_geometry(tmp_path):
    s,p=fixture(tmp_path);a=s.objects[0];q=Rotation.from_euler('x',12,degrees=True).as_quat();a.orientation=q[[3,0,1,2]].tolist()
    corrected=relations.leveled_rotation(a,p)
    np.testing.assert_allclose(corrected.apply([0,0,1]),[0,0,1],atol=1e-6)


@pytest.mark.parametrize('kind',['cycle','missing','duplicate','nan','locked','bad_surface'])
def test_reject_untrusted_relation_plans(tmp_path,kind):
    s,p=fixture(tmp_path);r=plan(s)
    if kind=='cycle':r['objects'][0]['support']=s.objects[1].id
    if kind=='missing':r['objects'][1]['support']='unknown'
    if kind=='duplicate':r['objects'][1]['id']=s.objects[0].id
    if kind=='nan':r['objects'][0]['dx_m']=float('nan')
    if kind=='locked':s.objects[1].locked=True
    if kind=='bad_surface':r['objects'][1]['surface_index']=999
    with pytest.raises(ValueError):relations.propose_layout(s,p,r,p/'proposal')


def test_depth_regression_rejects_missing_or_worse_measurement():
    before={'objects':[{'id':'x','median_abs_z_error_m':.01,'coverage':.9}]}
    assert relations.depth_not_worse(before,before)
    for error,coverage in [(None,.9),(.02,.9),(.01,.7)]:
        after={'objects':[{'id':'x','median_abs_z_error_m':error,'coverage':coverage}]}
        assert not relations.depth_not_worse(before,after)


def test_failed_joint_hypothesis_is_not_published(tmp_path,monkeypatch):
    s,p=fixture(tmp_path);original=s.model_dump()['objects'];calls=[]
    monkeypatch.setattr(relations,'plan_relations',lambda *args:plan(s))
    monkeypatch.setattr(physics,'export_scene',lambda *args:p/'unused.xml')
    monkeypatch.setattr(physics,'silhouette_metrics',lambda *args:{'mean_iou':.7})
    monkeypatch.setattr(physics,'capture_views',lambda *args,**kwargs:None)
    monkeypatch.setattr(physics,'settle',lambda *args,**kwargs:{})
    def check(*args):
        calls.append(1)
        return {'physics_check_pass':False,'finite':True,'warnings':[]}
    monkeypatch.setattr(physics,'validate_scene',check)
    result=relations.optimize_relations(s,p,p/'loop',max_rounds=2)
    assert len(result['trials'])==2 and not result['accepted']
    assert s.model_dump()['objects']==original
    assert len(calls)==8  # raw, settled, two bounded settling retries per round
    assert result['status']=='unresolved_candidate_retained'
