import io
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from real2sim_workbench.asset_refinement import clean_mesh, PARAMS, refine_assets
from real2sim_workbench.models import ObjectRecord
from real2sim_workbench.store import Store
from real2sim_workbench.pipeline import adopt_candidates, assess_workflow_quality


def test_cleanup_preserves_components_color_and_bounds_displacement():
    # Separate small parts must survive; no global largest-component filtering.
    a=trimesh.creation.icosphere(subdivisions=2,radius=.05)
    b=trimesh.creation.icosphere(subdivisions=1,radius=.005);b.apply_translation([.12,0,0])
    source=trimesh.util.concatenate([a,b])
    source.visual.vertex_colors=np.tile([230,90,40,255],(len(source.vertices),1))
    before=source.vertices.copy()
    result,report=clean_mesh(source)
    assert len(result.split())==2
    assert result.is_watertight
    assert report['max_displacement_m']<=report['displacement_limit_m']+1e-12
    assert len(result.faces)==len(source.faces)
    np.testing.assert_array_equal(source.vertices,before)
    assert np.all(result.visual.vertex_colors[:,:3]==[230,90,40])


def test_all_asset_review_generates_cleanup_and_records_unresolved_keep(tmp_path):
    image=io.BytesIO();Image.new('RGB',(100,100),'orange').save(image,format='PNG')
    store=Store(tmp_path/'runs');scene=store.create(image.getvalue(),'assets')
    project=store.path(scene.id)
    mesh=trimesh.creation.icosphere(subdivisions=1,radius=.5)
    mesh.visual.vertex_colors=np.tile([200,60,30,255],(len(mesh.vertices),1))
    mesh.export(project/'original.obj');Image.new('RGB',(8,8),'orange').save(project/'texture.png')
    for oid in ['o_000000000001','o_000000000002']:
        scene.objects.append(ObjectRecord(id=oid,name='unusual object',bbox=[0,0,100,100],geometry='sam3d',
            mesh='original.obj',visual_texture='texture.png',size=[.1,.1,.1],position=[0,0,.05]))
    original=scene.model_dump()['objects']
    recipes=[]
    for obj,action in zip(scene.objects,['mesh_cleanup','keep']):
        recipes.append({'id':obj.id,'template':action,'review_status':'needs_repair','assessment':'Surface noise or missing detail',
                        'button_color':'none','parameters':{k:(lo+hi)/2 for k,(lo,hi) in PARAMS.items()},'limitations':['Unseen shape uncertain']})
    plan=project/'fixture_plan.json';plan.write_text(json.dumps({'assets':recipes}))
    report=refine_assets(scene,project,project/'jobs/refine',reuse_plan=plan)
    assert not report['actual_model_called']
    assert scene.model_dump()['objects']==original
    assert report['objects'][0]['cleanup']['semantic_repair'] is False
    assert (project/report['objects'][0]['candidate']['mesh']).exists()
    adopt_candidates(scene,project,report)
    assert len(scene.reports['asset_adoption']['reviewed_ids'])==2
    assert scene.objects[1].mesh=='original.obj'
    assert scene.objects[0].mass==scene.objects[1].mass==.1
    quality=assess_workflow_quality(scene)
    assert quality['all_selected_assets_reviewed']
    assert quality['unresolved_kept_ids']==[scene.objects[1].id]
    assert quality['status']=='needs_review'


@pytest.mark.parametrize('args,expected',[([],True),(['--skip-asset-review'],False),(['--refine-assets'],True)])
def test_cli_defaults_to_all_asset_review(monkeypatch,args,expected):
    from real2sim_workbench import cli,pipeline
    from types import SimpleNamespace
    called=[]
    def build(source,root,selection,refine,seconds):
        called.append(refine)
        return SimpleNamespace(reports={'validation':{'physics_check_pass':True}})
    monkeypatch.setattr(pipeline,'build',build)
    monkeypatch.setattr('sys.argv',['real2sim','build','example.jpg',*args])
    cli.main()
    assert called==[expected]


def test_finish_reviews_all_selected_assets_and_keeps_final_suggestions_readonly(tmp_path,monkeypatch):
    from real2sim_workbench import pipeline,asset_refinement,review,relations
    image=io.BytesIO();Image.new('RGB',(20,20)).save(image,format='PNG')
    store=Store(tmp_path/'runs');scene=store.create(image.getvalue(),'full workflow')
    scene.preprocessing['input_path']='single_image'
    scene.objects=[ObjectRecord(id='o_000000000001',name='apple',bbox=[0,0,10,10]),
                   ObjectRecord(id='o_000000000002',name='brush',bbox=[10,0,20,10]),
                   ObjectRecord(id='o_000000000003',name='excluded',bbox=[0,10,10,20],selected=False)]
    expected=[o.id for o in scene.objects if o.selected]
    def refine(scene,project,folder,ids):
        assert ids==expected
        return {'model':'gpt-6-astra','actual_model_called':False,'path':'jobs/fixture',
                'objects':[{'id':oid,'status':'kept','review_status':'acceptable',
                            'recipe':{'template':'keep','assessment':'fixture','limitations':[]}} for oid in ids]}
    monkeypatch.setattr(asset_refinement,'refine_assets',refine)
    def export(scene,project,folder):
        folder.mkdir(parents=True,exist_ok=True);p=folder/'scene.xml';p.write_text('<mujoco/>');return p
    monkeypatch.setattr(pipeline.physics,'export_scene',export)
    monkeypatch.setattr(pipeline.physics,'silhouette_metrics',lambda *args:{'available':True,'mean_iou':.8,'objects':[{'id':i,'iou':.8} for i in expected]})
    monkeypatch.setattr(pipeline.physics,'optimize_contacts',lambda *args:{})
    monkeypatch.setattr(pipeline.physics,'settle',lambda *args,**kwargs:{})
    # This orchestration unit test must never make a real, billed model call.
    monkeypatch.setattr(relations,'optimize_relations',lambda *args,**kwargs:{'accepted':False})
    monkeypatch.setattr(pipeline.physics,'validate_scene',lambda *args:{'physics_check_pass':True,'finite':True,'warnings':[]})
    monkeypatch.setattr(pipeline.physics,'capture_views',lambda *args,**kwargs:{'views':[]})
    def final_review(scene,project,views,folder):
        assert all(o.reconstruction.get('asset_review') for o in scene.objects if o.selected)
        return {'visual_match':'partial','adjustments':[{'id':expected[0],'dx_m':.1}],'pose_search_ids':[]}
    monkeypatch.setattr(review,'astra_review',final_review)
    positions=[o.position.copy() for o in scene.objects]
    result=pipeline.finish_scene(store,scene)
    assert [o.position for o in result.objects]==positions
    assert result.reports['workflow']['asset_review_policy']=='all_selected'
    assert result.reports['workflow_quality']['all_selected_assets_reviewed']
    assert 'astra_visual_review_required' in result.reports['workflow_quality']['flags']
    assert not result.reports['final_visual_review']['suggestions_applied']
    assert (store.path(scene.id)/'mujoco/workflow.json').exists()
