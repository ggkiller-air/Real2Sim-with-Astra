"""Headless entry for the two input routes; shared asset and MuJoCo stages."""
import hashlib
import json
from pathlib import Path
import shutil
import uuid
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from .store import Store,atomic_json
from .models import ObjectRecord
from . import adapters,physics
from .segmentation import AUTO_LABELS,load_mask
from .reconstruction import assemble


def select_rows(rows,folder,selection=None):
    if selection:
        selected=[]
        for target in selection['objects']:
            x,y=map(int,target['point']);matches=[]
            for row in rows:
                mask=load_mask(folder/row['mask'])
                if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
                    raise ValueError('Selection point outside the source image')
                if mask[y,x]: matches.append(row)
            if not matches:raise ValueError('SAM3 has no candidate at '+target['name']+'; refine the selection first')
            selected.append({**max(matches,key=lambda r:r['score']),'name':target['name']})
        if len({r['mask'] for r in selected})!=len(selected):raise ValueError('Two selection points identify the same instance')
        return selected
    masks=[load_mask(folder/r['mask']) for r in rows]
    return [r for i,r in enumerate(rows) if not any((masks[i]&m).sum()/max(1,masks[i].sum())>.9 and masks[i].sum()<m.sum()*.6 for j,m in enumerate(masks) if i!=j)]


def create_input(source,root,selection=None):
    from .rgbd import register_capture,warp_mask
    source=Path(source).resolve();store=Store(root)
    is_depth=source.is_dir();raw_image=source/'image_external_rgb.jpg' if is_depth else source
    scene=store.create(raw_image.read_bytes(),source.stem+(' · RGB-D' if is_depth else ' · 单图'));project=store.path(scene.id)
    print('PROJECT '+str(project),flush=True)
    registration=None
    if is_depth:
        register_capture(source,project/'input/rgbd');registration=np.load(project/'input/rgbd/registered_depth.npz',allow_pickle=False)
        shutil.copy2(project/scene.image,project/'input/raw_source.png');shutil.copy2(project/'input/rgbd/rectified_rgb.png',project/scene.image)
        scene.image_sha256=hashlib.sha256((project/scene.image).read_bytes()).hexdigest()
        scene.preprocessing.update(input_path='rgbd',depth_registration='input/rgbd/registered_depth.npz',capture_manifest='input/rgbd/registration.json',reference_views=[])
        metadata=json.loads((source/'capture_metadata.json').read_text())
        for name,view in metadata['rgb'].items():
            if name=='image_external':continue
            path=source/view['file'];target=project/'input'/path.name;shutil.copy2(path,target)
            scene.preprocessing['reference_views'].append({'image':str(target.relative_to(project)),'metric_fusion':False,'calibration_status':'historical mounting unverified'})
    else:scene.preprocessing['input_path']='single_image'
    folder=project/'jobs/segmentation'
    result=adapters.run_worker('sam3',{'mode':'auto','labels':AUTO_LABELS+['carrot','apple'],'image':str((project/('input/raw_source.png' if is_depth else scene.image)).resolve())},folder)
    for key,value in result.get('background',{}).items():
        if key.endswith('mask'):
            path=folder/value
            if registration is not None:
                rectified=folder/(key+'_rectified.png');warp_mask(path,registration,rectified);path=rectified
            scene.preprocessing[key]=str(path.relative_to(project))
    for row in select_rows(result['objects'],folder,selection):
        path=folder/row['mask']
        if registration is not None:
            rectified=folder/(Path(row['mask']).stem+'_rectified.png');mask=warp_mask(path,registration,rectified);path=rectified
        else:mask=load_mask(path)
        yy,xx=np.where(mask)
        scene.objects.append(ObjectRecord(id='o_'+uuid.uuid4().hex[:12],name=row['name'],mask=str(path.relative_to(project)),score=row['score'],bbox=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]))
    scene=store.save(scene);return store,scene


def adopt_candidates(scene,project,report):
    """Use candidate centre/scale consistently; keep mass/friction untouched."""
    from .rgbd import fit_mesh_to_observed
    import trimesh
    project=Path(project);changes=[]
    for row in report['objects']:
        if row['status']=='kept':continue
        obj=next(o for o in scene.objects if o.id==row['id']);old=obj.model_dump();candidate=row['candidate'];q=physics.body_quaternion(obj);rotation=Rotation.from_quat(q[[1,2,3,0]])
        obj.position=(np.array(obj.position)+rotation.apply(candidate['asset_center_offset'])).tolist();obj.size=candidate['size']
        for key in ['mesh','visual_texture','collision_meshes']:setattr(obj,key,candidate[key])
        fit=None
        if scene.preprocessing.get('input_path')=='rgbd':
            observed=np.load(project/'jobs'/scene.reports['reconstruction']['job']/(obj.id+'_observed.npz'),allow_pickle=False)['points']
            mesh=trimesh.load(project/obj.mesh,force='mesh',process=False);mesh.vertices=rotation.apply(mesh.vertices*obj.size)+obj.position
            center=mesh.bounds.mean(axis=0);fit=fit_mesh_to_observed(mesh,observed,limit=.025)
            if fit['accepted']:
                r=Rotation.from_rotvec(fit['rotation_vector']);scale=fit['scale'];obj.position=(r.apply(np.array(obj.position)-center)*scale+center+fit['translation_m']).tolist();obj.size=(np.array(obj.size)*scale).tolist();q=(r*rotation).as_quat();obj.orientation=q[[3,0,1,2]].tolist();obj.yaw=0
            obj.position[2]+=-float(mesh.bounds[0,2])+.0002
        obj.reconstruction['asset_refinement']={'model':'gpt-6-astra','template':row['recipe']['template'],'source_asset':old,'candidate_manifest':report['path']+'/'+obj.id+'/asset.json','post_adoption_depth_fit':fit}
        changes.append({'id':obj.id,'template':row['recipe']['template'],'depth_fit':fit})
    scene.reports['asset_adoption']={'changes':changes,'mass_friction_unchanged':True}
    return changes


def finish_scene(store,scene,refine=False,seconds=10,fit_depth_poses=True):
    from .asset_refinement import refine_assets
    project=store.path(scene.id)
    if refine:
        targets=[o.id for o in scene.objects if o.selected and (o.name=='hammer' or 'button box' in o.name)]
        if targets:
            report=refine_assets(scene,project,project/'jobs/asset_refinement',targets)
            adopt_candidates(scene,project,report);store.save(scene)
    if fit_depth_poses and scene.preprocessing.get('input_path')=='rgbd':
        from .depth_pose import fit_supported_poses
        scene.reports['supported_poses']=fit_supported_poses(scene,project,project/'jobs/depth_supported_poses')
    before=physics.export_scene(scene,project,project/'jobs/before_physics')
    scene.reports['initial_silhouette']=physics.silhouette_metrics(scene,project,before)
    if scene.preprocessing.get('input_path')=='rgbd':
        from .rgbd import depth_agreement
        scene.reports['depth_before_physics']=depth_agreement(scene,project,before)
    scene.reports['optimization']=physics.optimize_contacts(scene,project,project/'jobs/contacts')
    scene.reports['settling']=physics.settle(scene,project,project/'jobs/settling')
    output=project/'mujoco';xml=physics.export_scene(scene,project,output)
    scene.reports['validation']=physics.validate_scene(scene,xml,output,seconds)
    # Generated curved objects may still rock after the initial 3 s. Retain each
    # failed trial and allow bounded extra settling; thresholds never change.
    scene.reports['settling_retries']=[]
    for attempt in range(2):
        validation=scene.reports['validation']
        if validation['physics_check_pass'] or not validation['finite'] or validation['warnings']:
            break
        retry=project/'jobs'/f'extended_settling_{attempt}'
        atomic_json(retry/'failed_validation.json',validation)
        atomic_json(retry/'before_scene.json',scene.model_dump())
        settling=physics.settle(scene,project,retry,seconds=8)
        scene.reports['settling_retries'].append(settling)
        xml=physics.export_scene(scene,project,output)
        scene.reports['validation']=physics.validate_scene(scene,xml,output,seconds)
    scene.reports['silhouette']=physics.silhouette_metrics(scene,project,xml)
    if scene.preprocessing.get('input_path')=='rgbd':
        scene.reports['depth_after_physics']=depth_agreement(scene,project,xml)
        atomic_json(output/'depth_agreement.json',scene.reports['depth_after_physics'])
        scene.reports['depth_quality']=assess_depth_quality(scene.reports['depth_after_physics'])
        atomic_json(output/'depth_quality.json',scene.reports['depth_quality'])
    views=physics.capture_views(scene,xml,output/'evidence/views',reference=project/scene.image)
    scene.reports['views']={**views,'path':'mujoco/evidence/views'};scene.reports['export']={'path':'mujoco','source_revision':scene.revision}
    scene=store.save(scene)
    atomic_json(output/'scene_record.json',scene.model_dump())
    print(json.dumps({'project':str(project),'xml':str(xml),'physics_pass':scene.reports['validation']['physics_check_pass'],'mean_iou':scene.reports['silhouette'].get('mean_iou')},ensure_ascii=False),flush=True)
    return scene


def build(source,root='runs',selection=None,refine=False,seconds=10):
    store,scene=create_input(source,root,selection)
    project=store.path(scene.id);job=project/'jobs/reconstruct'
    request={'image':str((project/scene.image).resolve()),'objects':[{'id':o.id,'mask':str((project/o.mask).resolve())} for o in scene.objects],'seed':42}
    if scene.preprocessing.get('depth_registration'):request['registered_depth']=str((project/scene.preprocessing['depth_registration']).resolve())
    result=adapters.run_worker('sam3d',request,job,timeout=3600);assemble(scene,project,job,result);store.save(scene)
    return finish_scene(store,scene,refine,seconds)


def assess_depth_quality(report):
    """Heuristic review flags, independent from the dynamics pass/fail."""
    flagged=[]
    for row in report['objects']:
        error=row['median_abs_z_error_m']
        if error is None or error>.015 or row['coverage']<.8:
            flagged.append({'id':row['id'],'name':row['name'],'median_abs_z_error_m':error,'coverage':row['coverage']})
    return {'status':'needs_review' if flagged else 'within_review_thresholds',
            'flagged_objects':flagged,'thresholds':{'median_depth_error_m':.015,'minimum_visible_coverage':.8},
            'note':'Heuristic review criteria, not independent metrology. Physical stability does not establish reconstruction fidelity.'}
