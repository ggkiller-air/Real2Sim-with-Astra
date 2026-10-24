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


def prepare_input(source,root):
    from .rgbd import register_capture
    source=Path(source).resolve();store=Store(root)
    is_depth=source.is_dir();raw_image=source/'image_external_rgb.jpg' if is_depth else source
    scene=store.create(raw_image.read_bytes(),source.stem+(' · RGB-D' if is_depth else ' · 单图'));project=store.path(scene.id)
    print('PROJECT '+str(project),flush=True)
    if is_depth:
        register_capture(source,project/'input/rgbd')
        shutil.copy2(project/scene.image,project/'input/raw_source.png');shutil.copy2(project/'input/rgbd/rectified_rgb.png',project/scene.image)
        scene.image_sha256=hashlib.sha256((project/scene.image).read_bytes()).hexdigest()
        scene.preprocessing.update(input_path='rgbd',depth_registration='input/rgbd/registered_depth.npz',capture_manifest='input/rgbd/registration.json',reference_views=[])
        metadata=json.loads((source/'capture_metadata.json').read_text())
        for name,view in metadata['rgb'].items():
            if name=='image_external':continue
            path=source/view['file'];target=project/'input'/path.name;shutil.copy2(path,target)
            scene.preprocessing['reference_views'].append({'image':str(target.relative_to(project)),'metric_fusion':False,'calibration_status':'historical mounting unverified'})
    else:scene.preprocessing['input_path']='single_image'
    scene=store.save(scene)
    return store,scene


def create_input(source,root,selection=None):
    from .rgbd import warp_mask
    store,scene=prepare_input(source,root);project=store.path(scene.id)
    is_depth=scene.preprocessing.get('input_path')=='rgbd'
    registration=np.load(project/scene.preprocessing['depth_registration'],allow_pickle=False) if is_depth else None
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
        obj=next(o for o in scene.objects if o.id==row['id'])
        obj.reconstruction['asset_review']={'model':report['model'],'actual_model_called':report['actual_model_called'],
            'decision':row['recipe']['template'],'source_status':row.get('review_status','unassessed'),
            'assessment':row['recipe'].get('assessment',''),'limitations':row['recipe'].get('limitations',[]),
            'candidate_quality':'unverified' if row['status']!='kept' else 'unchanged',
            'report':report['path']+'/report.json'}
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
    scene.reports['asset_adoption']={'changes':changes,'reviewed_ids':[r['id'] for r in report['objects']],'candidate_status':'pending_scene_checks','mass_friction_unchanged':True}
    return changes


def finish_scene(store,scene,refine=True,seconds=10,fit_depth_poses=True,*,stage_root=None,output=None,progress=None,reuse_assets=False,optimize_layout=True):
    from .asset_refinement import refine_assets
    project=store.path(scene.id)
    stage_root=Path(stage_root) if stage_root else project/'jobs'/('workflow_'+uuid.uuid4().hex[:8])
    output=Path(output) if output else project/'mujoco'
    output_rel=str(output.relative_to(project))
    def announce(stage):
        print(stage,flush=True)
        if progress:progress(stage)
    scene.reports['workflow']={'version':'rigid_v0.4','input_path':scene.preprocessing.get('input_path','single_image'),
        'asset_review_policy':('preserved_existing_assets' if reuse_assets else 'all_selected') if refine else 'explicit_skip_ablation',
        'special_structures':'annotations_only','ui_pipeline_updated':True}
    if refine and not reuse_assets:
        announce('Astra 正在逐个检查和优化资产')
        targets=[o.id for o in scene.objects if o.selected]
        if targets:
            report=refine_assets(scene,project,stage_root/'asset_refinement',targets)
            adopt_candidates(scene,project,report);scene=store.save(scene)
    if fit_depth_poses and scene.preprocessing.get('input_path')=='rgbd':
        announce('利用深度测量优化三维摆放')
        from .depth_pose import fit_supported_poses
        scene.reports['supported_poses']=fit_supported_poses(scene,project,stage_root/'depth_supported_poses')
    announce('优化接触与支撑，运行物理检查' if optimize_layout else '保留手动摆放，运行物理检查')
    before=physics.export_scene(scene,project,stage_root/'before_physics')
    scene.reports['initial_silhouette']=physics.silhouette_metrics(scene,project,before)
    if scene.preprocessing.get('input_path')=='rgbd':
        from .rgbd import depth_agreement
        scene.reports['depth_before_physics']=depth_agreement(scene,project,before)
    relational_accepted=False
    if optimize_layout and refine:
        from .relations import optimize_relations
        relation_report=optimize_relations(scene,project,stage_root/'relations',seconds,progress=announce)
        relational_accepted=relation_report['accepted']
    if optimize_layout and not relational_accepted:
        scene.reports['optimization']=physics.optimize_contacts(scene,project,stage_root/'contacts')
        scene.reports['settling']=physics.settle(scene,project,stage_root/'settling')
    else:
        scene.reports['optimization']={'method':'astra_relational_layout' if relational_accepted else 'manual_pose_preserved','changed':relational_accepted}
        scene.reports.pop('settling',None)
    xml=physics.export_scene(scene,project,output)
    scene.reports['validation']=physics.validate_scene(scene,xml,output,seconds)
    # Generated curved objects may still rock after the initial 3 s. Retain each
    # failed trial and allow bounded extra settling; thresholds never change.
    scene.reports['settling_retries']=[]
    for attempt in range(2 if optimize_layout else 0):
        validation=scene.reports['validation']
        if validation['physics_check_pass'] or not validation['finite'] or validation['warnings']:
            break
        retry=stage_root/f'extended_settling_{attempt}'
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
    announce('采集 MuJoCo 多视角并计算视觉质量')
    views=physics.capture_views(scene,xml,output/'evidence/views',reference=project/scene.image)
    if refine:
        announce('Astra 正在进行最终多视角检查')
        from .review import astra_review
        final_review=astra_review(scene,project,output/'evidence/views',output/'evidence/astra')
        scene.reports['final_visual_review']={'model':'gpt-6-astra','actual_model_called':True,
            'review':final_review,'suggestions_applied':False}
        atomic_json(output/'evidence/astra/review.json',scene.reports['final_visual_review'])
    else:
        scene.reports.pop('final_visual_review',None)
    scene.reports['workflow_quality']=assess_workflow_quality(scene)
    atomic_json(output/'workflow.json',{'workflow':scene.reports['workflow'],'quality':scene.reports['workflow_quality']})
    scene.reports['views']={**views,'path':output_rel+'/evidence/views'};scene.reports['export']={'path':output_rel,'source_revision':scene.revision}
    scene=store.save(scene)
    atomic_json(output/'scene_record.json',scene.model_dump())
    print(json.dumps({'project':str(project),'xml':str(xml),'physics_pass':scene.reports['validation']['physics_check_pass'],'mean_iou':scene.reports['silhouette'].get('mean_iou')},ensure_ascii=False),flush=True)
    return scene


def generate_selected(store,scene,job,refine=True,seconds=10,*,output=None,progress=None):
    """Shared GUI/CLI path after selection; caller owns the project job lock."""
    project=store.path(scene.id);job=Path(job)
    selected=[o for o in scene.objects if o.selected]
    if not selected:raise ValueError('请至少保留一个物体')
    objects=[{'id':o.id,'mask':str(store.asset(scene.id,o.mask))} for o in selected if o.mask]
    if len(objects)!=len(selected):raise ValueError('所有选中物体都需要 SAM3 分割掩码')
    # Keep previous runs on disk, but never show old checks as current evidence.
    scene.reports={};scene=store.save(scene)
    print('SAM 3D 正在重建选中的资产',flush=True)
    if progress:progress('SAM 3D 正在重建选中的资产')
    request={'image':str(store.asset(scene.id,scene.image)),'objects':objects,'seed':42}
    if scene.preprocessing.get('depth_registration'):
        request['registered_depth']=str(store.asset(scene.id,scene.preprocessing['depth_registration']))
    result=adapters.run_worker('sam3d',request,job,timeout=3600)
    print('正在处理纹理与碰撞几何',flush=True)
    if progress:progress('正在处理纹理与碰撞几何')
    assemble(scene,project,job,result);scene=store.save(scene)
    return finish_scene(store,scene,refine,seconds,stage_root=job/'stages',output=output,progress=progress)


def build(source,root='runs',selection=None,refine=True,seconds=10):
    store,scene=create_input(source,root,selection)
    return generate_selected(store,scene,store.path(scene.id)/'jobs/reconstruct',refine,seconds)


def resume_reconstruction(project,refine=True,seconds=10):
    """Resume a failed draft reconstruction, retaining selection and old logs.

    Intentionally not a general scene-edit/resume command: once assets have
    been assembled, reject this route rather than overwrite reviewed work.
    """
    project=Path(project).resolve();store=Store(project.parent);scene=store.load(project.name)
    selected=[o for o in scene.objects if o.selected]
    if not selected or any(o.geometry!='none' or o.mesh for o in selected):
        raise ValueError('Resume requires selected masks and no assembled assets; later-stage recovery is not supported here')
    for obj in selected:store.asset(scene.id,obj.mask)
    store.asset(scene.id,scene.image)
    job=project/'jobs'/('reconstruct_retry_'+uuid.uuid4().hex[:8])
    request={'image':str((project/scene.image).resolve()),'objects':[{'id':o.id,'mask':str((project/o.mask).resolve())} for o in selected],'seed':42}
    if scene.preprocessing.get('depth_registration'):
        request['registered_depth']=str(store.asset(scene.id,scene.preprocessing['depth_registration']))
    print('PROJECT '+str(project),flush=True)
    result=adapters.run_worker('sam3d',request,job,timeout=3600)
    assemble(scene,project,job,result);scene=store.save(scene)
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


def assess_workflow_quality(scene):
    """Keep asset review, visual agreement and dynamics as separate evidence."""
    selected=[o for o in scene.objects if o.selected]
    reviews={o.id:o.reconstruction.get('asset_review') for o in selected}
    missing=[oid for oid,r in reviews.items() if not r]
    unresolved=[oid for oid,r in reviews.items() if r and r['decision']=='keep' and r['source_status']!='acceptable']
    flags=[]
    if missing:flags.append('assets_without_astra_review')
    if unresolved:flags.append('assets_kept_with_unresolved_defects')
    if scene.reports.get('depth_quality',{}).get('status')=='needs_review':flags.append('depth_mismatch')
    silhouette=scene.reports.get('silhouette',{})
    if not silhouette.get('available') or any(r['iou']<.6 for r in silhouette.get('objects',[])):flags.append('silhouette_review_required')
    visual_review=scene.reports.get('final_visual_review',{}).get('review',{})
    if visual_review.get('visual_match') in ('partial','poor'):flags.append('astra_visual_review_required')
    if not visual_review:flags.append('final_astra_review_missing')
    physical=bool(scene.reports.get('validation',{}).get('physics_check_pass'))
    if not physical:flags.append('physics_failed')
    if scene.reports.get('relations',{}).get('status')=='unresolved_candidate_retained':flags.append('automatic_layout_unresolved')
    return {'status':'needs_review' if flags else 'checks_passed_not_ground_truth',
            'physics_check_pass':physical,'all_selected_assets_reviewed':not missing,
            'unreviewed_ids':missing,'unresolved_kept_ids':unresolved,'flags':flags,
            'asset_candidates_independently_validated':False,
            'note':'Astra proposals and aggregate scene checks are not per-asset fidelity qualification.'}


def recheck_existing(store,scene,job,seconds=10,progress=None,*,auto_layout=False):
    """Re-export edited table/assets without repeating model reconstruction."""
    if not any(o.selected for o in scene.objects) or any(o.selected and o.geometry=='none' for o in scene.objects):
        raise ValueError('请先生成三维资产，再调整桌面和复检')
    job=Path(job)
    # Preserve original source records and failed job evidence; fresh evaluation only.
    for key in ('validation','export','views','final_visual_review','workflow_quality','silhouette','depth_after_physics','depth_quality'):
        scene.reports.pop(key,None)
    scene=store.save(scene)
    return finish_scene(store,scene,seconds=seconds,fit_depth_poses=False,
                        stage_root=job/'stages',output=job/'simulation',progress=progress,reuse_assets=True,optimize_layout=auto_layout)


def generate_target(store,scene,job,target='mujoco',seconds=10,progress=None):
    """One-click complete workflow; a second backend can reuse unchanged reviewed assets."""
    if target not in ('mujoco','isaac'):raise ValueError('Unknown target')
    job=Path(job)
    if target=='isaac':
        from .isaac import runtime_python,scene_fingerprint,export_isaac
        runtime_python()  # fail before expensive model work if not configured
        exported=scene.reports.get('export')
        reusable=False
        if exported and scene.reports.get('final_visual_review'):
            path=store.path(scene.id)/exported['path']/'scene_record.json'
            if path.is_file():
                from .models import SceneRecord
                saved=SceneRecord.model_validate_json(path.read_text())
                reusable=scene_fingerprint(saved)==scene_fingerprint(scene)
        if not reusable:scene=generate_selected(store,scene,job,seconds=seconds,output=job/'simulation',progress=progress)
        elif progress:progress('复用已完成精修和摆放检查的场景')
        return export_isaac(store,scene,job/'isaac',seconds,progress)
    return generate_selected(store,scene,job,seconds=seconds,output=job/'simulation',progress=progress)
