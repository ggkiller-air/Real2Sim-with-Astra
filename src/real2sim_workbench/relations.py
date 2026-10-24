"""Bounded Astra support graph proposals, executed and checked in native MuJoCo.

Geometry, masses and friction are immutable. A failed hypothesis is retained as
an inspectable candidate, never silently promoted to the delivered scene.
"""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from . import physics
from .models import SceneRecord
from .geometry import support_order
from .store import atomic_json


def mesh_for(obj, project, collision=False):
    paths=(obj.collision_meshes or [obj.mesh]) if collision else [obj.mesh]
    if not obj.mesh:
        return trimesh.creation.box(extents=obj.size)
    meshes=[trimesh.load(Path(project)/p,force='mesh',process=False) for p in paths]
    mesh=trimesh.util.concatenate(meshes);mesh.vertices*=obj.size
    return mesh


def rotation_for(obj):
    return Rotation.from_quat(physics.body_quaternion(obj)[[1,2,3,0]])


def leveled_rotation(obj, project):
    """Closest OBB principal up axis; bounded tilt, never assumes upright for all objects."""
    mesh=mesh_for(obj,project);r=rotation_for(obj)
    axes=r.as_matrix()@mesh.bounding_box_oriented.primitive.transform[:3,:3]
    normal=axes[:,np.argmax(abs(axes[2]))]
    if normal[2]<0:normal=-normal
    angle=math.acos(float(np.clip(normal[2],-1,1)))
    if angle>math.radians(30):raise ValueError('Upright correction exceeds 30 degrees')
    cross=np.cross(normal,[0,0,1]);norm=np.linalg.norm(cross)
    correction=Rotation.from_rotvec(cross/norm*angle) if norm>1e-10 else Rotation.identity()
    return correction*r


def surfaces(obj,project,level=False):
    """Broad upward triangle patches, including intermediate shelves, not just max Z."""
    mesh=mesh_for(obj,project);r=leveled_rotation(obj,project) if level else rotation_for(obj)
    mesh.vertices=r.apply(mesh.vertices)
    area=mesh.area_faces;normals=mesh.face_normals;centers=mesh.triangles_center
    good=np.flatnonzero((normals[:,2]>.94)&(area>1e-10))
    # Adaptive 8 mm patches at this object's scale, with near-planar faces grouped.
    tolerance=max(.003,min(.012,float(max(obj.size))*.012))
    clusters=[]
    for i in good[np.argsort(centers[good,2])]:
        if not clusters or centers[i,2]-centers[clusters[-1][0],2]>tolerance:clusters.append([])
        clusters[-1].append(int(i))
    result=[]
    for indices in clusters:
        ids=np.asarray(indices);total=float(area[ids].sum())
        if total<max(.0001,mesh.area*.006):continue
        vertices=mesh.triangles[ids].reshape(-1,3)
        result.append({'index':len(result),'z_local':float(np.average(centers[ids,2],weights=area[ids])),
                       'area_m2':total,'xy_bounds':vertices[:,:2].min(0).tolist()+vertices[:,:2].max(0).tolist()})
    return result


SCHEMA={'type':'object','properties':{
 'assessment':{'type':'string'},'complete':{'type':'boolean'},
 'objects':{'type':'array','items':{'type':'object','properties':{
   'id':{'type':'string'},'role':{'type':'string'},'support':{'type':'string'},
   'upright':{'type':'boolean'},'surface_index':{'type':'integer'},
   'dx_m':{'type':'number'},'dy_m':{'type':'number'},'yaw_deg':{'type':'number'},
   'reason':{'type':'string'}},'required':['id','role','support','upright','surface_index','dx_m','dy_m','yaw_deg','reason'],'additionalProperties':False}},
 'limitations':{'type':'array','items':{'type':'string'}}},
 'required':['assessment','complete','objects','limitations'],'additionalProperties':False}


def plan_relations(scene,project,views,output,previous=None):
    project,views,output=map(Path,(project,views,output));output.mkdir(parents=True,exist_ok=True)
    context={'objects':[], 'previous_trial':previous,'camera':scene.preprocessing.get('camera')}
    for o in scene.objects:
        if not o.selected:continue
        try:upright_patches=surfaces(o,project,True)
        except ValueError:upright_patches=[]
        context['objects'].append({'id':o.id,'name':o.name,'bbox':o.bbox,'position':o.position,
             'size':o.size,'support':o.support,'locked':o.locked,
             'surfaces_current':surfaces(o,project),'surfaces_upright':upright_patches})
    prompt=(Path(__file__).parents[2]/'prompts/astra_relations.txt').read_text()+'\n'+json.dumps(context,ensure_ascii=False)
    (output/'prompt.txt').write_text(prompt);atomic_json(output/'schema.json',SCHEMA)
    command=shutil.which(os.environ.get('R2S_CODEX_BIN','codex'))
    if not command:raise ValueError('Codex executable unavailable')
    args=[command,'exec','--model','gpt-6-astra','--sandbox','read-only','--ephemeral','--ignore-user-config','--skip-git-repo-check','-c','model_reasoning_effort="high"',
          '--output-schema',str((output/'schema.json').resolve()),'--output-last-message',str((output/'answer.json').resolve())]
    for p in (project/scene.image,views/'source_view.png',views/'views.png'):
        if p.is_file():args.extend(['--image',str(p.resolve())])
    with (output/'astra.log').open('w') as log:
        result=subprocess.run(args+['-'],input=prompt,text=True,stdout=log,stderr=subprocess.STDOUT,cwd=output,timeout=600)
    if result.returncode:raise RuntimeError('Astra support planning failed: '+str(output/'astra.log'))
    return json.loads((output/'answer.json').read_text())


def checked_plan(scene,plan):
    candidate=scene.model_copy(deep=True);items={o.id:o for o in candidate.objects if o.selected}
    if len(plan['objects'])!=len(items) or {r['id'] for r in plan['objects']}!=set(items):
        raise ValueError('Astra must identify every selected object exactly once')
    for row in plan['objects']:
        o=items[row['id']]
        if row['support']!='table' and row['support'] not in items:raise ValueError('Unknown support')
        if not isinstance(row['surface_index'],int) or row['surface_index'] < -1:raise ValueError('Invalid support surface')
        for key,bound in [('dx_m',.12),('dy_m',.12),('yaw_deg',30)]:
            if not math.isfinite(row[key]) or abs(row[key])>bound:raise ValueError('Unbounded pose suggestion')
        if o.locked and (o.support!=row['support'] or row['upright'] or any(row[k] for k in ('dx_m','dy_m','yaw_deg'))):raise ValueError('Proposal changes locked object')
        o.support=row['support']
    SceneRecord.model_validate(candidate.model_dump())
    return candidate


def project_point(scene,position):
    cam=scene.preprocessing['camera'];x=np.array(cam['xyaxes'][:3]);y=np.array(cam['xyaxes'][3:]);z=np.cross(x,y)
    rel=np.asarray(position)-cam['position'];depth=-rel@z
    if depth<=0:raise ValueError('Object behind source camera')
    if cam.get('K'):
        K=np.asarray(cam['K']);return np.array([K[0,0]*(rel@x)/depth+K[0,2],-K[1,1]*(rel@y)/depth+K[1,2]])
    focal=scene.image_size[1]/(2*np.tan(np.radians(cam['fovy'])/2))
    return focal*np.array([rel@x,-rel@y])/depth+np.array(scene.image_size)/2


def propose_layout(scene,project,plan,output):
    candidate=checked_plan(scene,plan);items={o.id:o for o in candidate.objects};rows={r['id']:r for r in plan['objects']}
    evidence=[]
    for o in support_order(candidate):
        row=rows[o.id]
        if o.locked:continue
        r=leveled_rotation(o,project) if row['upright'] else rotation_for(o)
        r=Rotation.from_euler('z',math.radians(row['yaw_deg']))*r
        o.orientation=r.as_quat()[[3,0,1,2]].tolist();o.yaw=0
        vertices=r.apply(mesh_for(o,project,True).vertices)
        height=0.;patch=None
        if o.support!='table':
            parent=items[o.support];patches=surfaces(parent,project)
            patch=next((p for p in patches if p['index']==row['surface_index']),None)
            if patch is None:raise ValueError('Astra selected unavailable support surface')
            height=parent.position[2]+patch['z_local']
        o.position[2]=float(height-vertices[:,2].min()+.0003)
        o.position[0]+=row['dx_m'];o.position[1]+=row['dy_m']
        # Recover the lateral position at the NEW support elevation from the
        # original mask centroid. Old XY estimates on table are not binding.
        if o.support!='table' and candidate.preprocessing.get('camera'):
            from .segmentation import load_mask
            mask=load_mask(Path(project)/o.mask);yy,xx=np.where(mask)
            if len(xx):
                desired=np.array([xx.mean(),yy.mean()]);base=np.asarray(o.position)
                for _ in range(4):
                    screen=project_point(candidate,base)
                    jac=np.column_stack([(project_point(candidate,base+np.eye(3)[i]*.0001)-screen)/.0001 for i in (0,1)])
                    delta=np.linalg.lstsq(jac,desired-screen,rcond=None)[0];base[:2]+=np.clip(delta,-.2,.2)
                o.position[:2]=base[:2].tolist()
            # Clamp the object's center inside this actual patch, not a parent's
            # total bounding box (which includes vertical side panels).
            bounds=np.array(patch['xy_bounds']).reshape(2,2)+np.asarray(parent.position[:2])
            half=np.ptp(vertices[:,:2],axis=0)/2
            lo,hi=bounds[0]+half,bounds[1]-half
            if np.any(lo>hi):raise ValueError('Child does not fit on selected support patch')
            o.position[:2]=np.clip(np.asarray(o.position[:2])+[row['dx_m'],row['dy_m']],lo,hi).tolist()
        o.reconstruction['support_inference']={'model':'gpt-6-astra','role':row['role'],'reason':row['reason'],'surface':patch,'status':'candidate'}
        evidence.append({'id':o.id,'support':o.support,'surface':patch,'position':o.position,'orientation':o.orientation})
    atomic_json(Path(output)/'proposal.json',{'plan':plan,'placement':evidence,'scene':candidate.model_dump()})
    return candidate


def depth_not_worse(before,after):
    """Preserve reliable RGB-D evidence; unknown depth cannot become a pass."""
    rows={r['id']:r for r in after['objects']}
    for old in before['objects']:
        if old['median_abs_z_error_m'] is None or old['coverage']<.5:continue
        new=rows.get(old['id'])
        if not new or new['median_abs_z_error_m'] is None:return False
        if new['median_abs_z_error_m']>old['median_abs_z_error_m']+.005:return False
        if new['coverage']<old['coverage']-.05:return False
    return True


def optimize_relations(scene,project,output,seconds=10,progress=None,max_rounds=2):
    """Joint graph/pose hypotheses; promote only on visual improvement AND physics."""
    project,output=Path(project),Path(output);output.mkdir(parents=True,exist_ok=True)
    xml=physics.export_scene(scene,project,output/'baseline')
    baseline=physics.silhouette_metrics(scene,project,xml)
    baseline_depth=None
    if scene.preprocessing.get('input_path')=='rgbd':
        from .rgbd import depth_agreement
        baseline_depth=depth_agreement(scene,project,xml)
    views=output/'baseline/views';physics.capture_views(scene,xml,views,reference=project/scene.image)
    current=scene.model_copy(deep=True);previous=None;trials=[];accepted=False
    for index in range(max_rounds):
        folder=output/f'round_{index+1}'
        if progress:progress(f'Astra 自动识别支撑关系与倾斜，修正第 {index+1}/{max_rounds} 轮')
        plan=plan_relations(current,project,views,folder/'astra',previous)
        try:
            candidate=propose_layout(current,project,plan,folder)
            xml=physics.export_scene(candidate,project,folder/'proposed')
            raw=physics.validate_scene(candidate,xml,folder/'proposed',seconds)
            # Save both original and settled checks; no reset hides a bad proposal.
            settling=physics.settle(candidate,project,folder/'settle')
            xml=physics.export_scene(candidate,project,folder/'checked')
            validation=physics.validate_scene(candidate,xml,folder/'checked',seconds)
            retries=[]
            for retry in range(2):
                if validation['physics_check_pass'] or not validation['finite'] or validation['warnings']:break
                retries.append({'before':validation})
                physics.settle(candidate,project,folder/f'extended_settle_{retry}',seconds=8)
                xml=physics.export_scene(candidate,project,folder/f'extended_check_{retry}')
                validation=physics.validate_scene(candidate,xml,folder/f'extended_check_{retry}',seconds)
                retries[-1]['after']=validation
            score=physics.silhouette_metrics(candidate,project,xml)
            views=folder/'views';physics.capture_views(candidate,xml,views,reference=project/scene.image)
            depth=None;depth_ok=True
            if baseline_depth is not None:
                depth=depth_agreement(candidate,project,xml)
                depth_ok=depth_not_worse(baseline_depth,depth)
            valid=bool(depth_ok and validation['physics_check_pass'] and (score.get('mean_iou',0)>baseline.get('mean_iou',0)+.001 or (plan.get('complete',False) and score.get('mean_iou',0)>=baseline.get('mean_iou',0)-.001)))
            previous={'physics':{k:v for k,v in validation.items() if k!='initial_contacts'},'silhouette':score,'accepted':valid,'depth':depth,'depth_ok':depth_ok,'plan':plan}
            trial={**previous,'raw_physics':raw,'settling':settling,'settling_retries':retries,'path':str(folder.relative_to(project))}
            atomic_json(folder/'trial.json',trial);trials.append(trial)
            current=candidate
            if valid:
                scene.objects=candidate.objects
                for o in scene.objects:
                    if o.selected and 'support_inference' in o.reconstruction:o.reconstruction['support_inference']['status']='scene_checks_passed'
                accepted=True;break
        except (ValueError,RuntimeError) as exc:
            previous={'error':str(exc),'plan':plan,'accepted':False};trials.append(previous);atomic_json(folder/'failure.json',previous)
    report={'model':'gpt-6-astra','actual_model_called':True,'accepted':accepted,'baseline':baseline,'trials':trials,
            'mass_friction_geometry_unchanged':True,'bounded_rounds':max_rounds,
            'status':'accepted' if accepted else 'unresolved_candidate_retained',
            'path':str(output.relative_to(project))}
    atomic_json(output/'report.json',report);scene.reports['relations']=report
    return report
