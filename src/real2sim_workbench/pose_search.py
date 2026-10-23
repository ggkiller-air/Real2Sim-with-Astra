"""Astra-selected object pose hypotheses, ranked by native silhouette and physics."""
from pathlib import Path
import math
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from . import physics
from .store import atomic_json


def search_stable_poses(scene,project,output,object_ids):
    project,output=Path(project),Path(output);output.mkdir(parents=True,exist_ok=True)
    baseline_xml=physics.export_scene(scene,project,output/'baseline')
    baseline=physics.silhouette_metrics(scene,project,baseline_xml)
    reports=[]
    for oid in object_ids[:2]:
        obj=next(o for o in scene.objects if o.id==oid and o.selected and not o.locked)
        mesh=trimesh.load(project/obj.mesh,force='mesh',process=False)
        mesh.vertices*=obj.size
        parts=[]
        for path in obj.collision_meshes:
            part=trimesh.load(project/path,force='mesh');part.vertices*=obj.size;parts.append(part)
        center=sum(abs(p.volume)*p.center_mass for p in parts)/sum(abs(p.volume) for p in parts)
        transforms,probabilities=mesh.convex_hull.compute_stable_poses(center_mass=center,threshold=.01,n_samples=1,sigma=0,seed=42)
        _,_,vt=np.linalg.svd(mesh.vertices-mesh.vertices.mean(axis=0),full_matrices=False)
        axis=vt[0]
        current=Rotation.from_quat(physics.body_quaternion(obj)[[1,2,3,0]])
        direction=current.apply(axis)
        angle=math.atan2(direction[1],direction[0])
        candidates=[]
        for index,transform in enumerate(transforms[:8]):
            rotation=Rotation.from_matrix(transform[:3,:3])
            aligned=rotation.apply(axis)
            yaw_delta=angle-math.atan2(aligned[1],aligned[0])
            for offset in (-15,0,15):
                candidate=scene.model_copy(deep=True);target=next(o for o in candidate.objects if o.id==oid)
                r=Rotation.from_euler('z',yaw_delta+math.radians(offset))*rotation
                q=r.as_quat();target.orientation=q[[3,0,1,2]].tolist();target.yaw=0
                # Set support against the actual collision vertices, not visual AABB.
                vertices=np.concatenate([p.vertices for p in parts])
                target.position[2]=float(-r.apply(vertices)[:,2].min()+.0002)
                folder=output/oid/f'pose_{index}_{offset+15}'
                xml=physics.export_scene(candidate,project,output/oid/'evaluation')
                score=physics.silhouette_metrics(candidate,project,xml)
                candidates.append((score['mean_iou'],candidate.model_copy(deep=True),folder,index,offset))
                # Also test centroid alignment; keep the unshifted hypothesis so a
                # shifted candidate falling off the assumed table cannot replace it.
                anchor=np.asarray(obj.position[:2]).copy()
                cam=candidate.preprocessing['camera'];cx=np.array(cam['xyaxes'][:3]);cy=np.array(cam['xyaxes'][3:]);cz=np.cross(cx,cy)
                focal=candidate.image_size[1]/(2*np.tan(np.radians(cam['fovy'])/2))
                def project_point(position):
                    rel=np.asarray(position)-cam['position'];depth=-rel@cz
                    return focal*np.array([rel@cx,-rel@cy])/depth+np.array(candidate.image_size)/2
                for _ in range(2):
                    row=next(v for v in score['objects'] if v['id']==oid)
                    if not row['rendered_centroid']:break
                    base=np.asarray(target.position);screen=project_point(base)
                    jac=np.column_stack([(project_point(base+np.eye(3)[i]*.0001)-screen)/.0001 for i in (0,1)])
                    delta_xy=np.linalg.lstsq(jac,np.array(row['reference_centroid'])-row['rendered_centroid'],rcond=None)[0]
                    target.position[:2]=np.clip(np.array(target.position[:2])+delta_xy,anchor-.12,anchor+.12).tolist()
                    xml=physics.export_scene(candidate,project,output/oid/'evaluation')
                    score=physics.silhouette_metrics(candidate,project,xml)
                candidates.append((score['mean_iou'],candidate,folder.with_name(folder.name+'_aligned'),index,offset))
        candidates.sort(key=lambda row:row[0],reverse=True)
        trials=[];accepted=False
        atomic_json(output/oid/'candidates.json',[{'score':score,'pose':index,'yaw_offset':offset,'scene':candidate.model_dump()} for score,candidate,folder,index,offset in candidates])
        for score,candidate,folder,index,offset in candidates[:6]:
            if score<=baseline['mean_iou']+.005:continue
            try:
                settling=physics.settle(candidate,project,folder/'settled')
                xml=physics.export_scene(candidate,project,folder/'checked')
                validation=physics.validate_scene(candidate,xml,folder/'checked')
                measured=physics.silhouette_metrics(candidate,project,xml)
                target_before=next(row['iou'] for row in baseline['objects'] if row['id']==oid)
                target_after=next(row['iou'] for row in measured['objects'] if row['id']==oid)
                valid=validation['physics_check_pass'] and measured['mean_iou']>baseline['mean_iou']+.005 and target_after>target_before+.005
                trials.append({'pose':index,'yaw_offset':offset,'before_settle_iou':score,'after_settle':measured,
                               'physics_pass':validation['physics_check_pass'],'accepted':valid})
                if valid:
                    scene.objects=candidate.objects;baseline=measured;accepted=True;break
            except (ValueError,RuntimeError) as exc:trials.append({'pose':index,'error':str(exc),'accepted':False})
        reports.append({'id':oid,'stable_pose_candidates':len(transforms),'rendered_candidates':len(candidates),
                        'accepted':accepted,'trials':trials,'best_unsettled_iou':candidates[0][0] if candidates else None})
    result={'method':'convex_support_pose_hypotheses_native_silhouette_and_physics','objects':reports,
            'final_silhouette':baseline,'note':'Pose search only; generated geometry, mass and friction unchanged.'}
    atomic_json(output/'pose_search.json',result)
    return result
