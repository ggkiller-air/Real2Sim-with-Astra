"""Choose supported poses using measured surfaces, then native RGB-D agreement.

Stable convex-hull poses are hypotheses, never fixed joints. The ordinary
contact and dynamic validation stages must still run afterwards.
"""
from pathlib import Path
import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from . import physics
from .rgbd import depth_agreement
from .store import atomic_json


def fit_supported_poses(scene, project, output, object_ids=None):
    project, output = Path(project), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for obj in scene.objects:
        if not obj.selected or obj.locked or (object_ids is not None and obj.id not in object_ids):
            continue
        mesh = trimesh.load(project / obj.mesh, force='mesh', process=False)
        mesh.vertices *= obj.size
        parts = []
        for path in obj.collision_meshes:
            part = trimesh.load(project / path, force='mesh')
            part.vertices *= obj.size
            parts.append(part)
        if not parts:
            continue
        vertices = np.concatenate([p.vertices for p in parts])
        center_mass = sum(abs(p.volume)*p.center_mass for p in parts)/sum(abs(p.volume) for p in parts)
        transforms, probabilities = mesh.convex_hull.compute_stable_poses(
            center_mass=center_mass, threshold=.005, n_samples=1, sigma=0, seed=42)
        observed = np.load(project / 'jobs' / scene.reports['reconstruction']['job'] / (obj.id+'_observed.npz'))['points']
        rng = np.random.default_rng(42)
        target = observed[rng.choice(len(observed), min(1200, len(observed)), replace=False)]
        surface, _ = trimesh.sample.sample_surface(mesh, 8000, seed=42)
        tree = cKDTree(surface)
        _, _, vt = np.linalg.svd(mesh.vertices-mesh.vertices.mean(axis=0), full_matrices=False)
        axis = vt[0]
        current = Rotation.from_quat(physics.body_quaternion(obj)[[1,2,3,0]])
        direction = current.apply(axis)
        heading = np.arctan2(direction[1], direction[0])
        anchor = np.array(obj.position[:2]); candidates = []
        for index, transform in enumerate(transforms[:12]):
            stable = Rotation.from_matrix(transform[:3,:3])
            direction = stable.apply(axis)
            yaw = heading-np.arctan2(direction[1], direction[0])
            def pose(v):
                rotation = Rotation.from_euler('z', yaw+v[2])*stable
                scale = np.exp(v[3])
                position = np.r_[anchor+v[:2], -rotation.apply(vertices)[:,2].min()*scale+.0002]
                return rotation, scale, position
            def residual(v):
                rotation, scale, position = pose(v)
                distances = tree.query(rotation.inv().apply(target-position)/scale)[0]*scale
                return np.r_[distances, v[:2]*.1, v[2]*.003, v[3]*.003]
            solved = least_squares(residual, np.zeros(4), bounds=([-.05,-.05,-.6,-.3],[.05,.05,.6,.2]),
                                   loss='soft_l1', f_scale=.003, max_nfev=55, diff_step=1e-3)
            rotation, scale, position = pose(solved.x)
            candidate = obj.model_copy(deep=True)
            q = rotation.as_quat(); candidate.orientation = q[[3,0,1,2]].tolist(); candidate.yaw = 0
            candidate.position = position.tolist(); candidate.size = (np.array(obj.size)*scale).tolist()
            candidates.append((float(np.median(residual(solved.x)[:len(target)])), candidate, index))
        candidates.sort(key=lambda row: row[0])
        trials = []
        for residual, candidate, index in candidates:
            proposal = scene.model_copy(deep=True)
            proposal.objects = [candidate if o.id == obj.id else o for o in proposal.objects]
            xml = physics.export_scene(proposal, project, output/'evaluation')
            silhouette = next(r for r in physics.silhouette_metrics(proposal, project, xml)['objects'] if r['id']==obj.id)
            # Refine the visible footprint without abandoning the support pose.
            # Depth fit alone can follow the wrong face of a symmetric component.
            initial_size = np.asarray(obj.size)
            cam = scene.preprocessing['camera']; K = np.asarray(cam['K'])
            cx = np.asarray(cam['xyaxes'][:3]); cy = np.asarray(cam['xyaxes'][3:]); cz = np.cross(cx,cy)
            def screen(position):
                rel = np.asarray(position)-cam['position']; z = -rel@cz
                return np.array([K[0,0]*(rel@cx)/z+K[0,2], -K[1,1]*(rel@cy)/z+K[1,2]])
            for _ in range(3):
                if not silhouette['rendered_centroid']: break
                factor = np.sqrt(silhouette['reference_pixels']/max(1,silhouette['rendered_pixels']))
                scale = float(np.clip(candidate.size[0]/initial_size[0]*factor, np.exp(-.3), np.exp(.2)))
                candidate.size = (initial_size*scale).tolist()
                rotation = Rotation.from_quat(np.asarray(candidate.orientation)[[1,2,3,0]])
                candidate.position[2] = float(-rotation.apply(vertices)[:,2].min()*scale+.0002)
                position = np.asarray(candidate.position); base = screen(position)
                jac = np.column_stack([(screen(position+np.eye(3)[j]*.0001)-base)/.0001 for j in (0,1)])
                delta = np.linalg.lstsq(jac,np.asarray(silhouette['reference_centroid'])-silhouette['rendered_centroid'],rcond=None)[0]
                candidate.position[:2] = np.clip(position[:2]+delta,anchor-.05,anchor+.05).tolist()
                proposal.objects = [candidate if o.id==obj.id else o for o in proposal.objects]
                xml = physics.export_scene(proposal, project, output/'evaluation')
                silhouette = next(r for r in physics.silhouette_metrics(proposal, project, xml)['objects'] if r['id']==obj.id)
            if obj.reconstruction.get('asset_refinement',{}).get('template')=='hammer':
                refine_supported_yaw(proposal,project,xml,candidate)
                xml=physics.export_scene(proposal,project,output/'evaluation')
                silhouette=next(r for r in physics.silhouette_metrics(proposal,project,xml)['objects'] if r['id']==obj.id)
            depth = next(r for r in depth_agreement(proposal, project, xml)['objects'] if r['id']==obj.id)
            loss = (1-silhouette['iou'])*.06 + (depth['median_abs_z_error_m'] or 1) + (1-depth['coverage'])*.01
            trials.append({'index':index,'loss':loss,'surface_median_m':residual,'silhouette':silhouette,'depth':depth,'object':candidate.model_dump()})
        if not trials:
            raise ValueError('No supported pose for '+obj.name)
        best = min(trials, key=lambda row:row['loss'])
        from .models import ObjectRecord
        chosen = ObjectRecord.model_validate(best['object'])
        scene.objects = [chosen if o.id==obj.id else o for o in scene.objects]
        reports.append({'id':obj.id,'name':obj.name,'before':obj.model_dump(),'chosen':best['index'],'trials':trials})
        print('Depth-supported pose: '+obj.name+' IoU='+str(round(best['silhouette']['iou'],3)),flush=True)
    result = {'method':'convex support hypotheses, metric visible-surface fit, native silhouette and optical-depth ranking',
              'objects':reports,'limits':{'xy_m':.05,'yaw_rad':.6,'log_scale':[-.3,.2]},
              'note':'Measured depth constrains visible surfaces; unseen shape and stable-pose hypotheses remain generated.'}
    atomic_json(output/'supported_poses.json',result)
    return result


def refine_supported_yaw(scene, project, xml, obj):
    """Search tabletop yaw with native masks/depth while preserving support."""
    import mujoco
    from PIL import Image
    from scipy.ndimage import binary_erosion
    sensor = np.load(Path(project)/scene.preprocessing['depth_registration'])
    expected = np.asarray(Image.open(Path(project)/obj.mask).convert('L'))>127
    observed = binary_erosion(expected,iterations=2)&sensor['observed_valid']
    yy,xx = np.where(expected); target = np.array([xx.mean(),yy.mean()])
    m = mujoco.MjModel.from_xml_path(str(xml)); d = mujoco.MjData(m)
    bid = mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,obj.id)
    adr = m.jnt_qposadr[m.body_jntadr[bid]]; geoms = np.flatnonzero(m.geom_bodyid==bid)
    original = Rotation.from_quat(physics.body_quaternion(obj)[[1,2,3,0]])
    anchor = np.asarray(obj.position); cam=scene.preprocessing['camera']; K=np.asarray(cam['K'])
    cx=np.asarray(cam['xyaxes'][:3]);cy=np.asarray(cam['xyaxes'][3:]);cz=np.cross(cx,cy)
    def screen(position):
        rel=position-cam['position'];z=-rel@cz
        return np.array([K[0,0]*(rel@cx)/z+K[0,2],-K[1,1]*(rel@cy)/z+K[1,2]])
    jac=np.column_stack([(screen(anchor+np.eye(3)[i]*.0001)-screen(anchor))/.0001 for i in (0,1)])
    option=mujoco.MjvOption();option.geomgroup[3]=0;best=None
    w,h=scene.image_size
    from . import rendering
    with rendering.renderer(m,scene.image_size) as renderer:
        for yaw in np.linspace(-np.pi/3,np.pi/3,13):
            rotation=Rotation.from_euler('z',yaw)*original;q=rotation.as_quat();position=anchor.copy()
            for _ in range(3):
                d.qpos[adr:adr+3]=position;d.qpos[adr+3:adr+7]=q[[3,0,1,2]];mujoco.mj_forward(m,d)
                renderer.disable_depth_rendering();renderer.enable_segmentation_rendering();renderer.update_scene(d,camera='source',scene_option=option);seg=rendering.source_pixels(renderer.render(),scene.image_size)
                mask=np.isin(seg[:,:,0],geoms)&(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM));ys,xs=np.where(mask)
                if not len(xs):break
                delta=np.linalg.lstsq(jac,target-[xs.mean(),ys.mean()],rcond=None)[0]
                if _<2:position[:2]=np.clip(position[:2]+delta,anchor[:2]-.04,anchor[:2]+.04)
            if not len(xs):continue
            renderer.disable_segmentation_rendering();renderer.enable_depth_rendering();renderer.update_scene(d,camera='source',scene_option=option);depth=rendering.source_pixels(renderer.render(),scene.image_size)
            shared=mask&observed;error=float(np.median(abs(depth[shared]-sensor['depth_m'][shared]))) if shared.any() else 1.
            iou=(mask&expected).sum()/max(1,(mask|expected).sum());coverage=shared.sum()/max(1,observed.sum())
            loss=(1-iou)*.06+error+(1-coverage)*.01
            if best is None or loss<best[0]:best=(loss,position.copy(),q[[3,0,1,2]].copy())
    if best:
        obj.position=best[1].tolist();obj.orientation=best[2].tolist();obj.yaw=0
