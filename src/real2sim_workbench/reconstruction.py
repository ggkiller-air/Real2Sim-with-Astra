"""Single-image model geometry to a common, explicit Z-up support frame."""
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
from .geometry import prepare_visual_collision


def fit_support_frame(pointmap, table_mask, assumed_width):
    valid = table_mask & np.isfinite(pointmap).all(axis=-1) & (np.linalg.norm(pointmap,axis=-1)>1e-6)
    points=pointmap[valid]
    if len(points)<100:
        raise ValueError('桌面深度有效点不足，无法建立共同支撑面')
    rng=np.random.default_rng(42)
    pts=points[rng.choice(len(points),min(len(points),8000),replace=False)]
    spread=np.linalg.norm(np.percentile(pts,90,axis=0)-np.percentile(pts,10,axis=0))
    tolerance=max(spread*.008,1e-5)
    best=np.zeros(len(pts),dtype=bool)
    for _ in range(250):
        a,b,c=pts[rng.choice(len(pts),3,replace=False)]
        n=np.cross(b-a,c-a);norm=np.linalg.norm(n)
        if norm<1e-9:continue
        n/=norm
        inliers=np.abs((pts-a)@n)<tolerance
        if inliers.sum()>best.sum():best=inliers
    if best.sum()<100 or best.mean()<.3:
        raise ValueError('单图预测的桌面不一致，需要修正桌面分割')
    origin=np.median(pts[best],axis=0)
    _,_,vt=np.linalg.svd(pts[best]-origin,full_matrices=False)
    normal=vt[-1]
    if normal@origin>0:normal=-normal
    x=np.array([-1.,0,0]);x-=normal*(x@normal);x/=np.linalg.norm(x)
    y=np.cross(normal,x)
    basis=np.column_stack([x,y,normal])
    planar=(pts[best]-origin)@basis
    span=np.percentile(planar,98,axis=0)-np.percentile(planar,2,axis=0)
    scale=assumed_width/span[0]
    return origin,basis,scale,{'method':'MoGe predicted pointmap + SAM3 table mask RANSAC',
        'inlier_fraction':float(best.mean()),'rms_model_units':float(np.sqrt(np.mean(((pts[best]-origin)@normal)**2))),
        'scale_source':'assumed_table_width_not_measured','meters_per_model_unit':float(scale)}


def assemble(scene, project_folder, job_folder, result):
    project_folder,job_folder=Path(project_folder),Path(job_folder)
    camera=np.load(job_folder/result['camera_geometry'])
    pointmap=camera['pointmap']
    table_path=scene.preprocessing.get('table_mask')
    if not table_path:raise ValueError('缺少桌面分割，请先自动识别')
    mask=np.asarray(Image.open(project_folder/table_path).convert('L').resize((pointmap.shape[1],pointmap.shape[0])))>127
    for obj in scene.objects:
        if obj.mask:
            mask &= np.asarray(Image.open(project_folder/obj.mask).convert('L').resize((pointmap.shape[1],pointmap.shape[0])))<128
    origin,basis,scale,report=fit_support_frame(pointmap,mask,scene.table.width)
    camera_pos=(-origin)@basis*scale
    intrinsics=camera['intrinsics'].squeeze()
    scene.preprocessing['camera']={'position':camera_pos.tolist(),
        'xyaxes':np.concatenate([np.array([-1.,0,0])@basis,np.array([0.,1,0])@basis]).tolist(),
        'fovy':float(np.degrees(2*np.arctan(.5/intrinsics[1,1]))),
        'intrinsics_normalized':intrinsics.tolist(),'source':'MoGe single-image estimate'}
    report.update(origin_model=origin.tolist(),world_basis_columns=basis.tolist())
    for row in result['objects']:
        obj=next(o for o in scene.objects if o.id==row['id'])
        mesh=trimesh.load(job_folder/row['camera_mesh'],force='scene',process=False).to_mesh()
        mesh.vertices=(mesh.vertices-origin)@basis*scale
        prep=prepare_visual_collision(mesh,job_folder/(obj.id+'.obj'))
        obj.mesh=str((job_folder/(obj.id+'.obj')).relative_to(project_folder))
        obj.visual_texture=str(prep['texture'].relative_to(project_folder))
        obj.collision_meshes=[str(p.relative_to(project_folder)) for p in prep['collision']]
        obj.geometry='sam3d';obj.size=prep['size'];obj.yaw=0;obj.orientation=[1,0,0,0]
        obj.position=prep['center']
        # Support correction is explicit and recorded, not a fabricated model pose.
        correction=obj.size[2]/2+.0002-obj.position[2]
        obj.position[2]+=correction
        obj.reconstruction={**row,'support_translation_z_m':correction,
                            **{k:v for k,v in prep.items() if k not in ('texture','collision')},
                            'dimension_source':'single_image_prediction_scaled_to_assumed_table_width',
                            'mass_friction_source':obj.parameter_source}
    chosen=[o for o in scene.objects if o.selected]
    # Accommodate the generated footprint without asking the user for background controls.
    scene.table.depth=max(scene.table.depth,2*max(abs(o.position[1])+o.size[1]/2 for o in chosen)+.12)
    scene.table.width=max(scene.table.width,2*max(abs(o.position[0])+o.size[0]/2 for o in chosen)+.12)
    scene.reports['reconstruction']={'job':job_folder.name,'raw_layout_applied':True,'frame':report,
                                    'real_mesh_count':len(result['objects']),'unseen_surfaces':'model inferred'}
    return scene.reports['reconstruction']
