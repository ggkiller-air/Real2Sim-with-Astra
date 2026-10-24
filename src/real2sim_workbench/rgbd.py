"""Calibrated RGB-D input: sensor registration, metric support frame and fitting.

Optical coordinates: +X right, +Y down, +Z forward. No historical robot/world
extrinsics are needed for the D455-local metric reconstruction.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates, distance_transform_edt, binary_erosion
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh
from .store import atomic_json


def project(points,intr):
    """RealSense Brown / inverse-Brown forward mapping (SDK convention)."""
    x,y=(np.asarray(points)[...,:2]/np.asarray(points)[...,2:3]).T
    model=intr.get('model','none');k1,k2,p1,p2,k3=intr.get('coeffs',[0]*5)
    if any([k1,k2,p1,p2,k3]):
        if not any(m in model for m in ['brown_conrady']):raise ValueError('Unsupported lens model: '+model)
        r2=x*x+y*y;factor=1+k1*r2+k2*r2*r2+k3*r2*r2*r2
        if 'inverse' in model or 'modified' in model:
            x=x*factor;y=y*factor
            x,y=x+2*p1*x*y+p2*(r2+2*x*x),y+2*p2*x*y+p1*(r2+2*y*y)
        else:x,y=x*factor+2*p1*x*y+p2*(r2+2*x*x),y*factor+2*p2*x*y+p1*(r2+2*y*y)
    return np.column_stack([x*intr['fx']+intr['ppx'],y*intr['fy']+intr['ppy']])


def deproject(depth,intr):
    h,w=depth.shape
    if (w,h)!=(intr['width'],intr['height']):raise ValueError('Depth shape disagrees with calibration')
    yy,xx=np.indices((h,w));x=(xx-intr['ppx'])/intr['fx'];y=(yy-intr['ppy'])/intr['fy'];xo=x.copy();yo=y.copy()
    k1,k2,p1,p2,k3=intr.get('coeffs',[0]*5);model=intr.get('model','none')
    if any([k1,k2,p1,p2,k3]):
        if model not in ('distortion.brown_conrady','distortion.inverse_brown_conrady'):raise ValueError('Unsupported depth distortion')
        for _ in range(10):
            r2=x*x+y*y;icdist=1/(1+((k3*r2+k2)*r2+k1)*r2)
            a,b=(x/icdist,y/icdist) if 'inverse' in model else (x,y)
            dx=2*p1*a*b+p2*(r2+2*a*a);dy=2*p2*a*b+p1*(r2+2*b*b)
            x=(xo-dx)*icdist;y=(yo-dy)*icdist
    return np.stack([x*depth,y*depth,depth],axis=-1)


def zbuffer(points,K,shape):
    """Four-neighbour forward splat with nearest-depth occlusion resolution."""
    h,w=shape;valid=np.isfinite(points).all(axis=1)&(points[:,2]>.05)&(points[:,2]<5)
    points=points[valid];uv=points[:,:2]/points[:,2:3]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
    raster=np.full(h*w,np.inf)
    for dx,dy in [(0,0),(0,1),(1,0),(1,1)]:
        px=np.floor(uv[:,0]).astype(int)+dx;py=np.floor(uv[:,1]).astype(int)+dy
        good=(px>=0)&(px<w)&(py>=0)&(py<h)
        np.minimum.at(raster,py[good]*w+px[good],points[good,2])
    raster=raster.reshape(h,w);raster[~np.isfinite(raster)]=0
    return raster.astype(np.float32)


def register_capture(folder,output):
    folder,output=Path(folder).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    meta=json.loads((folder/'capture_metadata.json').read_text())
    depth_info=meta['d455_depth'];rgb_info=meta['rgb']['image_external'];pre=rgb_info['preprocessing']
    if pre['serial']!=depth_info['serial']:raise ValueError('RGB/depth camera serial mismatch')
    raw=np.asarray(Image.open(folder/depth_info['file']));scale=depth_info['depth_scale_m']
    if raw.dtype!=np.uint16 or not 0<scale<.1:raise ValueError('Expected uint16 depth and a valid metre scale')
    depth=raw.astype(np.float64)*scale
    if (folder/'d455_depth_m.npy').exists():
        saved=np.load(folder/'d455_depth_m.npy',allow_pickle=False)
        if saved.shape!=depth.shape or not np.allclose(saved,depth,atol=2e-6):raise ValueError('PNG and NPY metric depth disagree')
    intr=depth_info['color_intrinsics_native'];T=np.array(depth_info['color_from_depth'])
    if T.shape!=(4,4) or not np.allclose(T[3],[0,0,0,1]) or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-4):raise ValueError('Invalid depth-to-color rigid transform')
    source=np.asarray(Image.open(folder/rgb_info['file']).convert('RGB'));h,w=source.shape[:2]
    crop=pre.get('crop_roi_xyxy') or [0,0,intr['width'],intr['height']];sx=w/(crop[2]-crop[0]);sy=h/(crop[3]-crop[1])
    K=np.array([[intr['fx']*sx,0,(intr['ppx']-crop[0]+.5)*sx-.5],[0,intr['fy']*sy,(intr['ppy']-crop[1]+.5)*sy-.5],[0,0,1.]])
    yy,xx=np.indices((h,w));rays=np.stack([(xx-K[0,2])/K[0,0],(yy-K[1,2])/K[1,1],np.ones((h,w))],axis=-1)
    distorted=project(rays.reshape(-1,3),intr)
    sample_x=((distorted[:,0]-crop[0]+.5)*sx-.5).reshape(h,w);sample_y=((distorted[:,1]-crop[1]+.5)*sy-.5).reshape(h,w)
    rectified=np.stack([map_coordinates(source[...,c],[sample_y,sample_x],order=1,mode='nearest') for c in range(3)],axis=-1).astype(np.uint8)
    Image.fromarray(rectified).save(output/'rectified_rgb.png')
    points=deproject(depth,depth_info['intrinsics']).reshape(-1,3);points=points[points[:,2]>0]@T[:3,:3].T+T[:3,3]
    aligned=zbuffer(points,K,(h,w));valid=aligned>0
    if valid.mean()<.3:raise ValueError('Insufficient registered sensor depth')
    distance,indices=distance_transform_edt(~valid,return_indices=True);filled=aligned[tuple(indices)]
    pointmap=(rays*filled[...,None]*[-1,-1,1]).astype(np.float32)
    np.savez_compressed(output/'registered_depth.npz',depth_m=aligned,observed_valid=valid,pointmap=pointmap,K=K,raw_sample_x=sample_x,raw_sample_y=sample_y,hole_distance_px=distance)
    report={'kind':'rgbd','primary_camera':'D455','source_folder':str(folder),'metric_units':'metres','depth_scale_m':scale,'registered_valid_fraction':float(valid.mean()),'inference_hole_fill':'nearest observed z, never used as a measurement','max_inference_hole_distance_px':float(distance.max()),'K_rectified':K.tolist(),'registration':'factory depth->color rigid transform; SDK lens model; crop/resize; rectified pinhole; nearest-depth z-buffer','hardware_synchronized':False,'host_receive_span_s':meta['receive_timestamp_span_s'],'historical_robot_extrinsics_used_for_metric_layout':False,'auxiliary_views':[{'name':k,'file':v['file'],'use':'appearance reference only; historical mounting not verified'} for k,v in meta['rgb'].items() if k!='image_external'],'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}}
    atomic_json(output/'registration.json',report);return report


def warp_mask(path,registration,destination):
    array=np.asarray(Image.open(path).convert('L'))
    aligned=map_coordinates(array,[registration['raw_sample_y'],registration['raw_sample_x']],order=0,mode='constant',cval=0)>127
    Image.fromarray(aligned.astype(np.uint8)*255).save(destination);return aligned


def metric_support_frame(pointmap,mask):
    valid=mask&np.isfinite(pointmap).all(axis=-1)&(pointmap[...,2]>0)
    pts=pointmap[valid];rng=np.random.default_rng(42)
    if len(pts)<100:raise ValueError('Insufficient observed table depth')
    pts=pts[rng.choice(len(pts),min(len(pts),10000),replace=False)];best=np.zeros(len(pts),bool)
    for _ in range(300):
        a,b,c=pts[rng.choice(len(pts),3,replace=False)];n=np.cross(b-a,c-a);norm=np.linalg.norm(n)
        if norm<1e-8:continue
        n/=norm;inliers=abs((pts-a)@n)<.004
        if inliers.sum()>best.sum():best=inliers
    if best.mean()<.5:raise ValueError('No consistent sensor-measured support plane')
    origin=np.mean(pts[best],axis=0);_,_,vt=np.linalg.svd(pts[best]-origin,full_matrices=False);z=vt[-1]
    if z@origin>0:z=-z
    x=np.array([-1.,0,0]);x-=z*(x@z);x/=np.linalg.norm(x);y=np.cross(z,x);basis=np.column_stack([x,y,z])
    return origin,basis,{'method':'observed D455 depth RANSAC + SVD','inlier_fraction':float(best.mean()),'rms_m':float(np.sqrt(np.mean(((pts[best]-origin)@z)**2))),'scale_source':'sensor_depth_metres','meters_per_model_unit':1.,'origin_model':origin.tolist(),'world_basis_columns':basis.tolist()}


def fit_mesh_to_observed(mesh,observed,limit=.06):
    """Bounded similarity fit to visible measured surfaces; not full-surface ICP."""
    if len(observed)<50:raise ValueError('Insufficient object depth; cannot claim metric placement')
    rng=np.random.default_rng(42);target=observed[rng.choice(len(observed),min(1600,len(observed)),replace=False)]
    surface,_=trimesh.sample.sample_surface(mesh,6000,seed=42);center=mesh.bounds.mean(axis=0);tree=cKDTree(surface-center)
    def error(v):
        r=Rotation.from_rotvec(v[3:6]).as_matrix();s=np.exp(v[6]);local=(target-center-v[:3])@r/s
        distances=tree.query(local,workers=1)[0]*s
        return np.r_[distances,np.array(v[:3])*.1,np.array(v[3:6])*.01,v[6]*.003]
    before=error(np.zeros(7))[:len(target)];solution=least_squares(error,np.zeros(7),bounds=([-limit]*3+[-.3]*3+[-.45],[limit]*3+[.3]*3+[.45]),loss='soft_l1',f_scale=.003,max_nfev=65,diff_step=1e-3)
    v=solution.x;after=error(v)[:len(target)];accepted=float(np.median(after))<float(np.median(before))
    if accepted:mesh.vertices=(mesh.vertices-center)@Rotation.from_rotvec(v[3:6]).as_matrix().T*np.exp(v[6])+center+v[:3]
    return {'observed_points':len(observed),'surface_distance_median_before_m':float(np.median(before)),'surface_distance_median_after_m':float(np.median(after)) if accepted else float(np.median(before)),'accepted':accepted,'translation_m':v[:3].tolist() if accepted else [0,0,0],'scale':float(np.exp(v[6])) if accepted else 1.,'rotation_vector':v[3:6].tolist() if accepted else [0,0,0],'note':'Visible-surface fit residual, not independent ground truth accuracy.'}


def assemble_metric(scene,project,job,result):
    from .geometry import prepare_visual_collision
    project,job=Path(project),Path(job);sensor=np.load(job/result['camera_geometry'],allow_pickle=False);pointmap=sensor['pointmap'];valid=sensor['observed_valid']
    table=np.asarray(Image.open(project/scene.preprocessing['table_mask']).convert('L'))>127
    table &= valid
    if scene.preprocessing.get('exclusion_mask'):table &= np.asarray(Image.open(project/scene.preprocessing['exclusion_mask']).convert('L'))<128
    for o in scene.objects:table &= np.asarray(Image.open(project/o.mask).convert('L'))<128
    origin,basis,frame=metric_support_frame(pointmap,table)
    world_points=(pointmap-origin)@basis;K=sensor['K'];width,height=scene.image_size
    scene.preprocessing['camera']={'position':((-origin)@basis).tolist(),'xyaxes':np.r_[np.array([-1.,0,0])@basis,np.array([0.,1,0])@basis].tolist(),'fovy':float(np.degrees(2*np.arctan(height/2/K[1,1]))),'K':K.tolist(),'resolution':[width,height],'source':'current factory RGB calibration, rectified'}
    rows=[]
    for row in result['objects']:
        obj=next(o for o in scene.objects if o.id==row['id']);mask=np.asarray(Image.open(project/obj.mask).convert('L'))>127
        observed=world_points[binary_erosion(mask,iterations=2)&valid&(world_points[...,2]>.003)&(world_points[...,2]<.5)]
        mesh=trimesh.load(job/row['camera_mesh'],force='scene',process=False).to_mesh();mesh.vertices=(mesh.vertices-origin)@basis
        fit=fit_mesh_to_observed(mesh,observed)
        correction=-float(mesh.bounds[0,2])+.0002;mesh.vertices[:,2]+=correction
        prep=prepare_visual_collision(mesh,job/(obj.id+'.obj'))
        obj.mesh=str((job/(obj.id+'.obj')).relative_to(project));obj.visual_texture=str(prep['texture'].relative_to(project));obj.collision_meshes=[str(p.relative_to(project)) for p in prep['collision']]
        obj.size=prep['size'];obj.position=prep['center'];obj.geometry='sam3d';obj.yaw=0;obj.orientation=[1,0,0,0]
        obj.reconstruction={**row,'dimension_source':'sensor_metric_depth_with_generated_unseen_geometry','support_translation_z_m':correction,'depth_fit':fit,'mass_friction_source':'assumed'}
        np.savez_compressed(job/(obj.id+'_observed.npz'),points=observed)
        rows.append({'id':obj.id,'name':obj.name,**fit,'observed_mask_fraction':float((mask&valid).sum()/max(1,mask.sum())),'sensor_height_98_m':float(np.quantile(observed[:,2],.98)),'support_translation_z_m':correction})
    surface=world_points[table];lo,hi=np.quantile(surface[:,:2],[.01,.99],axis=0)
    if scene.table.calibration_source!='user':
        scene.table.width=float(2*max(abs(lo[0]),abs(hi[0]))+.03);scene.table.depth=float(2*max(abs(lo[1]),abs(hi[1]))+.03)
    scene.preprocessing['metric_frame']=frame
    scene.reports['reconstruction']={'job':job.name,'input_path':'rgbd','real_mesh_count':len(rows),'frame':frame,'objects':rows,'table_extent_source':'observed support footprint plus 3 cm margin; not full unseen table dimensions'}
    return scene.reports['reconstruction']


def depth_agreement(scene,project,xml):
    """Compare native rendered optical Z with observed sensor Z on shared pixels."""
    import mujoco
    project=Path(project);sensor=np.load(project/scene.preprocessing['depth_registration'],allow_pickle=False)
    model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    width,height=scene.image_size;option=mujoco.MjvOption();option.geomgroup[3]=0
    from . import rendering
    with rendering.renderer(model,scene.image_size) as renderer:
        renderer.enable_depth_rendering();renderer.update_scene(data,camera='source',scene_option=option);depth=rendering.source_pixels(renderer.render(),scene.image_size)
        renderer.disable_depth_rendering();renderer.enable_segmentation_rendering();renderer.update_scene(data,camera='source',scene_option=option);seg=rendering.source_pixels(renderer.render(),scene.image_size)
    rows=[]
    for obj in scene.objects:
        if not obj.selected:continue
        mask=binary_erosion(np.asarray(Image.open(project/obj.mask))>127,iterations=2)&sensor['observed_valid']
        bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,obj.id);geoms=np.flatnonzero(model.geom_bodyid==bid)
        shared=mask&np.isin(seg[:,:,0],geoms)&(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
        error=abs(depth[shared]-sensor['depth_m'][shared])
        rows.append({'id':obj.id,'name':obj.name,'shared_pixels':int(shared.sum()),'observed_mask_pixels':int(mask.sum()),'coverage':float(shared.sum()/max(1,mask.sum())),'median_abs_z_error_m':float(np.median(error)) if len(error) else None,'p90_abs_z_error_m':float(np.quantile(error,.9)) if len(error) else None})
    return {'objects':rows,'render_size':list(rendering.render_size(scene.image_size)),'source_image_size':scene.image_size,'note':'Sensor agreement on shared visible pixels; excludes missing depth, hidden surfaces and independent metrology.'}
