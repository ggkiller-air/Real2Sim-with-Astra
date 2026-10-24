"""Standalone worker; run ONLY with Isaac Sim's bundled Python.

No workbench imports, pip installation, source data mutation or benchmark runtime.
"""
import argparse
import json
import math
from pathlib import Path
import time

parser=argparse.ArgumentParser()
parser.add_argument('--manifest',type=Path)
parser.add_argument('--view',type=Path)
parser.add_argument('--ready',type=Path)
parser.add_argument('--seconds',type=float,default=10)
args=parser.parse_args()
from isaacsim import SimulationApp
app=SimulationApp({'headless':not bool(args.view),'width':960,'height':720,'window_width':1100,'window_height':850,
                   'renderer':'RaytracedLighting','extra_args':['--/physics/physxDispatcher=true','--/plugins/carb.tasking.plugin/threadCount=4']})
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,UsdLux,PhysxSchema,Gf,Sdf,Vt
import omni.usd
from isaacsim.core.api import SimulationContext
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.viewports import set_camera_view


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False))


def material(stage,path,color=(.55,.5,.4),texture=None):
    mat=UsdShade.Material.Define(stage,path);shader=UsdShade.Shader.Define(stage,path+'/shader');shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color));shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.65)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    if texture:
        tex=UsdShade.Shader.Define(stage,path+'/texture');tex.CreateIdAttr('UsdUVTexture');tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture));tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
        reader=UsdShade.Shader.Define(stage,path+'/uv');reader.CreateIdAttr('UsdPrimvarReader_float2');reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
        tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
        shader.GetInput('diffuseColor').ConnectToSource(tex.ConnectableAPI(),'rgb')
    return mat


def friction_material(stage,path,friction):
    mat=UsdShade.Material.Define(stage,path);api=UsdPhysics.MaterialAPI.Apply(mat.GetPrim());api.CreateStaticFrictionAttr(friction);api.CreateDynamicFrictionAttr(friction);api.CreateRestitutionAttr(0)
    return mat


def collision(prim,mat,mesh=False):
    UsdPhysics.CollisionAPI.Apply(prim)
    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
    if mesh:UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
    api=PhysxSchema.PhysxCollisionAPI.Apply(prim);api.CreateContactOffsetAttr(.002);api.CreateRestOffsetAttr(0.)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat,materialPurpose='physics')


def cube(stage,path,size,position,mat,physical):
    geo=UsdGeom.Cube.Define(stage,path);geo.CreateSizeAttr(1.)
    geo.AddTranslateOp().Set(Gf.Vec3d(*position));geo.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(geo.GetPrim()).Bind(mat)
    collision(geo.GetPrim(),physical);return geo


def mesh(stage,path,data):
    geo=UsdGeom.Mesh.Define(stage,path);geo.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(data['vertices'],dtype=np.float32)))
    faces=np.asarray(data['faces'],dtype=np.int32);geo.CreateFaceVertexCountsAttr([3]*len(faces));geo.CreateFaceVertexIndicesAttr(faces.ravel().tolist());geo.CreateSubdivisionSchemeAttr('none');geo.CreateDoubleSidedAttr(True)
    if data.get('uv') is not None:
        uv=UsdGeom.PrimvarsAPI(geo).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex)
        uv.Set(Vt.Vec2fArray.FromNumpy(np.asarray(data['uv'],dtype=np.float32)))
    return geo


def author(manifest,path):
    stage=Usd.Stage.CreateNew(str(path));world=UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(world.GetPrim());UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1.)
    scene=UsdPhysics.Scene.Define(stage,'/World/Physics');scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));scene.CreateGravityMagnitudeAttr(9.81)
    phys=PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim());phys.CreateEnableGPUDynamicsAttr(False);phys.CreateBroadphaseTypeAttr('MBP');phys.CreateSolverTypeAttr('TGS');phys.CreateTimeStepsPerSecondAttr(500)
    groundmat=material(stage,'/World/Materials/table',manifest['table_rgba'][:3]);groundphys=friction_material(stage,'/World/Materials/table_physics',.5)
    t=manifest['table'];cube(stage,'/World/table',[t['width'],t['depth'],.05],[*t.get('center_xy',[0,0]),-.025],groundmat,groundphys)
    cube(stage,'/World/floor',[10,10,.02],[0,0,-.81],material(stage,'/World/Materials/floor',(.16,.19,.22)),groundphys)
    for o in manifest['objects']:
        path='/World/'+o['id'];body=UsdGeom.Xform.Define(stage,path);body.AddTranslateOp().Set(Gf.Vec3d(*o['position']));q=o['orientation'];body.AddOrientOp().Set(Gf.Quatf(q[0],Gf.Vec3f(*q[1:])))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim());PhysxSchema.PhysxContactReportAPI.Apply(body.GetPrim()).CreateThresholdAttr(0.);api=PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim());api.CreateSolverPositionIterationCountAttr(32);api.CreateSolverVelocityIterationCountAttr(8)
        mass=UsdPhysics.MassAPI.Apply(body.GetPrim());mass.CreateMassAttr(o['mass']);mass.CreateCenterOfMassAttr(Gf.Vec3f(*o['center_of_mass']));mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*o['inertia']));q=o['principal_axes'];mass.CreatePrincipalAxesAttr(Gf.Quatf(q[0],Gf.Vec3f(*q[1:])))
        body.GetPrim().CreateAttribute('real2sim:support',Sdf.ValueTypeNames.String).Set(o['support'])
        visual=mesh(stage,path+'/visual',o['visual']);mat=material(stage,'/World/Materials/'+o['id'],texture=o['texture']);UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(mat)
        mat=friction_material(stage,'/World/Materials/'+o['id']+'_physics',o['friction'])
        for i,data in enumerate(o['collisions']):
            geo=mesh(stage,path+f'/collision_{i}',data);geo.CreateVisibilityAttr('invisible');collision(geo.GetPrim(),mat,True)
    light=UsdLux.DomeLight.Define(stage,'/World/Light');light.CreateIntensityAttr(600)
    distant=UsdLux.DistantLight.Define(stage,'/World/Sun');distant.CreateIntensityAttr(1800);distant.AddRotateXYZOp().Set(Gf.Vec3f(30,-30,20))
    cam=manifest.get('camera')
    if cam:
        geo=UsdGeom.Camera.Define(stage,'/World/SourceCamera');x=np.array(cam['xyaxes'][:3]);y=np.array(cam['xyaxes'][3:]);z=np.cross(x,y)
        m=Gf.Matrix4d(1);m.SetRow(0,Gf.Vec4d(*x,0));m.SetRow(1,Gf.Vec4d(*y,0));m.SetRow(2,Gf.Vec4d(*z,0));m.SetRow(3,Gf.Vec4d(*cam['position'],1));geo.AddTransformOp().Set(m)
        w,h=manifest['image_size'];fy=cam['K'][1][1] if cam.get('K') else h/(2*math.tan(math.radians(cam['fovy'])/2))
        geo.CreateFocalLengthAttr(24.);geo.CreateVerticalApertureAttr(24*h/fy);geo.CreateHorizontalApertureAttr(24*w/fy);geo.CreateClippingRangeAttr(Gf.Vec2f(.01,100))
    stage.GetRootLayer().Save();return stage


def main():
    output=args.view.parent if args.view else args.manifest.parent
    manifest=json.loads((output/'manifest.json').read_text())
    usd=args.view or output/'scene.usdc'
    if not args.view:author(manifest,usd)
    omni.usd.get_context().open_stage(str(usd))
    for _ in range(12):app.update()
    context=SimulationContext(physics_dt=.002,rendering_dt=1/30,stage_units_in_meters=1.,backend='numpy',device='cpu',physics_prim_path='/World/Physics')
    paths=['/World/'+o['id'] for o in manifest['objects']];filters=['/World/table']+paths
    bodies=RigidPrim(prim_paths_expr=paths,name='reconstructed',reset_xform_properties=False,track_contact_forces=True,contact_filter_prim_paths_expr=[filters for _ in paths])
    context.initialize_physics();bodies.initialize();context.play()
    positions,quats=bodies.get_world_poses();initial=positions.copy();initial_q=quats.copy()
    center=np.mean(initial,axis=0);set_camera_view(eye=center+np.array([1,-1.7,1.1]),target=center)
    if args.view:
        context.pause()
        if args.ready:write(args.ready,{'state':'ready','engine':'isaac','usd':str(usd)})
        while app.is_running():context.step(render=True)
        return
    # Independent native check from the exported initial state, no implicit settling.
    steps=round(args.seconds/.002);finite=True;contact_samples=0
    for i in range(steps):
        context.step(render=False)
        forces=bodies.get_contact_force_matrix(dt=.002);contact_samples+=1
        if i%250==0:
            positions,quats=bodies.get_world_poses()
            if not np.isfinite(positions).all():finite=False;break
    positions,quats=bodies.get_world_poses();vel=bodies.get_velocities();rows=[]
    for i,o in enumerate(manifest['objects']):
        target=0 if o['support']=='table' else 1+[r['id'] for r in manifest['objects']].index(o['support'])
        supported=bool(np.linalg.norm(forces[i,target])>1e-5)
        drift=float(np.linalg.norm(positions[i]-initial[i]));angle=math.degrees(2*math.acos(float(np.clip(abs(np.dot(quats[i],initial_q[i])),0,1))))
        speed=float(np.linalg.norm(vel[i,:3]));angular=float(np.linalg.norm(vel[i,3:]))
        rows.append({'id':o['id'],'translation_m':drift,'rotation_deg':angle,'final_speed_m_s':speed,'final_angular_speed_rad_s':angular,'expected_support_contact':supported,'stable':bool(drift<=.01 and angle<=10 and speed<=.02 and angular<=.2 and supported)})
    report={'engine':'Isaac Sim / PhysX CPU TGS','route':'shared_scene_direct_usd','finite':finite,'seconds':args.seconds,'steps':contact_samples,'expected_steps':steps,'objects':rows,'final_contact_forces':forces.tolist(),
            'dynamics_check_pass':bool(finite and contact_samples==steps and all(r['stable'] for r in rows)),
            'initial_penetration_check':'not_measured_in_physx','physics_check_pass':None,
            'limitations':['PhysX and MuJoCo contact solvers differ; results are checked separately.','Static/dynamic friction map to the MuJoCo sliding coefficient; torsional/rolling friction are not equivalent.','No native initial-penetration qualification in this worker; dynamics pass is not full physics qualification.','Special object types remain rigid annotations.']}
    write(output/'validation.json',report)
    # Capture the initial authored scene, not the fallen/final state.
    bodies.set_world_poses(initial,initial_q);bodies.set_velocities(np.zeros_like(vel))
    for _ in range(4):context.render()
    from omni.kit.viewport.utility import get_active_viewport,capture_viewport_to_file
    viewport=get_active_viewport();viewport.resolution=(960,720)
    capture_viewport_to_file(viewport,str(output/'preview.png'))
    for _ in range(20):app.update()
    print('ISAAC_RESULT '+json.dumps(report),flush=True)

try:main()
except Exception:
    import traceback
    traceback.print_exc();raise
finally:app.close()
