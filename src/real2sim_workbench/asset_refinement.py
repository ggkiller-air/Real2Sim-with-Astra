"""Astra-directed, asset-only candidates. Never mutates assembled object records."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageOps
from scipy.spatial.transform import Rotation
from .store import atomic_json
from .geometry import prepare_visual_collision

PARAMS = {
    'head_length':(.12,.35), 'head_diameter':(.12,.35), 'grip_length':(.25,.65),
    'shaft_diameter':(.025,.09), 'grip_diameter':(.04,.13),
    'housing_height':(.5,.85), 'base_fraction':(.2,.6), 'button_diameter':(.25,.7),
    'cap_height':(.025,.15), 'stem_diameter':(.15,.45),
}
SCHEMA={'type':'object','properties':{'assets':{'type':'array','items':{'type':'object','properties':{
    'id':{'type':'string'},'template':{'type':'string','enum':['hammer','button_box','mesh_cleanup','keep']},
    'review_status':{'type':'string','enum':['acceptable','needs_repair']},
    'assessment':{'type':'string'},'button_color':{'type':'string','enum':['red','green','none']},
    'parameters':{'type':'object','properties':{k:{'type':'number'} for k in PARAMS},'required':list(PARAMS),'additionalProperties':False},
    'limitations':{'type':'array','items':{'type':'string'}}},
    'required':['id','template','review_status','assessment','button_color','parameters','limitations'],'additionalProperties':False}}},
    'required':['assets'],'additionalProperties':False}


def canonical_frame(mesh):
    """OBB only defines a candidate asset frame; never moves a scene body."""
    transform,extents=trimesh.bounds.oriented_bounds(mesh)
    rows=transform[:3,:3]
    elongated=max(extents)/sorted(extents)[1]>2.2
    if elongated:
        x=rows[np.argmax(extents)].copy()
        remaining=[i for i in range(3) if i!=np.argmax(extents)]
        z=rows[max(remaining,key=lambda i:abs(rows[i,2]))].copy()
        if z[2]<0:z=-z
        # The wide end is the head; place it on negative X.
        projection=mesh.vertices@x
        spans=[]
        for choose in [projection<np.quantile(projection,.2),projection>np.quantile(projection,.8)]:
            points=mesh.vertices[choose];spans.append(np.linalg.norm(np.ptp(points@rows[remaining].T,axis=0)))
        if spans[1]>spans[0]:x=-x
    else:
        z=rows[np.argmax(abs(rows[:,2]))].copy()
        if z[2]<0:z=-z
        remaining=[i for i in range(3) if abs(rows[i]@z)<.1]
        x=rows[max(remaining,key=lambda i:abs(rows[i,0]))].copy()
        if x[0]<0:x=-x
    y=np.cross(z,x);y/=np.linalg.norm(y);x=np.cross(y,z)
    rotation=np.vstack([x,y,z]);points=mesh.vertices@rotation.T
    center=(points.min(axis=0)+points.max(axis=0))/2
    matrix=np.eye(4);matrix[:3,:3]=rotation;matrix[:3,3]=-center
    return matrix,np.ptp(points,axis=0)


def colored(mesh,color):
    mesh.visual.vertex_colors=np.tile(np.array([*color,255],dtype=np.uint8),(len(mesh.vertices),1))
    return mesh


def rounded_box(extents,center,color,radius):
    extents=np.asarray(extents);radius=min(radius,min(extents)*.18)
    ball=trimesh.creation.icosphere(subdivisions=2,radius=radius)
    clouds=[ball.vertices+np.array(sign)*(extents/2-radius) for sign in __import__('itertools').product([-1,1],repeat=3)]
    mesh=trimesh.convex.convex_hull(np.vstack(clouds));mesh.apply_translation(center)
    return colored(mesh,color)


def cylinder(radius,height,center,color,axis='z',sections=64):
    mesh=trimesh.creation.cylinder(radius=radius,height=height,sections=sections)
    if axis=='x':mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi/2,[0,1,0]))
    mesh.apply_translation(center);return colored(mesh,color)


def build_template(recipe,extents):
    p=recipe['parameters'];parts=[];dark=(39,43,43);yellow=(238,197,40)
    if recipe['template']=='hammer':
        length=float(extents[0]);head_d=p['head_diameter']*length;head_h=p['head_length']*length
        head_x=-length/2+head_d/2;end=length/2
        parts.append(cylinder(head_d/2,head_h,[head_x,0,0],dark))
        # Slightly recessed end faces retain the rubber mallet's end treatment.
        for sign in [-1,1]:parts.append(cylinder(head_d*.465,length*.003,[head_x,0,sign*(head_h/2+.0005)],(32,35,35)))
        grip_l=p['grip_length']*length;shaft_start=head_x;grip_start=end-grip_l
        parts.append(cylinder(p['shaft_diameter']*length/2,grip_start-shaft_start+.012,[((shaft_start+grip_start)/2),0,0],yellow,'x'))
        radius=p['grip_diameter']*length/2
        grip=trimesh.creation.capsule(height=max(grip_l-2*radius,.001),radius=radius,count=[16,48])
        grip.apply_transform(trimesh.transformations.rotation_matrix(math.pi/2,[0,1,0]))
        grip.vertices[:,2]*=.8
        grip.apply_translation([end-grip_l/2,0,0]);parts.append(colored(grip,dark))
    elif recipe['template']=='button_box':
        width,depth,height=map(float,extents);housing=height*p['housing_height'];base=housing*p['base_fraction']
        parts.append(rounded_box([width,depth,base],[0,0,-height/2+base/2],dark,min(width,depth)*.025))
        parts.append(rounded_box([width,depth,housing-base],[0,0,-height/2+base+(housing-base)/2],(245,204,35),min(width,depth)*.025))
        top=-height/2+housing;cap_h=height*p['cap_height'];stem_h=height-housing-cap_h
        parts.append(cylinder(min(width,depth)*p['stem_diameter']*.64,height*.018,[0,0,top+height*.009],(65,66,60)))
        parts.append(cylinder(min(width,depth)*p['stem_diameter']/2,stem_h,[0,0,top+stem_h/2],(194,193,171)))
        color=(198,32,31) if recipe['button_color']=='red' else (28,151,110)
        parts.append(cylinder(min(width,depth)*p['button_diameter']/2,cap_h,[0,0,height/2-cap_h/2],color))
    else:raise ValueError('Only hammer and button_box templates can be rebuilt')
    return trimesh.util.concatenate(parts)


def check_recipes(answer,ids):
    rows=answer['assets']
    if len(rows)!=len(ids) or {r['id'] for r in rows}!=set(ids):raise ValueError('Astra asset IDs must match the request exactly')
    for row in rows:
        if row['template'] not in ('hammer','button_box','mesh_cleanup','keep'):raise ValueError('Unknown asset template')
        if row.get('review_status','unassessed') not in ('acceptable','needs_repair','unassessed'):raise ValueError('Invalid review status')
        for key,(low,high) in PARAMS.items():
            value=row['parameters'][key]
            if not math.isfinite(value) or not low<=value<=high:raise ValueError('Out of bounds asset parameter: '+key)
        if row['template']=='button_box' and row['parameters']['housing_height']+row['parameters']['cap_height']>=.98:raise ValueError('Button stem height must remain positive')
        if row['template']=='button_box' and row['button_color'] not in ('red','green'):raise ValueError('Button color is required')
    return rows


def clean_mesh(mesh):
    """Bounded topology-preserving smoothing, without deleting small components.

    Weld duplicated export vertices so smoothing reaches shared faces. Retain
    sampled vertex appearance; the normal exporter rebakes it into a texture.
    This cannot repair semantic shape, holes, missing bristles or wrong colors.
    """
    mesh=mesh.copy()
    colors=mesh.visual.to_color() if hasattr(mesh.visual,'to_color') else mesh.visual
    mesh.visual=trimesh.visual.ColorVisuals(mesh,vertex_colors=colors.vertex_colors.copy())
    mesh.merge_vertices(merge_tex=True,merge_norm=True)
    original=mesh.vertices.copy()
    limit=float(np.linalg.norm(mesh.extents)*.01)
    trimesh.smoothing.filter_taubin(mesh,lamb=.4,nu=.5,iterations=4)
    displacement=mesh.vertices-original
    length=np.linalg.norm(displacement,axis=1)
    mesh.vertices=original+displacement*np.minimum(1,limit/np.maximum(length,1e-12))[:,None]
    if not np.isfinite(mesh.vertices).all():raise ValueError('Non-finite cleaned mesh')
    return mesh,{'method':'bounded_taubin','iterations':4,'displacement_limit_m':limit,
                 'max_displacement_m':float(np.linalg.norm(mesh.vertices-original,axis=1).max()),
                 'components_removed':0,'semantic_repair':False}


def render_asset(xml_path,output,extent):
    os.environ.setdefault('MUJOCO_GL','egl')
    import mujoco
    model=mujoco.MjModel.from_xml_path(str(xml_path));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    output=Path(output);output.mkdir(parents=True,exist_ok=True);paths=[]
    with mujoco.Renderer(model,height=400,width=480) as renderer:
        for name,azimuth,elevation in [('front',110,-18),('overview',130,-32),('rear',-50,-25)]:
            cam=mujoco.MjvCamera();cam.lookat[:]=0;cam.distance=float(np.linalg.norm(extent)*1.65);cam.azimuth=azimuth;cam.elevation=elevation
            renderer.update_scene(data,camera=cam);path=output/(name+'.png');Image.fromarray(renderer.render()).save(path);paths.append(path)
    return paths


def asset_xml(mesh_path,texture_path,scale,transform,destination):
    root=ET.Element('mujoco');ET.SubElement(root,'compiler',angle='radian')
    visual=ET.SubElement(root,'visual');ET.SubElement(visual,'global',offwidth='480',offheight='400');ET.SubElement(visual,'headlight',ambient='.5 .5 .5',diffuse='.6 .6 .6',specular='.1 .1 .1')
    asset=ET.SubElement(root,'asset');ET.SubElement(asset,'texture',name='background',type='skybox',builtin='gradient',rgb1='.72 .76 .79',rgb2='.88 .9 .92',width='256',height='256');ET.SubElement(asset,'mesh',name='asset',file=str(Path(mesh_path).resolve()),scale=' '.join(map(str,scale)))
    ET.SubElement(asset,'texture',name='color',type='2d',file=str(Path(texture_path).resolve()))
    ET.SubElement(asset,'material',name='material',texture='color',rgba='1 1 1 1',specular='.15',shininess='.2')
    world=ET.SubElement(root,'worldbody');q=Rotation.from_matrix(transform[:3,:3]).as_quat()[[3,0,1,2]]
    ET.SubElement(world,'geom',type='mesh',mesh='asset',material='material',pos=' '.join(map(str,transform[:3,3])),quat=' '.join(map(str,q)),contype='0',conaffinity='0')
    ET.ElementTree(root).write(destination);return destination


def refine_assets(scene,project,output,object_ids=None,reuse_plan=None):
    """Save independent candidates + review; keep active geometry/poses/physics intact."""
    project,output=Path(project).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    objects=[o for o in scene.objects if o.selected and (not object_ids or o.id in object_ids)]
    if object_ids == []:raise ValueError('Select at least one asset')
    if not objects or (object_ids and {o.id for o in objects}!=set(object_ids)):raise ValueError('Invalid selected asset IDs')
    if any(o.geometry!='sam3d' or not o.mesh or not o.visual_texture for o in objects):raise ValueError('Generate SAM 3D assets first')
    original_snapshot=[o.model_dump() for o in scene.objects]
    image=Image.open(project/scene.image).convert('RGB');contexts=[];frames={};before_views={}
    for obj in objects:
        folder=output/obj.id;folder.mkdir()
        mesh=trimesh.load(project/obj.mesh,force='mesh',process=False);mesh.vertices*=obj.size
        transform,extents=canonical_frame(mesh);frames[obj.id]=(transform,extents)
        crop=image.crop(tuple(map(int,obj.bbox)));crop=ImageOps.contain(crop,(420,340),Image.Resampling.LANCZOS);canvas=Image.new('RGB',(480,400),'#eeeeee');canvas.paste(crop,((480-crop.width)//2,(400-crop.height)//2));canvas.save(folder/'reference.png')
        xml=asset_xml(project/obj.mesh,project/obj.visual_texture,obj.size,transform,folder/'before.xml')
        before_views[obj.id]=render_asset(xml,folder/'before',extents)
        contexts.append({'id':obj.id,'name':obj.name,'canonical_extents_m':extents.tolist(),'bbox':obj.bbox,'dimension_source':obj.reconstruction.get('dimension_source','assumed_or_predicted'),'special_structure_annotation':getattr(obj,'structure',None)})
    # One actual model call plans all requested assets, with no arbitrary code execution.
    sheet=Image.new('RGB',(960,len(objects)*430),'#eeeeee');draw=ImageDraw.Draw(sheet)
    for i,obj in enumerate(objects):
        sheet.paste(Image.open(output/obj.id/'reference.png'),(0,i*430+30));sheet.paste(Image.open(before_views[obj.id][1]),(480,i*430+30));draw.text((8,i*430+8),obj.id+' | reference crop / SAM 3D',fill='black')
    sheet.save(output/'input_comparison.png')
    prompt=(Path(__file__).parents[2]/'prompts/astra_asset_refinement.txt').read_text()+'\n'+json.dumps({'assets':contexts,'parameter_bounds':PARAMS,'auxiliary_views':scene.preprocessing.get('reference_views',[])},ensure_ascii=False)
    atomic_json(output/'schema.json',SCHEMA);(output/'prompt.txt').write_text(prompt)
    if reuse_plan:
        shutil.copy2(reuse_plan,output/'answer.json')
    else:
        cmd=shutil.which(os.environ.get('R2S_CODEX_BIN','codex'))
        if not cmd:raise ValueError('Codex CLI is required for Astra asset refinement')
        with (output/'astra.log').open('w') as log:
            result=subprocess.run([cmd,'exec','--model','gpt-6-astra','--sandbox','read-only','--ephemeral','--ignore-user-config','--skip-git-repo-check','-c','model_reasoning_effort="high"','--output-schema',str(output/'schema.json'),'--output-last-message',str(output/'answer.json'),'--image',str(project/scene.image),'--image',str(output/'input_comparison.png'),*[arg for view in scene.preprocessing.get('reference_views',[]) for arg in ['--image',str((project/view['image']).resolve())]],'-'],input=prompt,text=True,stdout=log,stderr=subprocess.STDOUT,cwd=output,timeout=600)
        if result.returncode:raise RuntimeError('Astra asset call failed; see astra.log')
    recipes=check_recipes(json.loads((output/'answer.json').read_text()),[o.id for o in objects]);reports=[]
    for recipe in recipes:
        obj=next(o for o in objects if o.id==recipe['id']);folder=output/obj.id;transform,extents=frames[obj.id]
        if recipe['template']=='keep':
            reports.append({'id':obj.id,'status':'kept','review_status':recipe.get('review_status','unassessed'),'recipe':recipe});continue
        cleanup=None
        if recipe['template']=='mesh_cleanup':
            mesh=trimesh.load(project/obj.mesh,force='mesh',process=False)
            mesh.vertices*=obj.size
            mesh.apply_transform(transform)
            mesh,cleanup=clean_mesh(mesh)
        else:
            mesh=build_template(recipe,extents)
            if not mesh.is_watertight:raise ValueError('Template components must be watertight')
        if not np.isfinite(mesh.vertices).all():raise ValueError('Candidate must be finite')
        mesh.export(folder/'optimized.glb')
        # Return to the original SAM asset frame; downstream pose/size conventions stay the same.
        object_mesh=mesh.copy();object_mesh.apply_transform(np.linalg.inv(transform))
        prep=prepare_visual_collision(object_mesh,folder/'optimized.obj')
        normalization=np.eye(4);normalization[:3,3]=prep['center']
        xml=asset_xml(folder/'optimized.obj',prep['texture'],prep['size'],transform@normalization,folder/'after.xml')
        after=render_asset(xml,folder/'after',extents)
        comparison=Image.new('RGB',(1440,1230),'#eeeeee');draw=ImageDraw.Draw(comparison)
        for i,name in enumerate(['front','overview','rear']):
            comparison.paste(Image.open(folder/'reference.png'),(0,30+i*400));comparison.paste(Image.open(before_views[obj.id][i]),(480,30+i*400));comparison.paste(Image.open(after[i]),(960,30+i*400))
        for x,label in [(10,'Reference crop'),(490,'SAM 3D original'),(970,'Astra-directed candidate')]:draw.text((x,8),label,fill='black')
        comparison.save(folder/'comparison.png')
        before_mesh=trimesh.load(project/obj.mesh,force='mesh',process=False)
        report={'id':obj.id,'status':'candidate_only','review_status':recipe.get('review_status','unassessed'),'cleanup':cleanup,'recipe':recipe,'before_faces':len(before_mesh.faces),'after_faces':len(mesh.faces),'finite':True,'component_surfaces_watertight':bool(mesh.is_watertight),
            'canonical_from_original_asset':transform.tolist(),'comparison':str((folder/'comparison.png').relative_to(project)),
            'candidate':{'mesh':str((folder/'optimized.obj').relative_to(project)),'visual_texture':str(prep['texture'].relative_to(project)),'collision_meshes':[str(p.relative_to(project)) for p in prep['collision']], 'size':prep['size'],'asset_center_offset':prep['center'],'canonical_glb':str((folder/'optimized.glb').relative_to(project))},
            'limitations':['Candidate approximation; measured/predicted dimension provenance remains in the source scene.','Templates use overlapping closed components; cleanup preserves source topology and may retain open surfaces.','No scene assembly, collision dynamics, or grasp qualification performed.']}
        atomic_json(folder/'asset.json',report);reports.append(report)
    assert original_snapshot==[o.model_dump() for o in scene.objects]
    summary=Image.new('RGB',(1440,430*len(reports)),'#eeeeee');draw=ImageDraw.Draw(summary)
    for i,row in enumerate(reports):
        obj=next(o for o in objects if o.id==row['id']);folder=output/obj.id
        for column,path in enumerate([folder/'reference.png',before_views[obj.id][1],folder/'after/overview.png' if row['status']!='kept' else before_views[obj.id][1]]):summary.paste(Image.open(path),(480*column,i*430+30))
        draw.text((10,i*430+8),row['id']+' | Reference / SAM 3D / Astra candidate',fill='black')
    summary.save(output/'comparison.png')
    report={'model':'gpt-6-astra','stage':'asset_only','plan_reused_from':str(reuse_plan) if reuse_plan else None,'actual_model_called':not bool(reuse_plan),'active_objects_unchanged':True,'scene_assembly_run':False,'physics_run':False,'path':str(output.relative_to(project)),'objects':reports,'comparison':str((output/'comparison.png').relative_to(project))}
    atomic_json(output/'report.json',report);scene.reports['asset_refinement']=report;return report
