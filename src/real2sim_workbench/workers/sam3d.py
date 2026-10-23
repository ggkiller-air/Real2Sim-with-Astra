"""Official SAM 3D boundary. Preserve generated appearance and native camera pose."""
import json
import os
import sys
from pathlib import Path


def main():
    import numpy as np
    import torch
    from PIL import Image
    repo = Path(os.environ['R2S_SAM3D_REPO']).resolve()
    toolkit = Path(os.environ.get('R2S_CUDA_HOME', repo.parent/'cuda121'))
    os.environ['CONDA_PREFIX'] = str(toolkit)
    os.environ.setdefault('ATTN_BACKEND', 'xformers')
    os.environ.setdefault('SPCONV_ALGO', 'native')
    sys.path[:0] = [str(repo/'notebook'), str(repo)]
    from inference import Inference
    from pytorch3d.transforms import quaternion_to_matrix
    from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
    req = json.loads(Path(sys.argv[1]).read_text())
    dest = Path(sys.argv[2])
    def progress(stage, **extra):
        (dest.parent/'progress.json').write_text(json.dumps({'stage':stage,**extra}))
    progress('正在加载三维模型')
    print('Loading SAM 3D Objects', flush=True)
    model = Inference(os.environ['R2S_SAM3D_CONFIG'], compile=False)
    image = np.asarray(Image.open(req['image']).convert('RGB'))
    rgba = np.dstack([image, np.full(image.shape[:2],255,dtype=np.uint8)])
    progress('正在估计共同深度与相机')
    with torch.inference_mode():
        depth = model._pipeline.compute_pointmap(rgba)
        pointmap = depth['pointmap'].permute(1,2,0)
        np.savez_compressed(dest.parent/'camera_geometry.npz',
                            pointmap=pointmap.cpu().numpy(), intrinsics=depth['intrinsics'].cpu().numpy())
    outputs=[]
    for index,row in enumerate(req['objects']):
        progress('正在生成物体',current=index+1,total=len(req['objects']),object_id=row['id'])
        print(f'Generating {index+1}/{len(req["objects"])}: {row["id"]}',flush=True)
        mask=np.asarray(Image.open(row['mask']).convert('L'))>0
        result=model(image,mask,seed=req.get('seed',42),pointmap=pointmap)
        if result.get('glb') is None:
            raise RuntimeError('SAM 3D Objects did not return a mesh')
        filename=row['id']+'.glb'
        result['glb'].export(dest.parent/filename)
        pose={key:result[key].detach().cpu().tolist() for key in ('rotation','translation','scale')}
        # Match the official get_mesh / Transform3d convention exactly.
        transform=compose_transform(scale=result['scale'],rotation=quaternion_to_matrix(result['rotation']),translation=result['translation'])
        native=result['glb'].vertices @ np.array([[1,0,0],[0,0,-1],[0,1,0]]).T
        camera_vertices=transform.transform_points(torch.as_tensor(native,dtype=torch.float32,device='cuda')[None])[0].detach().cpu().numpy()
        camera_mesh=result['glb'].copy()
        camera_mesh.vertices=camera_vertices
        camera_name=row['id']+'_camera.glb'
        camera_mesh.export(dest.parent/camera_name)
        outputs.append({'id':row['id'],'mesh':filename,'camera_mesh':camera_name,'raw_model_pose':pose})
        dest.write_text(json.dumps({'backend':'official_sam3d','objects':outputs,'camera_geometry':'camera_geometry.npz',
                                    'complete':len(outputs)==len(req['objects']), 'camera_frame':'pytorch3d_x_left_y_up_z_forward'}))
        del result
        torch.cuda.empty_cache()
    print('SAM 3D generation complete',flush=True)

if __name__=='__main__':
    main()
