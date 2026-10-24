"""Native MuJoCo preview with a ready receipt, starts paused; Space toggles physics."""
import argparse,json,time
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
p=argparse.ArgumentParser();p.add_argument('scene',type=Path);p.add_argument('--ready',type=Path);args=p.parse_args()
model=mujoco.MjModel.from_xml_path(str(args.scene));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
running=False

def key(keycode):
    global running
    if keycode==32:running=not running

with mujoco.viewer.launch_passive(model,data,key_callback=key) as viewer:
    with viewer.lock():
        viewer.opt.geomgroup[3]=0
        viewer.cam.lookat[:]=np.mean(data.xpos[1:],axis=0)
        viewer.cam.distance=max(1.5,model.stat.extent*1.3);viewer.cam.azimuth=90;viewer.cam.elevation=-25
    viewer.sync()
    if args.ready:args.ready.write_text(json.dumps({'state':'ready','engine':'mujoco','xml':str(args.scene)}))
    while viewer.is_running():
        start=time.monotonic()
        if running:
            for _ in range(max(1,round((1/60)/model.opt.timestep))):mujoco.mj_step(model,data)
        viewer.sync();time.sleep(max(0,1/60-(time.monotonic()-start)))
