"""Local desktop preview launcher: fixed executable/paths, no client commands."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
from .store import atomic_json


class PreviewManager:
    def __init__(self,store):self.store=store;self.processes={};self.lock=threading.RLock()

    def status(self,sid,engine):
        with self.lock:
            item=self.processes.get((sid,engine))
            if not item:return {'state':'closed'}
            process=item['process'];code=process.poll()
            if code is not None:return {'state':'closed' if code==0 else 'failed','error':None if code==0 else '预览启动失败，请查看 '+str(item['log']),'exit_code':code}
            if not item['ready'].is_file() and time.monotonic()-item['started']>180:
                process.terminate()
                return {'state':'failed','error':'预览启动超时，请查看 '+str(item['log'])}
            if item['ready'].is_file():return {'state':'ready','pid':process.pid}
            return {'state':'starting','pid':process.pid}

    def launch(self,scene,engine):
        if engine not in ('mujoco','isaac'):raise ValueError('Unknown preview engine')
        if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):raise ValueError('当前服务没有桌面显示环境；请在本机图形桌面启动服务')
        with self.lock:
            state=self.status(scene.id,engine)
            if state['state'] in ('starting','ready'):return state
            report=scene.reports.get('export' if engine=='mujoco' else 'isaac')
            if not report:raise ValueError('请先生成对应模拟器场景')
            path=self.store.asset(scene.id,report['path']+('/scene.xml' if engine=='mujoco' else '/scene.usdc'))
            folder=self.store.path(scene.id)/'jobs'/('preview_'+engine+'_'+uuid.uuid4().hex[:8]);folder.mkdir(parents=True)
            ready=folder/'ready.json';log=folder/'preview.log'
            env=os.environ.copy();env.pop('MUJOCO_GL',None)
            if engine=='mujoco':
                command=[sys.executable,str(Path(__file__).with_name('preview_worker.py')),str(path),'--ready',str(ready)]
                env['MUJOCO_GL']='glfw'
            else:
                from .isaac import runtime_python,worker_path
                command=[str(runtime_python()),str(worker_path()),'--view',str(path),'--ready',str(ready)]
            with log.open('w') as stream:
                process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            self.processes[(scene.id,engine)]={'process':process,'ready':ready,'log':log,'started':time.monotonic()}
            atomic_json(folder/'launch.json',{'engine':engine,'scene':str(path),'pid':process.pid})
            return self.status(scene.id,engine)
