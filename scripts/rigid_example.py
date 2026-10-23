"""Synthetic rigid example. Does not exercise image reconstruction models."""
import argparse
import io
from pathlib import Path
from PIL import Image, ImageDraw
from real2sim_workbench.store import Store
from real2sim_workbench.models import ObjectRecord
from real2sim_workbench.physics import export_scene, validate_scene

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=Path('runs/rigid-example'))
args = parser.parse_args()
im = Image.new('RGB', (640, 480), '#e8eee4')
draw = ImageDraw.Draw(im)
draw.rectangle([120, 160, 240, 300], fill='#cf805d')
draw.rectangle([360, 180, 460, 300], fill='#5f9985')
draw.text((25, 25), 'SYNTHETIC RIGID EXAMPLE - NOT IMAGE RECONSTRUCTION', fill='#263c30')
buffer = io.BytesIO()
im.save(buffer, format='PNG')
store = Store(args.output/'projects')
scene = store.create(buffer.getvalue(), 'Synthetic rigid example')
scene.objects = [ObjectRecord(id='o_000000000001', name='Synthetic block A', bbox=[120,160,240,300], geometry='proxy', position=[-.15,0,.0502]),
                 ObjectRecord(id='o_000000000002', name='Synthetic block B', bbox=[360,180,460,300], geometry='proxy', position=[.15,0,.0502])]
store.save(scene)
output = args.output/'simulation'
xml = export_scene(scene, store.path(scene.id), output)
report = validate_scene(scene, xml, output, seconds=3)
print(f'{xml.resolve()}\nphysics_check_pass={report["physics_check_pass"]}; synthetic_example=True')
if not report['physics_check_pass']:
    raise SystemExit(1)
