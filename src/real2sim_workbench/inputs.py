"""Local input library, independent of generated project storage."""
import json
from pathlib import Path

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}


class InputLibrary:
    def __init__(self, root):
        self.root = Path(root).resolve()
        for kind in ('image', 'depth'):
            (self.root/kind).mkdir(parents=True, exist_ok=True)

    def catalog(self):
        images = [p for p in (self.root/'image').rglob('*')
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and p.resolve().is_relative_to(self.root/'image')]
        captures = [p for p in (self.root/'depth').iterdir()
                    if p.is_dir() and p.resolve().is_relative_to(self.root/'depth') and (p/'capture_metadata.json').is_file()]
        return {'root': str(self.root), 'image': sorted(str(p.relative_to(self.root/'image')) for p in images),
                'depth': sorted(p.name for p in captures)}

    def resolve(self, kind, name):
        if kind not in ('image', 'depth'):
            raise ValueError('输入类型必须是 image 或 depth')
        base = self.root/kind
        path = (base/name).resolve()
        if not name or not path.is_relative_to(base) or path == base:
            raise ValueError('只能导入 data 下当前类型的文件')
        if kind == 'image':
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError('请选择有效的图片')
        else:
            if not path.is_dir():
                raise ValueError('深度输入必须是一个完整采集文件夹')
            meta = json.loads((path/'capture_metadata.json').read_text())
            names = ['image_external_rgb.jpg', 'capture_metadata.json', meta['d455_depth']['file']]
            names += [row['file'] for row in meta['rgb'].values()]
            if (path/'d455_depth_m.npy').exists():names.append('d455_depth_m.npy')
            for name in names:
                target = (path/name).resolve()
                if not target.is_relative_to(path) or not target.is_file():
                    raise ValueError('采集文件缺失或指向采集目录以外：'+name)
            if any(p.is_symlink() for p in path.iterdir()):
                raise ValueError('采集目录请使用实际文件，不要使用符号链接')
        return path
