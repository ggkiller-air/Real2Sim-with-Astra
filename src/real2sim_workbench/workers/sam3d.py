"""Official SAM 3D Objects boundary; requires gated weights. Not locally qualified."""
import json
import os
import sys
from pathlib import Path


def main():
    import numpy as np
    from PIL import Image
    repo = Path(os.environ["R2S_SAM3D_REPO"]).resolve()
    sys.path.insert(0, str(repo / "notebook"))
    sys.path.insert(0, str(repo))
    from inference import Inference
    req = json.loads(Path(sys.argv[1]).read_text())
    dest = Path(sys.argv[2])
    model = Inference(os.environ["R2S_SAM3D_CONFIG"], compile=False)
    image = np.asarray(Image.open(req["image"]).convert("RGB"))
    outputs = []
    for row in req["objects"]:
        mask = np.asarray(Image.open(row["mask"]).convert("L")) > 0
        result = model(image, mask, seed=req.get("seed", 42))
        if result.get("glb") is None:
            raise RuntimeError("SAM 3D Objects did not return a mesh")
        filename = row["id"] + ".glb"
        result["glb"].export(dest.parent / filename)
        pose = {}
        for key in ("rotation", "translation", "scale"):
            if key in result:
                value = result[key]
                pose[key] = value.detach().cpu().tolist() if hasattr(value, "detach") else np.asarray(value).tolist()
        outputs.append({"id": row["id"], "mesh": filename, "raw_model_pose": pose})
    dest.write_text(json.dumps({"backend": "official_sam3d", "objects": outputs,
                               "layout_status": "raw model frame retained; world-frame conversion not qualified"}))


if __name__ == "__main__":
    main()
