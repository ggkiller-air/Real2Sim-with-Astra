"""Executed with an independently installed official SAM3 environment."""
import json
import os
import sys
from pathlib import Path


def main():
    import numpy as np
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    req = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    image = Image.open(req["image"]).convert("RGB")
    model = build_sam3_image_model(device="cuda", checkpoint_path=os.environ["R2S_SAM3_CHECKPOINT"],
                                  load_from_HF=False, enable_inst_interactivity=True)
    processor = Sam3Processor(model, device="cuda", confidence_threshold=req.get("threshold", .35))
    rows = []

    def save(mask, score, label):
        mask = np.asarray(mask).squeeze().astype(bool)
        ys, xs = np.where(mask)
        if len(xs) < 20:
            return
        name = f"mask_{len(rows):03d}.png"
        Image.fromarray(mask.astype("uint8") * 255).save(out.parent / name)
        rows.append({"name": label, "mask": name, "score": float(score),
                     "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]})

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        if req["mode"] == "text":
            for label in req["labels"]:
                state = processor.set_image(image)
                result = processor.set_text_prompt(state=state, prompt=label)
                masks = result["masks"].detach().float().cpu().numpy()
                scores = result["scores"].detach().float().cpu().numpy()
                for mask, score in zip(masks, scores):
                    save(mask, score, label)
        else:
            state = processor.set_image(image)
            points = req.get("points", [])
            masks, scores, _ = model.predict_inst(
                state, box=np.array(req["box"], np.float32) if req.get("box") else None,
                point_coords=np.array([p[:2] for p in points], np.float32) if points else None,
                point_labels=np.array([p[2] for p in points], np.int32) if points else None,
                multimask_output=True)
            best = int(np.argmax(scores))
            save(masks[best], scores[best], req.get("label", "选中物体"))
    out.write_text(json.dumps({"backend": "official_sam3", "objects": rows}))


if __name__ == "__main__":
    main()
