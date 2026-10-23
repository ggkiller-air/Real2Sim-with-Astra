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
    mask_counter = 0
    background = {}

    def save(mask, score, label):
        nonlocal mask_counter
        mask = np.asarray(mask).squeeze().astype(bool)
        ys, xs = np.where(mask)
        if len(xs) < 20:
            return
        name = f"mask_{mask_counter:03d}.png"
        mask_counter += 1
        Image.fromarray(mask.astype("uint8") * 255).save(out.parent / name)
        rows.append({"name": label, "mask": name, "score": float(score),
                     "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]})

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Robot masks remain separate and are never promoted to reconstruction assets.
        exclusion = np.zeros((image.height, image.width), dtype=bool)
        for path in req.get("exclusion_masks", []):
            exclusion |= np.asarray(Image.open(path).convert("L")) > 127
        if req["mode"] == "auto":
            for label in ["robot arm", "robot gripper"]:
                state = processor.set_image(image)
                result = processor.set_text_prompt(state=state, prompt=label)
                for mask in result["masks"].detach().float().cpu().numpy():
                    exclusion |= np.asarray(mask).squeeze().astype(bool)
            Image.fromarray(exclusion.astype("uint8")*255).save(out.parent/"excluded_robot.png")
            background["exclusion_mask"] = "excluded_robot.png"
            state = processor.set_image(image)
            result = processor.set_text_prompt(state=state, prompt="table top")
            table_masks = result["masks"].detach().float().cpu().numpy()
            if len(table_masks):
                table = max((np.asarray(m).squeeze().astype(bool) for m in table_masks), key=np.count_nonzero)
                Image.fromarray(table.astype("uint8")*255).save(out.parent/"table_surface.png")
                ys,xs=np.where(table)
                if len(xs)>100:
                    background["table_mask"]="table_surface.png"
                    background["table_bbox"]=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]
        if req["mode"] in ("text", "auto"):
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
    # Import this small pure-NumPy helper without importing the web runtime.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from segmentation import clean_candidates
    rows = clean_candidates(rows, out.parent, exclusion)
    out.write_text(json.dumps({"backend": "official_sam3", "objects": rows, "background": background}))


if __name__ == "__main__":
    main()
