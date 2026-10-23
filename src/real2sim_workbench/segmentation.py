"""Candidate cleanup; policies operate on masks, not category names alone."""
from pathlib import Path
import numpy as np
from PIL import Image

AUTO_LABELS = ["small object on the table", "hammer", "brush", "fruit", "button box", "box",
               "cup", "bowl", "tray", "bottle", "spoon", "bag", "tool", "jar", "tongs", "utensil rack", "tea bag"]
EXCLUDE_LABELS = ["robot arm", "robot gripper"]


def load_mask(path):
    return np.asarray(Image.open(path).convert("L")) > 127


def duplicate(mask, previous, threshold=.65):
    for old in previous:
        if old.shape != mask.shape:
            continue
        intersection = np.count_nonzero(mask & old)
        union = np.count_nonzero(mask | old)
        if union and intersection / union >= threshold:
            return True
    return False


def clean_candidates(rows, folder, exclusion=None):
    """Prefer specific categories to a generic object/tool when masks coincide."""
    masks, accepted = [], []
    ordered = sorted(rows, key=lambda row: (row['name'] in ('small object on the table', 'tool'), -row['score']))
    for row in ordered:
        mask = load_mask(Path(folder)/row['mask'])
        area = np.count_nonzero(mask)
        if area == 0 or area > mask.size * .45:
            continue
        if exclusion is not None and np.count_nonzero(mask & exclusion) / area > .35:
            continue
        if duplicate(mask, masks):
            continue
        masks.append(mask)
        accepted.append(row)
    return accepted
