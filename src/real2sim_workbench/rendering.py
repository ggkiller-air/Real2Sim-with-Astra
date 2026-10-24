"""Aspect-preserving, bounded offscreen rendering in source-image coordinates."""
import numpy as np

MAX_RENDER_SIDE = 1600


def render_size(size, max_side=MAX_RENDER_SIDE):
    w, h = map(int, size)
    if min(w,h) <= 0:raise ValueError('Image dimensions must be positive')
    scale=min(1.,max_side/max(w,h))
    return max(1,round(w*scale)),max(1,round(h*scale))


def renderer(model, size):
    import mujoco
    w,h=render_size(size)
    # Also works for historical XMLs with a fixed 960 x 720 framebuffer.
    model.vis.global_.offwidth=max(model.vis.global_.offwidth,w)
    model.vis.global_.offheight=max(model.vis.global_.offheight,h)
    return mujoco.Renderer(model,width=w,height=h)


def source_pixels(pixels, size):
    """Nearest pixel-center sampling preserves segmentation IDs and metre depth.

    Only rendered evidence is resampled. Original masks, K, and prompt coordinates
    remain untouched; reports record the actual rendering resolution.
    """
    w,h=map(int,size);rh,rw=pixels.shape[:2]
    if (rw,rh)==(w,h):return pixels.copy()
    x=np.minimum(((np.arange(w)+.5)*rw/w).astype(int),rw-1)
    y=np.minimum(((np.arange(h)+.5)*rh/h).astype(int),rh-1)
    return pixels[y[:,None],x[None,:]]
