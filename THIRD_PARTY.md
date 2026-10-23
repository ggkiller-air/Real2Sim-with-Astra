# Third-party components

- Three.js `0.180.0`: vendored browser modules in `src/real2sim_workbench/web/vendor/`, distributed under MIT. Original copyright and license are retained in `THREE-LICENSE.txt`. Source: https://github.com/mrdoob/three/tree/r180
- SAM3: external runtime; no model weights or upstream code copied into the tracked application. https://github.com/facebookresearch/sam3
- SAM 3D Objects: optional external runtime and ignored source checkout; upstream code and weights remain subject to the SAM License. https://github.com/facebookresearch/sam-3d-objects
- REST3D and φ-Scene are research references only. Their code is not incorporated into this implementation. The current layout heuristic is authored here and is not a reproduction of either paper.
- Python runtime dependencies and their versions are listed in `environments/core.lock.txt`; each dependency retains its own license.

The project author's own software license has not yet been selected. Public repository visibility does not by itself grant a redistribution license for that code.
