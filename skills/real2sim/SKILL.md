---
name: real2sim
description: "Reconstruct a single-image tabletop scene with a local Real2Sim with Astra checkout: SAM3 selection, real SAM 3D rigid assets, MuJoCo physics, and Astra multiview review. Special structures remain annotations."
---

# Real2Sim with Astra

Use the user's checkout and its local API; this skill requires the implementation and installed runtimes. Read [API operations](../../docs/api.md) for request formats, [model setup](../../docs/models.md) when a backend is unavailable, and [validation evidence](../../docs/validation.json) before making capability claims.

## Working sequence

- Reuse the current single-image project and respect the user's selected objects. SAM3 automatically proposes common tabletop objects, excludes detected robot arms/grippers, and identifies the support surface. Inspect overlays because this is not exhaustive object discovery.
- For missed or ambiguous objects use positive/negative points or a box with `preview:true`, inspect the mask, then confirm with `/selection/accept` and the current revision. A box is not a reliable object hit test; the GUI selects existing instances by mask. Preserve stable IDs and user removals.
- `GET /api/status` distinguishes configured paths from a recorded local inference success. If real reconstruction is requested, use `reconstruct`; never silently turn an inference failure into proxies.
- Real `reconstruct` preserves GLB colors and model poses, shares predicted depth, fits a support plane, and builds colored visual meshes plus convex collision parts. Do **not** run the older `initialize` afterward: it would replace model layout with a bounding-box heuristic.
- Run `optimize → settle → validate → review`. Wait for each job before starting the next. Native-contact optimization and settling propose poses; independent validation checks them with unchanged thresholds. A layout lock does not weld a body in simulation.
- `review` captures the reference camera and diagnostic views, calls `gpt-6-astra`, and tests bounded XY/yaw suggestions and requested stable-support pose hypotheses. Accept only native silhouette improvements that also pass physics. Preserve rejected candidates, failed checks and the original model output. Do not change mass/friction or add fixed joints merely to conceal layout problems.
- Inspect the final native views and report both physical results and remaining visual mismatches. Deliver the exported ZIP or scene.xml and the local GUI. Unknown scale, inferred unseen surfaces, cropped objects and assumed mass/friction remain explicit; stability is not grasp/task qualification.

## Execution and provenance

Launch `bash scripts/run.sh` in the checkout; default UI/API is http://127.0.0.1:7865. UI and agent call the same backend. Never modify scene records during an active job; revisions reject stale edits. Generated requests, assets, candidate scenes, model answers and logs are under `runs/<project>/jobs/<job>/`. Do not commit user images, weights, credentials, environments or generated runs.

SAM3 and SAM 3D run locally. Astra review sends the reference image, generated views and scene summary through the user's logged-in Codex CLI; use that step when it is part of the user's requested workflow. No unrelated messages or data are needed. The prompt is in [astra_scene_review.txt](../../prompts/astra_scene_review.txt), while [review.py](../../src/real2sim_workbench/review.py) enforces and measures proposals.

Only ordinary rigid MuJoCo scenes are implemented. Hinge/drawer/cloth/soft fields do not create joints or deformables. Multi-view/depth input and Isaac Lab are future stages. Imported meshes and explicit proxy rehearsals retain their separate fallback paths and limitations.
