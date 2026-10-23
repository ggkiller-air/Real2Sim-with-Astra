---
name: real2sim
description: Use a local Real2Sim with Astra checkout to select image objects, refine SAM3 masks, edit rigid layouts, and export and check MuJoCo scenes. Special structures are annotations only in v0.1.
---

# Real2Sim with Astra

Use the user's Real2Sim-with-Astra checkout and its local API. Read README.md and docs/api.md in that checkout; docs/models.md describes model setup. This skill requires the codebase and installed dependencies, and is not itself a reconstruction implementation.

## Workflow

1. Infer object scope from the request. Reuse the current project and preserve necessary support surfaces.
2. Inspect GET /api/status. Configured paths do not establish successful inference. Explain unavailable backends and continue independent work.
3. Upload/open the image. Run SAM3 text candidates or instance prompts and inspect overlays. Preserve original-image coordinates and object IDs; allow user corrections in the UI.
4. Obtain geometry through SAM 3D or mesh import. Proxy boxes require an explicit prototype/layout-rehearsal choice; never silently replace failed reconstruction with proxies.
5. Edit sizes, poses, provenance and support. v0.1 uses box-bottom projection to a plane, not recovered depth. Hinge/drawer/cloth/soft tags do not create those structures.
6. Save the current revision. Initialize or optimize as appropriate; initialization replaces unlocked poses. Layout locks do not fix bodies in simulation. Inspect the result against the source.
7. Validate and examine initial penetration, drift, rotation, velocities, support contacts, warnings and simulated time. Preserve failures. Do not tune friction or add fixed joints merely to hide incorrect support.
8. Export the current revision and provide the scene ZIP and precise validation conclusions. Distinguish rigid numerical checks from visual accuracy, true parameters and task success.

## Execution

Run `bash scripts/run.sh` from the checkout. Default service: http://127.0.0.1:7865. UI and agent use the same API. Poll returned jobs to done/failed, then refetch the scene. Never edit during active jobs. Edits invalidate old reports.

Requests, model outputs and logs are stored in runs/<project>/jobs/<job>/. Keep credentials, weights and user images outside Git by default.

## Current boundaries

SAM 3D is unqualified until actual inference succeeds; raw model poses are preserved but their world-frame conversion is not applied. Imported meshes have convex-hull collisions, with no cavities. Only ordinary rigid MuJoCo scenes are implemented; Isaac Lab and special structures are future work.
