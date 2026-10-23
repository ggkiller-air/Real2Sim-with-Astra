# Project scope

- This project implements a reusable image-to-physics reconstruction workflow. Keep changes inside this repository unless the user requests otherwise; existing DexBench, PICO and reconstruction experiments are independent projects.
- Current stage targets ordinary rigid bodies in MuJoCo. Special structures are annotations only; Isaac Lab is a later stage. SAM 3D requires gated weights and is not yet qualified. README and reports must distinguish implemented, tested and proposed capabilities.
- Keep core scene records independent of model and simulator runtimes. UI, CLI and future skills should call the same core operations.
- Preserve source-image coordinates, object identifiers, physical units, coordinate conventions and parameter provenance. Separate measured values, model predictions, category priors and assumptions.
- Treat initial penetration, support consistency, visual agreement and dynamic stability as separate checks. A successful load or short numerical smoke test is not full physical validation.
- Keep optimization hypotheses separate from observations. Do not conceal incorrect support with unexplained fixed joints or parameter changes.
- Models and simulator dependencies should be isolated and configurable. Do not commit weights, credentials, local virtual environments or generated run directories.
- Migrate existing experimental code deliberately, with its limitations and relevant evidence; do not assume previous scene results generalize.
