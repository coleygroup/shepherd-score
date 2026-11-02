# Accelerated API guide

This guide covers batched alignment, the scoring modes, library stores,
screening and parallel execution. Signatures and parameters are in the
[API reference](api/index.rst), and the objectives are defined in
[scoring theory](theory.md).

| Capability | Interface | Notes |
|---|---|---|
| Single-pair alignment | `MoleculePair.align(mode, ...)`, `align_with_<mode>` | PyTorch reference optimizer; analytic gradients where available, else autograd |
| Batched alignment | `MoleculePairBatch.align(mode, ...)`, `align_with_<mode>` | Numba CPU or Triton GPU kernels; pairs grouped by size, several starting poses per pair |
| Library stores | `ProfileStore.create`, `add`, `close`, `open` | Sharded on disk; keeps the arrays the declared modes need |
| Screening | `screen`, `screen_many` | Streams the store shard by shard and keeps a running top-K; `screen_many` reads each shard once for all queries |
| Multiple GPUs | `screen(..., ndev=...)`, `align_multi_gpu`, `MultiGPUAligner` | One worker process per GPU |
| CPU worker processes | `screen_parallel`, `screen_parallel_close` | In-memory library, persistent pool of forked workers |

## Preparing molecules

```python
from shepherd_score.conformer_generation import embed_conformer_from_smiles
from shepherd_score.container import Molecule, MoleculePair, MoleculePairBatch

def prepare(smiles):
    rd = embed_conformer_from_smiles(smiles, MMFF_optimize=True)
    return Molecule(rd, num_surf_points=200, pharm_multi_vector=False,
                    charge_model="mmff")

ref = prepare("CCOc1ccccc1")
fit = prepare("CCNc1ccccc1")
```

`Molecule` holds a conformer and its features:

- **Surfaces.** Set `num_surf_points` for surface modes. Mesh sampling is
  random, and a fixed conformer seed does not fix the surface, so save and
  reuse surface points and ESP values for reproducible comparisons.
- **Pharmacophores.** Set `pharm_multi_vector`, or pass the pharmacophore
  arrays directly. `feature_set="shepherd"` (default) or `"rdkit_base"`
  selects the SMARTS definitions.
- **Charges.** Computed on first use. The default `charge_model="xtb_water"`
  runs xTB in implicit water at the molecule's formal charge and falls back to
  MMFF94 with a warning if xTB is unavailable or fails. Use `"xtb"` for vacuum
  xTB or `"mmff"` for MMFF94, and choose one explicitly when comparing results.
- **Fukui fields** (`vol_fukui`) use the same solvent and formal charge, need
  xTB and have no MMFF94 fallback.

`embed_conformer_from_smiles(..., MMFF_optimize=True)` sanitizes the relaxed
conformer so that aromaticity is set for pharmacophore and Crippen features.
See [Molecule](api/container/molecule.rst) and
[profile containers](api/container/profiles.rst).

## Choosing a mode

| Mode | Objective |
|---|---|
| `vol`, `surf` | Gaussian overlap of heavy-atom or surface point clouds |
| `vol_esp`, `surf_esp` | Overlap weighted by atomic charges or surface potentials |
| `pharm` | Typed, directional pharmacophore overlap |
| `vol_color` | Shape plus directionless pharmacophore overlap |
| `vol_pharm` | Shape plus directional pharmacophore overlap |
| `vol_lipo` | Shape plus per-atom Crippen lipophilicity |
| `vol_mr` | Shape plus per-atom Crippen molar refractivity |
| `vol_fukui` | Shape plus condensed Fukui reactivity |
| `vol_atomtype` | Shape plus element-matched overlap |
| `vol_and_surf_esp` | Shape plus masked surface-potential agreement |
| `vol_avoid` | Shape minus an excluded-volume penalty; needs `avoid_points` |

`align_with_esp` and `align_with_esp_combo` are aliases for `surf_esp` and
`vol_and_surf_esp`.

`vol_color` ignores pharmacophore directions; `vol_pharm` and `pharm` use
them. All three read the pharmacophores already on each `Molecule`; the mode
does not re-extract them.

### Similarity

Overlap modes take `similarity=`:

| Value | Weights `(a, b)` on reference / fit self-overlap |
|---|---|
| `"tanimoto"` (default) | symmetric Tanimoto |
| `"tversky"` | `(0.95, 0.05)` |
| `"tversky_ref"` | `(1.0, 0.0)` |
| `"tversky_fit"` | `(0.05, 0.95)` |
| `(a, b)` | explicit weights |

```python
pair.align("vol", similarity="tversky")
pair.align_with_vol_color(similarity=(0.7, 0.3))   # applies to shape and color
MoleculePairBatch(pairs).align_with_vol(similarity="tversky", backend="triton")
```

A Tversky result is stored in the `<mode>_tversky` slots
(`sim_aligned_vol_tversky`, `transform_vol_tversky`, ...). On the batched
backends, `similarity=` is accepted by `vol`, `vol_esp`, `surf`, `surf_esp`,
`pharm`, `vol_color`, `vol_lipo` and `vol_and_surf_esp`; for
`vol_and_surf_esp` it applies to the shape term only. Pharmacophore Tversky
scores are clamped at one; the other Tversky scores are not.

The batch kernels and `screen` also accept the registry names
`<mode>_tversky` (for example `screen(..., mode="vol_tversky")`) with
`tversky_alpha` and `tversky_beta` keywords.

## Aligning pairs

```python
pair = MoleculePair(ref, fit, num_surf_points=200)   # needed for surface modes

res = pair.align("vol_esp", lam=0.1)            # PyTorch, returns AlignmentResult
scores, aligned = MoleculePairBatch([pair]).align_with_vol(backend="numba")
```

`MoleculePair.align` returns the result without storing it;
`align_with_<mode>` stores the score and transform on the pair.
`MoleculePairBatch` writes to the same slots on each pair and returns the
scores and, with `return_aligned=True`, the aligned coordinates. A mode can
also be passed as a `shepherd_score.modes.Mode` instance; see
[modes](api/modes.rst).

On `backend="numba"` and `"triton"`, `num_repeats`, `max_num_steps` and `lr`
default to each mode's values in
[`accel/_modes.py`](../shepherd_score/accel/_modes.py). On `"jax"` and
`"torch"` they default to 50 / 200 / 0.1. See
[MoleculePairBatch](api/container/molecule_pair_batch.rst).

### Early stopping

The step budget is a maximum. Where early stopping is enabled, the CPU loop,
the GPU loop and the GPU CUDA-graph loop all check for improvement every five
steps with the same patience. Pairs are optimized in sub-batches, and a
sub-batch stops only once none of its pairs has improved for the required
number of checks. The CUDA-graph loops for `vol_and_surf_esp` and
`vol_and_surf_esp_tversky` always run the full budget. Rounding differences
between CPU and GPU, and which pairs share a sub-batch, can change where
optimization stops and which pose is selected.

For `vol_and_surf_esp` the optimizer follows the gradient of the full reported
score, including the ESP agreement term.

## Building a library and screening it

```python
from shepherd_score.screen import ProfileStore, screen, screen_many

with ProfileStore.create("example-library.fss", num_surf_points=200,
                         modes=["vol", "surf_esp"]) as store:
    store.add(fit, id="compound-1-conformer-1")

library = ProfileStore.open("example-library.fss")
hits = screen(ref, library, mode="vol", backend="numba", top_k=1)
panel_hits = screen_many([ref, fit], library, mode="vol",
                         backend="numba", top_k=1)
for hit in hits:
    print(hit.id, hit.score, hit.transform)
```

Declare every mode the store will be screened with when creating it; only the
arrays those modes need are saved. Stores and queries must use the same
`num_surf_points`. Canonical stores rotate each molecule into its
principal-axis frame; returned transforms are in the molecule's input frame.

Hits are per conformer. For compound-level ranking, aggregate over each
compound's conformers; the paper benchmarks use the maximum score.

`screen_many` returns one hit list per query. `scores_out` collects every
score in caller-provided arrays on the single-process path. See
[screening](api/screening.rst).

## Parallel execution

For disk-backed GPU screening, pass `backend="triton", ndev=2` to `screen` or
`screen_many`. Workers start on the first call and are reused;
`close_multigpu_pool()` stops them. `worker_threads` sets the host compute
threads per GPU worker; by default the available CPUs are divided among the
GPUs. Prefetch threads are not counted in this limit.

For in-memory pairs, `shepherd_score.accel.multi_gpu.align_multi_gpu` aligns
one list across GPUs, and `MultiGPUAligner` keeps its workers for repeated
calls.

`shepherd_score.accel.screen_parallel.screen_parallel` screens a query against
an in-memory list of molecules with forked CPU workers. The pool is kept for
later calls against the same library; `screen_parallel_close()` stops it.
The first call includes pool startup, so exclude it from steady-state timing.

Guard scripts that start GPU workers with `if __name__ == "__main__":`, since
those workers are spawned. The CPU pool needs a platform that supports fork.

## Reproducible GPU launch configurations

Triton picks kernel launch configurations by timing candidates, and the
choices can vary between installations. To record the choices made during a
warm-up run:

```python
from shepherd_score.accel.kernels.tuning import export_configurations
# Run the intended workloads once, outside timing.
export_configurations("triton-configurations.json")
```

Set `FSS_TRITON_CONFIGS=/absolute/path/triton-configurations.json` before
importing the kernels to replay them, including in spawned screening workers.
A kernel or shape missing from the file, or a different GPU model,
architecture or Triton version, raises an error instead of re-tuning. The file
holds only the configurations used in the exporting process, so run every
workload shape before exporting. Files from separate runs can be combined only
if they agree on shared entries.

Replaying fixes the launch configurations; timings still depend on hardware
and cache state. Store the file with the commit, software versions, inputs and
timing results it was recorded with. Without the variable, Triton autotunes as
usual.

## Further reference

- [Installation](installation.rst): optional backends and external executables
- [Batch alignment](api/container/molecule_pair_batch.rst)
- [Modes and similarity](api/modes.rst)
- [Screening and parallel APIs](api/screening.rst)
- [Analytical gradients](api/alignment/analytical_gradients.rst)
- [Scoring theory](theory.md)
- [Measured throughput](performance/timings.md)
- [Scoring-mode development skills](scoring_mode_skills.md)
- [Paper benchmark procedures](https://github.com/Nate-Somers/Shepherd-Score-Paper/blob/main/benchmarks/README.md)

The tutorial notebooks cover the original preparation and scoring workflows,
not the batched API.
