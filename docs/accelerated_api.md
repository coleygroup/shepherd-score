# Accelerated API guide

This guide covers batched alignment, library stores, screening and parallel
execution. The scoring modes, similarity options and backend support are in
[Scoring modes](modes.md). Signatures are in the [API reference](api/index.rst)
and the objectives in [scoring theory](theory.md).

`MoleculePair.align` runs the PyTorch reference optimizer on one pair.
`MoleculePairBatch` runs Numba (CPU) or Triton (GPU) kernels on many pairs,
`ProfileStore` holds a library on disk, and `screen` / `screen_many` stream it
against one or several queries, optionally across GPUs.

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

## Aligning pairs

```python
pair = MoleculePair(ref, fit, num_surf_points=200, do_center=True)   # surface modes need num_surf_points

res = pair.align("vol_esp", lam=0.1)            # PyTorch, returns AlignmentResult
scores, aligned = MoleculePairBatch([pair]).align_with_vol(backend="numba")
```

`MoleculePair.align` returns an `AlignmentResult` and leaves the pair unchanged;
`MoleculePairBatch.align` and `align_with_<mode>` store `transform_<mode>` and
`sim_aligned_<mode>` on each pair, and return the scores and, with
`return_aligned=True`, the aligned coordinates. See [modes](modes.md) for mode
names and [Modes and Similarity](api/modes.rst) for `Mode` instances.

On `backend="numba"` and `"triton"`, `num_repeats`, `max_num_steps` and `lr`
default to each mode's values in
[`accel/_modes.py`](https://github.com/coleygroup/shepherd-score/blob/main/shepherd_score/accel/_modes.py). On `"jax"` and
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
score in caller-provided arrays on the single-process path.

The avoid modes take `avoid_points=`; see [Scoring modes](modes.md) for the
frame it must be in (for `screen`, centered on the query). See
[screening](api/screening.rst).

## Parallel execution

```python
# Disk-backed library on several GPUs: one worker process per GPU
hits = screen(ref, library, mode="vol", backend="triton", ndev=2, top_k=100)
from shepherd_score.screen import close_multigpu_pool
close_multigpu_pool()          # stop the workers when done

# In-memory pairs on several GPUs
from shepherd_score.accel.multi_gpu import align_multi_gpu
align_multi_gpu(pairs, "vol", ndev=2)       # results written back to each pair

# CPU: shard a batch over forked numba worker processes
MoleculePairBatch(pairs).align_with_vol(backend="numba", num_workers=8)
```

Workers for `ndev` start on the first call and are reused;
`MultiGPUAligner(pairs, ndev=2)` keeps its workers for repeated `align` calls.
`shepherd_score.accel.screen_parallel.screen_parallel(query, library, mode,
n_workers=...)` screens an in-memory list of molecules with a persistent pool of
forked CPU workers (stop it with `screen_parallel_close()`); the first call
includes pool startup, so exclude it from steady-state timing. Guard scripts
that start worker processes (`ndev`, `num_workers`) with
`if __name__ == "__main__":`, since those workers are spawned.
`screen_parallel` needs a platform that supports fork.

### CPU threads

On the numba backend, the kernels use one thread per CPU the process may run on.
On a shared node, cap that with `cpu_threads`:

```python
batch = MoleculePairBatch(pairs, cpu_threads=8)
hits = screen(query, store, mode="vol_color_avoid", backend="numba", cpu_threads=8, ...)
```

The cap applies for the duration of each call. It can be set up to numba's
thread pool size, which is fixed when numba is first imported, from
`NUMBA_NUM_THREADS` or else the process's CPU affinity (so `taskset`,
`--cpus-per-task` and `--cpuset-cpus` cap it; a time quota such as Docker
`--cpus` does not). With `ndev > 1` use `worker_threads` (host threads per GPU
worker, default CPUs divided among GPUs); with `num_workers > 1` each worker is
single-threaded and `cpu_threads` is rejected. Logical CPUs count SMT threads,
which add little for these arithmetic-bound kernels; see
[timings](performance/timings.md).

## Reproducible GPU launch configurations

Triton picks launch configurations by timing candidates, so choices can vary
between installations. To record and replay them:

```python
from shepherd_score.accel.kernels.tuning import export_configurations
# Run every intended workload shape once, outside timing.
export_configurations("triton-configurations.json")
```

Set `FSS_TRITON_CONFIGS=/absolute/path/triton-configurations.json` before
importing the kernels (spawned workers inherit it). A kernel or shape missing
from the file, or a different GPU model, architecture or Triton version, raises
an error instead of re-tuning. Timings still depend on hardware and cache
state, so store the file with the commit, software versions and inputs.

## Further reference

- [Installation](installation.rst): optional backends and external executables
- [Batch alignment](api/container/molecule_pair_batch.rst)
- [Scoring modes](modes.md) and [Modes and similarity](api/modes.rst)
- [Screening and parallel APIs](api/screening.rst)
- [Analytical gradients](api/alignment/analytical_gradients.rst)
- [Scoring theory](theory.md)
- [Measured throughput](performance/timings.md)
- [Scoring-mode development skills](scoring_mode_skills.md)
- [Benchmark procedures](https://github.com/Nate-Somers/Shepherd-Score-Benchmarks)

The [screening tutorial](tutorials/04_screening.ipynb) walks through the batched
and screening API; the other tutorial notebooks cover the original preparation
and scoring workflows.
