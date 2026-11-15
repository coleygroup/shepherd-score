# Migrating from v1 to v2

ShEPhERD-score v2.0 keeps the v1 API: `Molecule`, `MoleculePair.align_with_*`, `MoleculePairBatch.align_with_*`, the `score` and `alignment` functions and the evaluation pipelines still work and store results in the same attributes. The changes below can still alter results or break existing code.

## Breaking changes

| Change | v1 | v2 | To keep v1 behavior |
|---|---|---|---|
| Python | ≥3.9 | ≥3.10 | — |
| Default partial charges (`Molecule` without `partial_charges`) | MMFF94 | gfn2-xTB in implicit water (`charge_model='xtb_water'`), falling back to MMFF94 with a warning if xTB is missing | `Molecule(..., charge_model='mmff')` |
| xTB charges for charged molecules | — | computed with the molecule's formal charge | — |
| `Molecule` option values | — | `charge_model`, `feature_set` and `surface_method` raise `ValueError` for unknown values | — |

Charge-dependent scores (`vol_esp`, `surf_esp`, `vol_and_surf_esp`) will differ from v1 unless the charges match. Surface, shape and pharmacophore scores are unaffected.

## New in v2

- **`MoleculePair.align(mode, ...)` / `score(mode, ...)`** take any [scoring mode](modes.md) by name, return an `AlignmentResult`, and leave the pair unchanged. The `align_with_*` methods remain and still write `transform_<mode>` / `sim_aligned_<mode>` on the pair.
- **`MoleculePairBatch.align(mode, backend=...)`** runs batched Numba (CPU) or Triton (GPU) kernels and, like `align_with_*`, stores results on each pair.
- **`ProfileStore` and `screen`** for library screening, with multi-GPU (`ndev=`) and CPU thread control (`cpu_threads=`). See [Accelerated API](accelerated_api.md).
- **New modes:** `vol_pharm`, `vol_color`, `vol_lipo`, `vol_mr`, `vol_fukui`, `vol_atomtype`, `vol_avoid`, `vol_color_avoid`, plus `similarity='tversky'` on every mode.
- **Install extras:** `[cpu]` adds Intel SVML for the Numba kernels; `[gpu]` adds Triton (needs a CUDA build of PyTorch ≥2.6).
- **TF32 matmuls** On CUDA, the accelerated engine (`MoleculePairBatch.align*` and `screen` on the `triton` and `numba` backends) runs its own fp32 matmuls with TF32 allowed. v2 sets this only for the duration of each engine call and then restores your setting.

## Renamed modes

`esp` and `esp_combo` are now `surf_esp` and `vol_and_surf_esp`. The old names still work as aliases: `align_with_esp`, `align_with_esp_combo`, `sim_aligned_esp` and `transform_esp` read and write the new modes' results.
