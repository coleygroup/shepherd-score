---
name: accelerate-scoring-mode
description: >-
  Add a Numba/Triton batched and screening implementation of a validated
  shepherd_score reference alignment mode using the mode and channel registries.
  Use when an existing reference mode needs accelerated execution and parity checks.
---

# Accelerate a scoring mode

Start from a tested reference mode: a `Mode` registered in
`modes.py::MODES`, its `MoleculePair.align_with_<mode>` method and its
`_ALIGN_KEYS` result slot. If those do not exist, establish them with
`design-scoring-mode` first. Work within the user's requested scope; adding a
mode does not authorize changing existing defaults or rerunning a full benchmark.

Paths below are relative to `shepherd_score/`, except tests and this skill's
references. Read the current sources rather than copying historical constants.

## Prefer existing terms and channels

`accel/_modes.py::ModeSpec` describes features, terms, reductions, weights,
seeds, steps, stopping patience, and execution options. `accel/channels.py`
describes molecular arrays, tensor attributes, coordinate bases, and storage.
Generated batched aligners, array screening, and worker dispatch consume these
registries. A mode using existing terms and channels normally needs a spec,
a public batch method, result registration, and tests, not another driver loop.

Read [seams.md](seams.md) for the implementation map. The current term families are:

| Family | Purpose |
|---|---|
| `shape` | Gaussian point-cloud overlap |
| `esp` | position-weighted scalar-field overlap |
| `color` | directionless typed-anchor overlap |
| `pharm` | directional pharmacophore overlap |
| `esp_cmp` | sampled surface-potential agreement and its pose gradient |
| `avoid` | excluded-volume penalty |

`vol_lipo`, `vol_mr`, and `vol_fukui` reuse the scalar-field term. Element
labels reuse the color kernel with element tables. Tversky changes a reduction,
not the underlying overlap kernel: each batched Tversky variant is its own
`<base>_tversky` spec, and `MoleculePairBatch._align_with` routes
`similarity=` to it. All currently registered terms contribute
their gradients, including the ESP term in combined volume-and-surface modes.

## Implementation procedure

1. Decompose the reference objective into channels and terms. Preserve the
   exact score, weight signs, normalization, empty-feature handling, and gradients.
2. Reuse existing kernels where the mathematics matches. For genuinely new
   math, add CPU and Triton implementations with matching signatures, a device
   dispatch wrapper, and the term evaluator/self-overlap support. Read
   [kernel_anatomy.md](kernel_anatomy.md) before implementing these pieces.
   Add the fused CPU closure if that execution path will be supported.
3. Add channel rows only for new data. Follow [screen_wiring.md](screen_wiring.md)
   for profile fields, storage, canonical transforms, and query-side third inputs.
4. Register a `ModeSpec`, including measured or explicitly inherited search
   settings. Copy the nearest mathematical objective, not just a similar mode
   name. Check `lam_scaling`, `pharm_style`, `channel_switch`, `seed_channel`,
   `graph_full_steps`, and CPU/graph gates individually.
5. Add `MoleculePairBatch.align_with_<mode>` in `container/_batch.py` as a thin
   wrapper over `self._align_mode(name, ...)`, following `align_with_vol_mr`.
   `_align_mode` handles backend resolution (`backend=None` is device-aware),
   registry defaults, the accelerated run and the `torch`/`jax` fallback, which
   calls `pair.align(name, ...)`. The spec name must therefore equal the
   reference name in `modes.MODES`, and its keyword parameters must be accepted
   by the reference mode. If Tversky is supported, add a `<name>_tversky` spec
   and route `similarity=` through `_align_with`. Audit effective defaults
   across reference, batch and screen calls. Do not change other modes.
6. Update `tests/modes/test_mode_registry.py`'s mode count and any necessary fixtures.
   Registry-derived tests provide coverage only if their fixtures actually
   supply the new features and required arguments. Inspect their parametrization;
   do not assume registration alone makes every test applicable.
7. Complete the applicable checks in [parity_gates.md](parity_gates.md), including
   direct rescoring of returned transforms and actual execution-path checks.

For new GPU kernels, use the project's `accel.kernels.tuning.autotune` wrapper
so frozen configuration replay remains supported. A new kernel or shape needs
new profile coverage; do not silently fall back to tuning in a strict replay.

## Boundaries that must remain explicit

- The CUDA graph uses the caller's patience without an extra margin. Combined
  ESP graph modes still use their full step budget.
- Early stopping is sub-batch-wide, based on each pair's best seed. Changing
  sub-batch membership can change stopping and scores.
- Size bucketing and adaptive fine-loop sub-batching limit working tensors;
  they do not make arbitrary in-memory pair construction memory-free.
- Kernel reuse does not prove the new blend or reduction is correct. Compare
  it with the reference at fixed poses before interpreting optimized scores.
- A reference can contain a bug. Investigate discrepancies in both paths;
  do not alter the reference merely to hide a parity failure.

Report changed APIs, validated backends, numerical tolerances, timing boundaries,
and any checks that could not run. Avoid promising a fixed throughput for a
new mode or using archived timings as acceptance thresholds.
