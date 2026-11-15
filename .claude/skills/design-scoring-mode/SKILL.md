---
name: design-scoring-mode
description: >-
  Implement a new molecular similarity objective and a tested PyTorch reference
  alignment method in shepherd_score from a mathematical or plain-language description.
  Use for adding scoring modes; use accelerate-scoring-mode for their Numba/Triton backend.
---

# Design a scoring mode

Paths are under `shepherd_score/` unless they start with `tests/`. Read the
current sources before choosing an extension point; this file describes the
architecture but does not replace reading it.

A reference mode is a frozen dataclass in `modes.py`. It declares the arrays it
reads from a `Molecule` and how it scores a posed fit against the reference.
Seeding, the SE(3) optimizer (`alignment/_optimize.py::optimize_se3`), backend
selection and result handling are shared. A new mode is usually one class or
one `Combined` blend, a registry entry, and a thin `align_with_<mode>` wrapper.

## Establish the objective

Check whether the request is already covered:

- `modes.py::MODES` lists the reference modes (13 names).
- `accel/_modes.py::SPECS` lists the batched modes (21, including the
  `<base>_tversky` registry variants).
- `similarity=` (`score/similarity.py`) gives Tanimoto, the three named Tversky
  settings or explicit `(a, b)` weights on any overlap mode. A Tversky variant
  of an existing mode is a `similarity` argument, not a new mode.

If an existing mode or parameter does what is asked, explain how to use it
instead of adding a duplicate.

Write the score as a function of a fixed reference and a transformed fit.
Specify inputs, units, normalization, weights and which terms steer the
optimizer. State whether the objective is symmetric and what its self-score
should be. Penalties and the surface-potential agreement in `vol_and_surf_esp`
do not have the normalization properties of a Gaussian Tanimoto overlap.

A third input does not make a mode pair-only: `VolAvoid` takes a fixed
`avoid_points` cloud as a dataclass field, and screening passes it as a
query-side keyword (`screen.py::_with_avoid`).

## Implement the reference

Read [seams.md](seams.md) for the file map and [pitfalls.md](pitfalls.md)
before adding a feature or reduction.

1. **Reuse before writing.** If the objective is an existing overlap on
   different inputs, or a weighted sum of existing modes, build it from those:
   `vol_mr` and `vol_fukui` are `Field` with a different `field`; `vol_pharm`
   is `vol_color(directionless=False)`; `blend(shape, other, w, label)` gives
   `(1 - w) * shape + w * other`; `Combined` takes any weighted terms
   (negative weights for penalties).
2. **New mathematics** goes in a `Mode` subclass:
   - `@dataclass(frozen=True)`; parameters (`alpha`, `lam`, weights,
     `similarity`) are fields. Use `eq=False` if a field holds an array.
   - `name: ClassVar[str]`, and `requires` (`'surface'`, `'pharmacophores'`)
     so `MoleculePair` raises a clear error when features are missing.
   - `inputs(mol)` returns a dict of numpy arrays.
   - `points` / `vectors` (ClassVars) name the inputs the pose translates and
     rotates, or only rotates. `seed` names the input used for COM/PCA seeds.
   - Implement `overlap(a, b)` as a raw cross overlap. The base `score` applies
     `similarity_from_overlaps` with self overlaps from `prepare`. Override
     `score` (and `prepare`) only when the result is not an overlap ratio.
   - Put new scoring math in `score/`, built from raw overlaps, not from an
     already normalized score.
   - `backends()` may return specialized `'analytical'` or `'jax'` optimizers.
     Without them the mode runs on autograd through `optimize_se3`, which is
     the expected starting point.
3. **New molecular property.** Add it to `Molecule` in with-H RDKit atom order,
   with a strict-heavy getter and matching positions (see the per-atom feature
   contract in seams.md). For fields read by `Field`, add an entry to
   `modes.py::_FIELDS`. Tests should inject fixed fields instead of running xTB
   or building surfaces where possible.
4. **Register and expose.**
   - Add the name to `MODES`, with a factory function if it is a blend. The
     name must match the batched registry name used later, because
     `MoleculePairBatch` falls back to `pair.align(name, ...)` on
     `backend="torch"`/`"jax"`.
   - Add a result slot to `container/_core.py::_ALIGN_KEYS`. The
     `transform_<mode>` / `sim_aligned_<mode>` properties are generated from it.
     Add `<mode>_tversky` too if Tversky results need their own slot
     (`_similarity_slot` picks it).
   - Add `MoleculePair.align_with_<mode>` as a wrapper over
     `self._align_and_store(slot, mode_instance, ...)`, following
     `align_with_vol_mr`. Defaults are 50 repeats, lr 0.1 and 200 steps.
5. **Keep it out of `accel/_modes.py`** until batched terms and channels exist.
   `accelerate-scoring-mode` adds the `ModeSpec` and batch method.

Do not change existing defaults or other modes' behavior when adding one. The
JAX and NumPy scorers are separate implementations: support and test a
requested backend, or state that the new mode is PyTorch-only.

## Validate

Reference-mode gates are shared, parametrized tests; add rows, not a new file.
The tests must cover:

- the fixed-pose and optimized self-score on identical molecules:
  `tests/test_modes.py::test_every_mode_scores_one_on_itself` runs every entry
  in `MODES` under Tanimoto and Tversky. Make sure its `_self_pair` fixture
  supplies the new mode's features, or extend it;
- autograd against finite differences at a non-identity pose on distinct
  inputs, and recovery of a planted rigid transform with the returned transform
  rescored by `mode.score`: add the mode (and any Tversky variant) to `CASES`
  in `tests/test_reference_modes.py`. Its inputs come from `mode.inputs`, so a
  per-atom field the library cannot compute without external tools must be
  injected in `_molecule` (as the synthetic Fukui field is);
- field/position correspondence on a molecule with a retained isotope-labelled
  H, if the mode reads a per-atom field: add it to
  `test_retained_h_field_follows_heavy_atoms` and `test_retained_h_self_alignment`;
- reference/fit asymmetry for Tversky reductions: add a substructure pair to
  `test_tversky_asymmetry_fits_inside`.

A mode whose inputs cannot be built that way (surfaces, caller-supplied clouds)
gets its own file; [template_test.py](template_test.py) lays out the same gates.

Base tolerances on the nearest existing tests (`tests/test_reference_modes.py`,
`tests/test_modes.py`). A unit self-score alone does
not show that features or gradients are correct. If the reference and a later
accelerated version disagree, investigate both.

Run `tests/test_reference_modes.py`, `tests/test_modes.py`, `tests/test_mode_registry.py`
and any new test file.
Mark GPU tests (or GPU parameters) `cuda`; `tests/conftest.py` skips them without CUDA.
Report which backends were and were not tested.

## Handoff

Give `accelerate-scoring-mode` the objective, the `Mode` class and registered
name, its terms and reductions, feature arrays and their coordinate bases, any
third inputs, defaults, tests and tolerances.
