# Reference implementation map

Paths are under `shepherd_score/` unless they start with `tests/`.

| Concern | Implementation |
|---|---|
| Mode classes, blends, `MODES`, `get_mode` | `modes.py` |
| Shared autograd SE(3) optimizer, `tree_map` | `alignment/_optimize.py::optimize_se3` |
| Analytic-gradient optimizers (vol, ESP, pharm, avoid) | `alignment/_torch_analytical.py`, `score/analytical_gradients/` |
| Tanimoto / Tversky reduction | `score/similarity.py` |
| Gaussian raw overlap, hard-sphere penalty | `score/gaussian_overlap.py` |
| ESP overlap and surface-potential agreement | `score/electrostatic_scoring.py` |
| Typed directional overlap | `score/pharmacophore_scoring.py` |
| Element-identity overlap | `score/atomtype_scoring.py` |
| Molecule features, `MoleculePair.align/score`, `align_with_*` | `container/_core.py` |
| Result slots | `_ALIGN_KEYS`, `_alignment_property`, `_similarity_slot`, `AlignmentResult` in `container/_core.py` |
| Batched mode definitions (acceleration only) | `accel/_modes.py` |

`score/__init__.py` is empty; import score modules directly.

## How an alignment runs

`MoleculePair.align(mode, **params)`:

1. `get_mode(name, **params)` builds the mode (or accepts a `Mode` instance).
2. `_mode_inputs` checks `mode.requires`, then converts `mode.inputs(mol)` for
   both molecules to tensors with `tree_map`. A `Combined` mode gives a tuple
   with one dict per term.
3. The backend is `'analytical'` if `mode.backends()` has one, otherwise
   `'torch'` (`optimize_se3`). `backend=` can force either, or `'jax'`.
4. `optimize_se3` seeds from `mode.seed_points` (or `trans_centers` with
   `trans_init`), poses the fit with `mode.pose`, and minimizes
   `1 - mode.score(ref, posed_fit, cache)`; `cache` comes from `mode.prepare`.
5. It returns an `AlignmentResult(score, transform, aligned)`.

`align_with_<mode>` calls `_align_and_store(slot, mode, ...)`, which runs
`align` and writes the score and transform to that slot.

`tests/_support/utils.py::mode_loss(mode, se3, ref, fit)` is the loss `optimize_se3`
minimizes, for finite-difference tests.

## Reuse examples

| Mode | Built from |
|---|---|
| `vol`, `surf` | `_Shape.overlap` on atom or surface points |
| `vol_esp`, `surf_esp` | `_ESP`; surface ESP scales `lam` by `LAM_SCALING` in `_lam()` |
| `vol_avoid` | `Vol` minus `avoid_weight * Avoid` |
| `vol_color`, `vol_pharm` | `blend(Vol, Pharm(directionless=...))` |
| `vol_lipo`, `vol_mr`, `vol_fukui` | `blend(Vol, Field(field=...))` |
| `vol_atomtype` | `blend(Vol, AtomType)` |
| any Tversky variant | the same mode with `similarity=` |

`tests/modes/test_modes.py::HeavyAtomCount` is a minimal `Mode` subclass defined
outside the package.

## Per-atom feature contract

Follow `get_lipophilicity_contribs`, `get_lipophilicity` and
`get_lipo_positions` in `container/_core.py`:

1. Store the full field in with-H RDKit atom order. Cheap descriptors can be
   eager; expensive ones can be cached lazy properties.
2. Compute it from the same molecule and conformer as its positions.
3. Slice with `_nonH_atoms_idx` for `no_H=True`.
4. Take matching positions from the with-H conformer at the same strict-heavy
   indices (`modes.py::_heavy_positions`). Do not use `atom_pos`, which comes
   from `RemoveHs`.

The matching `MoleculeProfile` accessors in `screen.py` let the batched readers
use stored data. A new field needs store support when it is accelerated.

## Registration and defaults

- `modes.MODES`: names accepted by `MoleculePair.align`.
- `_ALIGN_KEYS`: per-pair result slots.
- `accel/_modes.SPECS`: executable batched modes. Do not add a spec for kernels
  or channels that do not exist yet.

Reference and batched defaults differ: `align_with_*` and `align` use 50
repeats, lr 0.1 and 200 steps, while each `ModeSpec` has its own seeds, steps
and lr (for example `vol_pharm`: 32 seeds, 50 steps). Use explicit settings in
comparisons. On the batched backends `num_repeats`, `max_num_steps` and `lr`
set to `None` take the registry values.
