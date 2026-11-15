# Scoring and preparation pitfalls

## Coordinate bases and transforms

`Molecule.atom_pos` uses `Chem.RemoveHs`, which can retain isotope-labelled
hydrogen. Atomic fields use strict-heavy indices (`atomic number != 1`). A
field must use its corresponding `get_<feature>_positions()` accessor. Include
a molecule such as `[2H]OC(=O)c1ccccc1` and verify the different counts explicitly.
Neither a self-score nor rigid recovery alone proves chemical assignment:
also verify field values against their intended atom indices.

Only the fit moves. Inputs named in a mode's `points` rotate and translate;
inputs in `vectors` only rotate. An input in neither is left in place by
`Mode.pose`, so a missing entry silently scores an unposed array. Re-score
returned transforms, including canonical-store transforms, instead of checking
only the optimizer's reported scalar.

## Reductions

`Mode.overlap` must return a raw cross overlap; the base `score` normalizes it
with `similarity_from_overlaps`. `get_overlap` and `get_overlap_esp` already
return normalized scores, so do not call them from `overlap`. Use the raw
functions (`VAB_2nd_order`, `VAB_2nd_order_esp`). The ESP overlap includes the
positional Gaussian; it is not a spatially independent field.

For Gaussian Tversky, `A` is reference and `B` is fit:
`AB / (AB + a*(AA-AB) + b*(BB-AB))`. `'tversky'` is (0.95, 0.05),
`'tversky_ref'` (1, 0) and `'tversky_fit'` (0.05, 0.95). These ratios can
exceed one. Pharmacophore Tversky clamps at one and has its own guarded
derivative; `pharm_similarity` maps weights to the names the pharmacophore
scorers take. In `vol_and_surf_esp`, `similarity` changes only the shape term,
not the electrostatic agreement term.

Test a normalized overlap at identical fixed inputs for a unit self-score.
An optimized self-alignment can stop short of that value. Surface-potential
agreement and excluded-volume penalties need tests of their own definitions;
do not impose a universal unit-self-score gate on composite objectives.

## Charge and width conventions

`LAM_SCALING = COULOMB_SCALING**2`. `SurfESP._lam()` scales the user's `lam`;
atom-centered ESP and `Field` modes use it unscaled. The batched side has its
own flag, `ModeSpec.lam_scaling`. Check both so `lam` is scaled exactly once.
Typical public defaults are 0.1 for atom ESP/descriptor modes, 0.3 for surface
ESP, and 0.001 for combined volume-and-surface ESP.

Charges are computed on first use. The default `charge_model="xtb_water"` runs
xTB in implicit water at the molecule's formal charge and falls back to MMFF94
with a warning; `"xtb"` is vacuum xTB. Fukui uses the same solvent and formal
charge, plus adjacent charge states, and has no MMFF substitute. Select `charge_model="mmff"` or
inject fixed fields when testing scoring without an xTB executable.

Use `num_surf_points=None` (and no density or supplied surface) to omit surfaces
from `Molecule`; integer zero does not mean disabled. Preparing a surface also
computes its electrostatic potential and therefore accesses charges. Freeze
surface positions and features when comparing scoring implementations.

## Pharmacophores

Directional and directionless scoring can consume the same stored anchors and
types. Selecting `vol_color` does not regenerate features with different SMARTS.
Feature-generation options are separate: requesting directionless extraction
can change anchor multiplicity. Zero vectors are not a substitute for bypassing
the directional weighting. Direction weights clamp the dot product to [0, 1];
aromatic normals use its absolute value before clamping.

## Gradient and optimizer checks

Use nontrivial poses and distinct molecules. Test finite differences away from
hard exclusion-mask boundaries and other nondifferentiable points; also test
the boundary convention separately. `tests/modes/test_esp_agreement_grad.py` is an
example for the combined ESP gradient. `tests/_support/utils.py::mode_loss` gives the
exact loss `optimize_se3` minimizes.

The SE(3) helpers used by the reference commonly work in float32. Check their
dtypes before choosing finite-difference precision. Existing float32 tests use
steps around 1e-3; choose tolerances based on the actual function and scale.

`optimize_se3` stops each seed when the mean loss changes by less than 1e-5
for 10 steps, so optimized self-scores can stop short of the fixed-pose value.
Reference and accelerated seed generators and Adam implementations differ.
Fixed-pose score/gradient agreement is a stronger isolation test than demanding
identical optimized scores. Freeze all inputs, seeds, feature arrays, backend,
and numerical settings for repeatability tests; report any remaining differences.
