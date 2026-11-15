# Scoring modes

A scoring mode defines what the representations by which molecules are aligned and scored.
Pass its name to `MoleculePair.align`, `MoleculePairBatch.align` or `screen`, with any
parameters as keywords:

```python
pair.align("surf_esp", lam=0.3)
batch.align("vol_color", color_weight=0.3, backend="numba")
screen(query, store, mode="pharm", top_k=100)
```

## Available modes

| Mode | Compares | Needs on `Molecule` | Default parameters |
|---|---|---|---|
| `vol` | Gaussian overlap of heavy atoms | nothing extra | `alpha` (0.81) |
| `surf` | Gaussian overlap of surface points | `num_surf_points` | `alpha` |
| `vol_esp` | Atom overlap weighted by partial-charge agreement | partial charges | `lam` (0.1) |
| `surf_esp` | Surface overlap weighted by ESP agreement | `num_surf_points` | `lam` (0.3) |
| `vol_and_surf_esp` | Shape plus ShaEP-style surface ESP agreement | `num_surf_points` | `esp_weight` (0.5), `lam` (0.001), `probe_radius` (1.0) |
| `pharm` | Typed, directional pharmacophore overlap | pharmacophores | `extended_points`, `directionless` (False) |
| `vol_color` | Shape plus directionless pharmacophore overlap | pharmacophores | `color_weight` (0.5) |
| `vol_pharm` | Shape plus directional pharmacophore overlap | pharmacophores | `color_weight` (0.5) |
| `vol_lipo` | Shape plus Crippen logP field | nothing extra | `lipo_weight` (0.5), `lam` (0.1) |
| `vol_mr` | Shape plus Crippen molar-refractivity field | nothing extra | `mr_weight` (0.5), `lam` (0.1) |
| `vol_fukui` | Shape plus condensed Fukui dual-descriptor field | Fukui values (xTB, no MMFF fallback) | `fukui_weight` (0.5), `lam` (0.1) |
| `vol_atomtype` | Shape plus same-element overlap | nothing extra | `atomtype_weight` (0.5) |
| `vol_avoid` | Shape minus an excluded-volume penalty | `avoid_points` passed to the call | `avoid_weight` (1.0), `avoid_min_dist` (2.0) |
| `vol_color_avoid` | `vol_color` minus the `vol_avoid` penalty | pharmacophores, `avoid_points` passed to the call | `color_weight` (0.5), `avoid_weight` (1.0), `avoid_min_dist` (2.0) |

"Shape" is the `vol` term. In the blended modes the weight parameter sets the
non-shape share, so `color_weight=0.5` scores `0.5 * vol + 0.5 * color`.
`vol_and_surf_esp` uses volumetric shape at the default `alpha=0.81` and surface
shape otherwise.

What the "Needs" column means:

- **`num_surf_points`**: build the `Molecule` with a surface, and pass the same
  value to `MoleculePair`.
- **Partial charges**: set by `charge_model` (`"xtb_water"` by default, falling
  back to MMFF94 without xTB) or passed as `partial_charges`. The surface ESP
  modes compute their potentials from the same charges.
- **Pharmacophores**: set `pharm_multi_vector` (`False` recommended), or pass
  the pharmacophore arrays directly.
- **`avoid_points`**: a `(K, 3)` cloud (for example a pocket wall), used
  as-is in the frame the molecules are scored in. It is never shifted for you.
  `MoleculePair` defaults to `do_center=False`, so the cloud is used in your
  original coordinates. With `do_center=True`, which centers the reference,
  pass `cloud - pair.ref_shift`. `screen` centers the query on a pre-centered
  store, so pass `cloud - query.atom_pos.mean(0)` there.

`vol_color` uses whatever pharmacophores the `Molecule` holds. The default
`Molecule(feature_set="shepherd")` gives ShEPhERD features; build molecules
with `feature_set="rdkit_base"` for ROCS-style color features.

`esp` and `esp_combo` are accepted as aliases for `surf_esp` and
`vol_and_surf_esp`.

## Similarity

`similarity=` sets how an overlap becomes a score. Overlap modes accept:

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

`MoleculePair.align` accepts it for every mode. `MoleculePairBatch.align` and
`screen` accept it for `vol`, `vol_esp`, `surf`, `surf_esp`, `pharm`, `vol_color`, `vol_lipo`,
`vol_color_avoid` (shape and color, not the penalty) and `vol_and_surf_esp`
(shape term only). A Tversky result is stored in the `<mode>_tversky` slots
(`sim_aligned_vol_tversky`, `transform_vol_tversky`, ...), and the batch kernels
and `screen` also accept the equivalent registry names `<mode>_tversky` (for example
`screen(..., mode="vol_tversky")`) with `tversky_alpha` and `tversky_beta`. Other
modes raise `ValueError` for a Tversky `similarity`.
Pharmacophore Tversky scores are clamped at one; the other Tversky scores are
not. See [scoring theory](theory.md) for the definitions.

## Backends

| Backend | Modes |
|---|---|
| PyTorch (`MoleculePair.align`) | all |
| Numba (CPU) and Triton (GPU), via `MoleculePairBatch` | all |
| `screen` / `screen_many` | all |
| JAX | `vol`, `surf`, `vol_esp`, `surf_esp`, `vol_and_surf_esp`, `pharm` |

`backend="jax"` on any other mode warns and falls back to per-pair PyTorch.
The batched backends align heavy atoms only (`no_H=True`). Their seed counts,
step budgets and learning rates per mode are in
[`accel/_modes.py`](https://github.com/coleygroup/shepherd-score/blob/main/shepherd_score/accel/_modes.py).

## Adding a mode

- **Reference (PyTorch).** Subclass the `Mode` dataclass in
  [`modes.py`](https://github.com/coleygroup/shepherd-score/blob/main/shepherd_score/modes.py)
  (implement `inputs` and `overlap`, or `score`) and register the class under a name in `modes.MODES`. Once registered, the name works in
  `MoleculePair.align`. A `Mode` instance can also be passed directly without
  registering. See [Modes and Similarity](api/modes.rst).
- **Kernels (Numba/Triton).** Add a `ModeSpec` built from `Term`s in
  [`accel/_modes.py`](https://github.com/coleygroup/shepherd-score/blob/main/shepherd_score/accel/_modes.py),
  with any new per-molecule feature channels in
  [`accel/channels.py`](https://github.com/coleygroup/shepherd-score/blob/main/shepherd_score/accel/channels.py).
  The spec gives batching, `screen` and store support.
- Validate against the reference before relying on the fast path. The agent
  workflow for this is in [Scoring-mode skills](scoring_mode_skills.md).
