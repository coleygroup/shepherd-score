# Scoring modes

A scoring mode defines what two molecules are compared on. Pass its name to
`MoleculePair.align`, `MoleculePairBatch.align` or `screen`, with any
parameters as keywords:

```python
pair.align("surf_esp", lam=0.3)
batch.align("vol_color", color_weight=0.3, backend="numba")
screen(query, store, mode="vol_esp", top_k=100)
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

`esp` and `esp_combo` are accepted as aliases for `surf_esp` and
`vol_and_surf_esp`.

## Similarity

`similarity=` sets how an overlap becomes a score: `"tanimoto"` (default),
`"tversky"`, `"tversky_ref"`, `"tversky_fit"` or explicit weights `(a, b)`.
Every mode accepts it in `MoleculePair.align`; the batched backends accept it
for `vol`, `vol_esp`, `surf`, `surf_esp`, `pharm`, `vol_color`, `vol_lipo` and
`vol_and_surf_esp` (shape term only). The Similarity section of the
[accelerated API guide](accelerated_api.md) gives the weights and the
`<mode>_tversky` names that `screen` accepts.

## Backends

| Backend | Modes |
|---|---|
| PyTorch (`MoleculePair.align`) | all |
| Numba (CPU) and Triton (GPU), via `MoleculePairBatch` | all |
| `screen` / `screen_many` | all |
| JAX | `vol`, `surf`, `vol_esp`, `surf_esp`, `vol_and_surf_esp` |

The batched backends align heavy atoms only (`no_H=True`). Their seed counts,
step budgets and learning rates per mode are in `shepherd_score/accel/_modes.py`.

## Adding a mode

To define your own objective, see [Modes and Similarity](api/modes.rst) for
the `Mode` API and [Scoring mode skills](scoring_mode_skills.md) for adding a
reference mode and its Numba/Triton implementation.
