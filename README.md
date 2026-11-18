# ShEPhERD-score

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/shepherd-score.svg)](https://pypi.org/project/shepherd-score/)
[![Python versions](https://img.shields.io/pypi/pyversions/shepherd-score.svg)](https://pypi.org/project/shepherd-score/)
[![Documentation Status](https://readthedocs.org/projects/shepherd-score/badge/?version=latest)](https://shepherd-score.readthedocs.io/en/latest/?badge=latest)


📄 **[Paper](https://openreview.net/forum?id=KSLkFYHlYg)** | 📚 **[Documentation](https://shepherd-score.readthedocs.io/en/latest/)** | 📦 **[PyPI](https://pypi.org/project/shepherd-score/)**

</div>

ShEPhERD-score is a Python package for **3D molecular similarity**. It **extracts interaction profiles** (shape, surface electrostatics and pharmacophores), **aligns and scores** molecules with differentiable 3D similarity functions, **screens** compound libraries at high throughput, and **evaluates** 3D molecules sampled from generative models such as *ShEPhERD* and *ShEPhERD–2*.

> [!IMPORTANT]
> **New in v2.0 (Nov. 2026)**
>
> Register a scoring mode once and use it across backends. Numba (CPU) and Triton (GPU) accelerate batched alignment and library screening, with multi-GPU support for large screens. [See benchmark timings](https://shepherd-score.readthedocs.io/en/latest/performance/timings.html).
>
> **Upgrading from v1?** v2.0 requires Python ≥3.10 and changes the default partial charges used for electrostatic scoring. [Read the migration notes](https://shepherd-score.readthedocs.io/en/latest/migration.html).

The interaction profile representation and scoring functions were introduced with the *ShEPhERD* diffusion model ([ICLR 2025](https://openreview.net/forum?id=KSLkFYHlYg)). The generative models live in separate repositories: [ShEPhERD](https://github.com/coleygroup/shepherd) and [ShEPhERD–2](https://github.com/coleygroup/shepherd2).

<p align="center">
  <img width="200" src="./logo.svg">
</p>

## Documentation

Full documentation is available at [shepherd-score.readthedocs.io](https://shepherd-score.readthedocs.io/en/latest/).

## Installation

To install the basic tooling for interaction profile extraction, scoring, alignment, and evaluation suites, install via PyPI.

```bash
uv pip install shepherd-score
```

xTB is required for ShEPhERD–2 workflows (xTB partial charges and ESP). xTB can be installed through conda (`conda install conda-forge::xtb`) in a conda environment or from [source](https://xtb-docs.readthedocs.io/en/latest/setup.html) and adding it to `PATH`. We recommend the latter.

### Accelerators

The Numba CPU backend is included in the base install and needs no extra. The optional `cpu` extra adds Intel SVML, which speeds up a few less common CPU paths; see [Intel SVML](https://shepherd-score.readthedocs.io/en/latest/installation.html#intel-svml) for whether you need it.

```bash
uv pip install "shepherd-score[cpu]"   # optional, x86-64 Linux only
```

Alternatively, `conda env create -f environment.yml` builds a CPU environment with SVML, xTB and Vina.

For GPU-accelerated scoring and alignment backends with Triton (needs a CUDA build of PyTorch ≥2.6 which should be installed separately for your device):

```bash
uv pip install "shepherd-score[gpu]"
```


### Extra features
JAX support: `[jax]`. Vina docking: `[docking]`. Everything, including dev and docs tooling: `[all]`. See the [installation docs](https://shepherd-score.readthedocs.io/en/latest/installation.html) for external executables (xTB, Vina, ChemAxon).


## Usage

1. [Interaction profile extraction](#interaction-profile-extraction)
2. [Molecular alignment and scoring](#molecular-alignment-and-scoring)
3. [Accelerated alignment](#accelerated-alignment)
4. [Screen a library](#screen-a-library)
5. [Evaluation of generated samples](#evaluation-of-generated-samples)

### Interaction profile extraction

```python
from shepherd_score.conformer_generation import embed_conformer_from_smiles
from shepherd_score.container import Molecule, MoleculePair, MoleculePairBatch

ref_mol = embed_conformer_from_smiles('Oc1ccc(CC=C)cc1')
# `Molecule` extracts and owns all interaction profiles
ref_molec = Molecule(
    ref_mol,
    num_surf_points=200,        # omit to skip the surface
    charge_model='xtb_water',   # partial charges for the ESP, computed lazily
    pharm_multi_vector=False,   # recommended: one HBA vector per carbonyl, not two
)

# surface point cloud + electrostatic potential
surf_pos = ref_molec.surface.positions # np.array (200,3)
esp = ref_molec.surface.esp # np.array (200,)

# unpacks as (types, positions, vectors)
# ref_molec.pharmacophore.types: np.array (P,)
# ref_molec.pharmacophore.{positions/vectors}: np.array (P,3)
pharm = ref_molec.pharmacophore
```

### Molecular alignment and scoring

Alignment using a volumetric scoring function with default PyTorch.

```python
ref = Molecule(embed_conformer_from_smiles("CCOc1ccccc1"))
fit = Molecule(embed_conformer_from_smiles("CCNc1ccccc1"))
pair = MoleculePair(ref, fit)

result = pair.align("vol")
result.score                 # np.ndarray, scalar: optimal vol similarity
result.transform             # np.ndarray (4, 4): fit -> ref SE(3)
result.aligned["points"]     # np.ndarray (N, 3): fit heavy atoms after transform
```

> [!NOTE]
> `MoleculePair.align()` (new in v2.0) returns an `AlignmentResult` and leaves the `MoleculePair` object unchanged. The v1 `align_with_*()` methods still exist in v2.0 and store the score and transform as attributes on the pair, as does `MoleculePairBatch.align()` below.

Swap `"vol"` for any scoring mode. The main ones:

| Mode | Compares | Needs on `Molecule` |
|---|---|---|
| `vol` | Heavy-atom shape | nothing extra |
| `surf` | Surface shape | `num_surf_points` |
| `vol_esp` | Shape weighted by partial charges | partial charges |
| `surf_esp` | Surface shape weighted by ESP | `num_surf_points` |
| `vol_and_surf_esp` | Shape plus surface ESP agreement | `num_surf_points` |
| `pharm` | Directional ShEPhERD-style pharmacophores | pharmacophores |
| `vol_color` | Shape plus directionless RDKit pharmacophore features | pharmacophores (`feature_set='rdkit_base'`) |
| `vol_lipo` | Shape plus Crippen logP field | nothing extra |
| `vol_avoid` | Shape minus an excluded-volume penalty | `avoid_points` passed to the call |
| `vol_color_avoid` | `vol_color` minus an excluded-volume penalty | pharmacophores, `avoid_points` passed to the call |

Modes also take `similarity="tversky"` for asymmetric scoring. Shape blends with molar refractivity, Fukui and atom-type terms are also available. See [scoring modes](https://shepherd-score.readthedocs.io/en/latest/modes.html) for the full list, parameters and backend support.

### Accelerated alignment

Align many molecules to a reference with an accelerated backend: `"numba"` (CPU), `"triton"` (GPU) or `"jax"`.

```python
fits = [Molecule(embed_conformer_from_smiles(s)) for s in ["CCNc1ccccc1", "OCc1ccccc1"]]
batch = MoleculePairBatch([MoleculePair(ref, f) for f in fits])

scores, aligned = batch.align("vol", backend="numba")
scores   # np.ndarray (n_pairs,)
aligned  # list of aligned fit coordinates; None entries unless return_aligned=True
```

### Screen a library

```python
from shepherd_score.screen import ProfileStore, screen

with ProfileStore.create("library.fss", num_surf_points=0, modes=("vol",)) as store:
    for i, rdmol in enumerate(library_rdkit_mols):
        store.add(Molecule(rdmol), id=f"cmpd-{i}")

hits = screen(ref, ProfileStore.open("library.fss"), mode="vol", top_k=100)
```

A store saves only the arrays needed by the modes it was created for. See [screening](https://shepherd-score.readthedocs.io/en/latest/api/screening.html) for additional examples for storing arrays for multiple scoring modes or multi-cpu/gpu workflows.

### Evaluation of generated samples

We implement three evaluations of 3D molecules sampled by generative models. `UnconditionalEvalPipeline` evaluates molecules generated unconditionally, `ConsistencyEvalPipeline` compares the consistency of ShEPhERD's jointly generated interaction profiles with the true interaction profile, and `ConditionalEvalPipeline` additionally evaluates the 3D similarity between generated molecules and a target molecule.

Full jupyter notebook tutorials/examples for evaluations are found in the [examples](./examples/) folder. Some minimal examples are below.

```python
from shepherd_score.evaluations.evaluate import ConfEval
from shepherd_score.evaluations.evaluate import UnconditionalEvalPipeline

# `atom_array` np.ndarray (N,) atomic numbers of the molecule (with explicit H)
# `position_array` np.ndarray (N,3) atom coordinates for the molecule
conf_eval = ConfEval(atoms=atom_array, positions=position_array)

# Alternatively, if you have a list of molecules you want to test:
uncond_pipe = UnconditionalEvalPipeline(
    generated_mols = [(a, p) for a, p in zip(atom_arrays, position_arrays)],
    expected_charge = 1,  # optional: formal charge to try first when inferring bonds (default tries 0, +1, -1, +2, -2)
)
uncond_pipe.evaluate(num_workers=4)

# Properties are stored as attributes and can be converted into pandas DataFrames
global_series, sample_df = uncond_pipe.to_pandas()
```

## License

This project is licensed under the MIT License -- see [LICENSE](./LICENSE) file for details.

## Citation

An article describing ShEPhERD-score v2.0 is forthcoming. If you use or adapt ShEPhERD-score in your work, please cite it once available:

```bibtex
Citation forthcoming.
```

The interaction profile representation and scoring functions were introduced with *[ShEPhERD](https://github.com/coleygroup/shepherd)*. If you use them or the ShEPhERD models, please also cite:

```bibtex
@inproceedings{
adams2025shepherd,
title={Sh{EP}h{ERD}: Diffusing shape, electrostatics, and pharmacophores for bioisosteric drug design},
author={Keir Adams and Kento Abeywardane and Jenna Fromer and Connor W. Coley},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=KSLkFYHlYg}
}
```
