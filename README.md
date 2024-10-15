# ShEPhERD Scoring Functions
3D interaction profiles and their differentiable similarity scoring functions used in ShEPhERD (**S**hape, **E**lectrostatics, and **Ph**armacophores **E**xplicit **R**epresentation **D**iffusion).

## Table of Contents
1. [Requirements](##requirements)
2. [Installation](##installation)
3. [File Structure](##file-structure)
4. [Usage](##how-to-use)
5. [Scoring and Alignment Examples](##scoring-and-alignment-examples)

## Requirements
#### Minimum requirements for interaction profile extraction, scoring/alignment, and evaluations
```
python>=3.8
numpy>1.2,<2.0
pytorch>=1.12
mkl==2024.0 (use conda)
open3d>=0.18
rdkit>=2023.03
xtb>=6.6 (Use conda)
```

<sub><sup>Make sure that mkl is *not* 2024.1 since there is a known [issue](https://github.com/pytorch/pytorch/issues/123097) that prevents importing torch.</sup></sub>

#### Optional software necessary for docking evaluation
```
meeko
vina==1.2.5
```
You can pip install the python bindings for Autodock Vina for the python interface. However, this also requires an installation of the executable of Autodock Vina v1.2.5: [https://vina.scripps.edu/downloads/](https://vina.scripps.edu/downloads/) and the ADFR software suite: [https://ccsb.scripps.edu/adfr/implementation/](https://ccsb.scripps.edu/adfr/implementation/).

#### Other optional packages
```
jax==0.4.26
jaxlib==0.4.26+cuda12.cudnn89
optax==0.2.2
py3dmol==2.1.0
```

## Installation
1. Clone this repo
2. Navigate to this repo's top-level directory
3. Create or use a generic conda environment and activate it
4. Install xtb with `conda install -c conda-forge xtb`
5. Run `pip install -e .` for developer install (this will automatically install numpy, pytorch, open3d, and rdkit)

## File Structure
```
.
├── shepherd_score/
│   ├── alignment_utils/                    # Alignment and rigid transformations tools
│   │   ├── pca.py
│   │   └── se3.py
│   ├── evaluations/                        # Evaluation suite
│   │   ├── pdbs/
│   │   ├── utils/
│   │   │   ├── convert_data.py
│   │   │   └── interactions.py
│   │   ├── docking.py                      # Docking evaluations
│   │   └── evaluate.py                     # Generated conformer evaluation pipelines
│   ├── pharm_utils/
│   │   ├── pharmacophore.py
│   │   ├── pharm_vec.py
│   │   └── smarts_featues.fdef             # Pharmacophore definitions
│   ├── score/                              # Scoring related functions and constants
│   │   ├── constants.py
│   │   ├── electrostatic_scoring.py
│   │   ├── gaussian_overlap.py
│   │   └── pharmacophore_scoring.py
│   ├── alignment.py
│   ├── conformer_generation.py             # Conformer generation with rdkit and xtb
│   ├── container.py                        # Molecule and MoleculePair classes
│   ├── extract_profiles.py                 # Functions to extract interaction profiles
│   ├── generate_point_cloud.py
│   ├── objective.py                        # Objective function used for REINVENT
│   └── visualize.py                        # Visualization tools
├── examples                                # Jupyter notebook tutorials/examples 
├── tests
└── README.md
```

## Usage
The package has base functions and convenience wrappers. Scoring can be done with either NumPy or Torch, but alignment requires Torch. There are also Jax implementations for both scoring and alignment of gaussian overlap and ESP similarity, but currently *not* for pharmacophores.

### Base functions
#### Conformer generation
Useful conformer generation functions are found in the `shepherd_score.conformer_generation` module.

#### Interaction profile extraction
| Interaction profile | Function |
| :------- | :------- |
| shape | `shepherd_score.extract_profiles.get_molecular_surface()` |
| electrostatics | `shepherd_score.extract_profiles.get_electrostatic_potential()` |
| pharmacophores | `shepherd_score.extract_profiles.get_pharmacophores()` |

#### Scoring
```shepherd_score.score``` contains the base scoring functions with seperate modules for those dependent on PyTorch (`*.py`), NumPy (`*_np.py`), and Jax (`*_jax.py`).

| Similarity | Function |
| :------- | :------- |
| shape | `shepherd_score.score.gaussian_overlap.get_overlap()` |
| electrostatics | `shepherd_score.score.electrostatic_scoring.get_overlap_esp()` |
| pharmacophores | `shepherd_score.score.pharmacophore_scoring.get_overlap_pharm()` |

### Convenience wrappers
- `Molecule` class
    - `shepherd_score.container.Molecule` accepts an RDKit `Mol` object (with an associated conformer) and generates user-specified interaction profiles
- `MoleculePair` class
    - `shepherd_score.container.MoleculePair` operates on `Molecule` objects and prepares them for scoring and alignment


## Scoring and Alignment Examples

Jupyter notebook tutorials/examples for extraction, scoring, and alignments are found in the `examples` folder. Some minimal exmamples are below.

Extraction of interaction profiles.

```python
from shepherd_score.conformer_generation import embed_conformer_from_smiles
from shepherd_score.conformer_generation import charges_from_single_point_conformer_with_xtb
from shepherd_score.extract_profiles import get_atomic_vdw_radii, get_molecular_surface
from shepherd_score.extract_profiles import get_pharmacophores, get_electrostatic_potential
from shepherd_score.extract_profiles import get_electrostatic_potential

# Embed conformer with RDKit and partial charges from xTB
ref_mol = embed_conformer_from_smiles('Oc1ccc(CC=C)cc1', MMFF_optimize=True)
partial_charges = charges_from_single_point_conformer_with_xtb(ref_mol)

# Radii are needed for surface extraction
radii = get_atomic_vdw_radii(ref_mol)
# `surface` is an np.array with shape (200, 3)
surface = get_molecular_surface(ref_mol.GetConformer().GetPositions(), radii, num_points=200)

# Get electrostatic potential at each point on the surface
# `esp`: np.array (200,)
esp = get_electrostatic_potential(ref_mol, partial_charges, surface)

# Pharmacophores as arrays with averaged vectors
# pharm_types: np.array (P,)
# pharm_{pos/vecs}: np.array (P,3)
pharm_types, pharm_pos, pharm_vecs = get_pharmacophores(ref_mol, multi_vector=False)
```

An example of scoring the similarity of two different molecules using 3D surface, ESP, and pharmacophore similarity metrics.

```python
from shepherd_score.score.constants import ALPHA
from shepherd_score.conformer_generation import embed_conformer_from_smiles
from shepherd_score.conformer_generation import optimize_conformer_with_xtb
from shepherd_score.container import Molecule, MoleculePair

# Embed a random conformer with RDKit
ref_mol_rdkit = embed_conformer_from_smiles('Oc1ccc(CC=C)cc1', MMFF_optimize=True)
fit_mol_rdkit = embed_conformer_from_smiles('O=CCc1ccccc1', MMFF_optimize=True)
# Local relaxation with xTB
ref_mol, _, ref_charges = optimize_conformer_with_xtb(ref_mol_rdkit)
fit_mol, _, fit_charges = optimize_conformer_with_xtb(fit_mol_rdkit)

# Extract interaction profiles
ref_molec = Molecule(ref_mol,
                     num_surf_points=200,
                     partial_charges=ref_charges,
                     pharm_multi_vector=False)
fit_molec = Molecule(fit_mol,
                     num_surf_points=200,
                     partial_charges=fit_charges,
                     pharm_multi_vector=False)

# Centers the two molecules' COM's to the origin
mp = MoleculePair(ref_molec, fit_molec, num_surf_points=200, do_center=True)

# Compute the similarity score for each interaction profile
shape_score = mp.score_with_surf(ALPHA(mp.num_surf_points))
esp_score = mp.score_with_esp(ALPHA(mp.num_surf_points), lam=0.3)
pharm_score = mp.score_with_pharm()
```

Next we show alignment using the same MoleculePair class.

```python
# Centers the two molecules' COM's to the origin
mp = MoleculePair(ref_molec, fit_molec, num_surf_points=200, do_center=True)

# Align fit_molec to ref_molec with your preferred objective function
# By default we use automatic differentiation via pytorch
surf_points_aligned = mp.align_with_surf(ALPHA(mp.num_surf_points),
                                         num_repeats=50)
surf_points_esp_aligned = mp.align_with_esp(ALPHA(mp.num_surf_points),
                                            lam=0.3,
                                            num_repeats=50)
pharm_pos_aligned, pharm_vec_aligned = mp.align_with_pharm(num_repeats=50)

# Optimal scores and SE(3) transformation matrices are stored as attributes
mp.sim_aligned_{surf/esp/pharm}
mp.transform_{surf/esp/pharm}

# Get a copy of the optimally aligned fit Molecule object
transformed_fit_molec = mp.get_transformed_molecule(
    se3_transform=mp.transform_{surf/esp/pharm}
)
```

Evaluations can be done on an individual basis or in a pipeline. Here we show the most basic use case in the unconditional setting.

```python
from shepherd_score.evaluations.evalutate import ConfEval
from shepherd_score.evaluations.evalutate import UnconditionalEvalPipeline

# ConfEval evaluates the validity of a given molecule, optimizes it with xTB,
#   and also computes various 2D graph properties
# `atom_array` np.ndarray (N,) atomic numbers of the molecule (with explicit H)
# `position_array` np.ndarray (N,3) atom coordinates for the molecule
conf_eval = ConfEval(atoms=atom_array, positions=position_array)

# Alternatively, if you have a list of molecules you want to test:
uncond_pipe = UnconditionalEvalPipeline(
    generated_mols = [(a, p) for a, p in zip(atom_arrays, position_arrays)]
)
uncond_pipe.evaluate()

# Properties are stored as attributes and can be converted into pandas df's
sample_df, global_series = uncond_pipe.to_pandas()
```

## Copyright

Copyright (c) 2024, Kento Abeywardane; Coley Research Lab
