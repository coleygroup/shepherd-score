# *ShEPhERD* Scoring Functions
This repository contains the code for **generating/optimizing conformers**, **extracting interaction profiles**, **aligning interaction profiles**, and **differentiably scoring 3D similarity**. It also contains modules to evaluate conformers generated with *ShEPhERD*<sup>1</sup> and other generative models.

The formulation of the interaction profile representation, scoring, alignment, and evaluations are found in our preprint [*ShEPhERD*: Diffusing shape, electrostatics, and pharmacophores for bioisosteric drug design](https://arxiv.org/abs/2411.04130). The diffusion model itself is found in a *separate* repository: [https://github.com/coleygroup/shepherd](https://github.com/coleygroup/shepherd).


<p align="center">
  <img width="200" src="./logo.svg">
</p>

<sub><sup>1</sup> *ShEPhERD*: **S**hape, **E**lectrostatics, and **Ph**armacophores **E**xplicit **R**epresentation **D**iffusion</sub>

## Table of Contents
1. [File Structure](#file-structure)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#how-to-use)
5. [Scoring and Alignment Examples](#scoring-and-alignment-examples)
6. [Evaluation Examples and Scripts](#evaluation-examples-and-scripts)
7. [Data](#data)


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
├── scripts/                                # Scripts for running evaluations
├── examples/                               # Jupyter notebook tutorials/examples 
├── tests/
├── environment.yml                         # Environment
└── README.md
```


## Requirements
An example environment can be found at `environment.yml`, however, this package should generally work where PyTorch, Open3D, RDKit, and xTB can be installed in an environment with Python >=3.8.

#### Minimum requirements for interaction profile extraction, scoring/alignment, and evaluations
```
python>=3.8
numpy>1.2,<2.0
pytorch>=1.12
mkl==2024.0 (use conda)
open3d>=0.18
rdkit>=2023.03 (newest available is recommended)
xtb>=6.6 (use conda)
scipy>=1.10
pandas>=2.0
```

<sup>Make sure that mkl is *not* 2024.1 since there is a known [issue](https://github.com/pytorch/pytorch/issues/123097) that prevents importing torch.</sup>

#### If you are coming from the *ShEPhERD* repository, you can use the same environment as described there and add the optional packages listed below, if needed.


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
py3dmol>=2.1.0
biopython>=1.84
prolif>=2.0.3
mdanalysis>=2.2.0
scikit-learn>=1.3
```

## Installation
### via PyPI
`pip install shepherd-score`

### For local development (includes `examples/`, `tests/`, etc.)
1. Clone this repo
2. Navigate to this repo's top-level directory
3. Set up the environment following the instructions above
4. Run `pip install -e .` for developer install

## Usage
The package has base functions and convenience wrappers. Scoring can be done with either NumPy or Torch, but alignment requires Torch. There are also Jax implementations for both scoring and alignment of gaussian overlap, ESP similarity, and pharmacophore similarity.

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

Full jupyter notebook tutorials/examples for extraction, scoring, and alignments are found in the [`examples`](./examples/) folder. Some minimal examples are below.

### Extraction
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

### 3D similarity scoring
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

### Alignment
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

## Evaluation Examples and Scripts

We implement three evaluations of generated 3D conformers. Evaluations can be done on an individual basis or in a pipeline. Here we show the most basic use case in the unconditional setting.


- `ConfEval`
    - Checks validity, pre-/post-xTB relaxation
    - Calculates 2D graph properties
- `ConsistencyEval`
    - Inherits from `ConfEval` and evaluates the consistency of the molecule's jointly generated interaction profiles with the true interaction profiles using 3D similarity scoring functions
- `ConditionalEval`
    - Inherits from `ConfEval` and evaluates the 3D similarity between generated molecules and the target molecule

**Note**: Evaluations can be run from any molecule's atomic numbers and positions with explicit hydrogens (i.e., straight from an xyz file).

### Examples

Full jupyter notebook tutorials/examples for evaluations are found in the [`examples`](./examples/) folder. Some minimal examples are below.

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

### Scripts
Scripts to evaluate *ShEPhERD*-generated samples can be found in the `scripts` directory.

## Data
We provide the data used for model training, benchmarking, and all *ShEPhERD*-generated samples reported in the paper at this [Dropbox link](https://www.dropbox.com/scl/fo/rgn33g9kwthnjt27bsc3m/ADGt-CplyEXSU7u5MKc0aTo?rlkey=fhi74vkktpoj1irl84ehnw95h&e=1&st=wn46d6o2&dl=0). There are comprehensive READMEs in the Dropbox describing the different folders.

## License

This project is licensed under the MIT License -- see [LICENSE](./LICENSE) file for details.

## Citation
If you use or adapt `shepherd_score` or [*ShEPhERD*](https://github.com/coleygroup/shepherd) in your work, please cite us:

```bibtex
@misc{adamsShEPhERD2024,
  title = {{{ShEPhERD}}: {{Diffusing}} Shape, Electrostatics, and Pharmacophores for Bioisosteric Drug Design},
  author = {Adams, Keir and Abeywardane, Kento and Fromer, Jenna and Coley, Connor W.},
  year = {2024},
  number = {arXiv:2411.04130},
  eprint = {2411.04130},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2411.04130},
  archiveprefix = {arXiv}
}
```
