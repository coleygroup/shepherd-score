"""Shared fixture for the legacy-contract tests.

Molecules are rebuilt from saved arrays (``tests/data/legacy_molecules.npz``) so that the
stochastic surface sampling does not change the inputs between runs.
"""
import json
from pathlib import Path

import numpy as np
from rdkit import Chem

DATA_DIR = Path(__file__).parents[1] / "data"
MOLECULES_PATH = DATA_DIR / "legacy_molecules.npz"
BASELINE_PATH = DATA_DIR / "legacy_baseline.json"

SMILES = ("CC(=O)Nc1ccc(O)cc1", "OC(=O)c1ccccc1O", "c1ccc2[nH]ccc2c1", "CCOC(=O)C1CCN(C)CC1")
NUM_SURF_POINTS = 100


def generate_molecules(path: Path = MOLECULES_PATH) -> None:
    """Embed the fixture molecules and save their geometry and interaction profiles."""
    from rdkit.Chem import AllChem
    from shepherd_score.container import Molecule

    np.random.seed(0)
    arrays, blocks = {}, []
    for i, smi in enumerate(SMILES):
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(mol, randomSeed=42 + i)
        AllChem.MMFFOptimizeMolecule(mol)
        molec = Molecule(mol, num_surf_points=NUM_SURF_POINTS, pharm_multi_vector=False)
        blocks.append(Chem.MolToMolBlock(molec.mol))
        arrays[f"surf_pos_{i}"] = molec.surf_pos
        arrays[f"surf_esp_{i}"] = molec.surf_esp
        arrays[f"charges_{i}"] = molec.partial_charges
        arrays[f"pharm_types_{i}"] = molec.pharm_types
        arrays[f"pharm_ancs_{i}"] = molec.pharm_ancs
        arrays[f"pharm_vecs_{i}"] = molec.pharm_vecs
    arrays["molblocks"] = np.array(json.dumps(blocks))
    np.savez(path, **arrays)


def load_molecules(path: Path = MOLECULES_PATH):
    """Rebuild the saved fixture molecules as ``Molecule`` objects."""
    from shepherd_score.container import Molecule

    data = np.load(path)
    blocks = json.loads(str(data["molblocks"]))
    molecs = []
    for i, block in enumerate(blocks):
        mol = Chem.MolFromMolBlock(block, removeHs=False)
        molecs.append(Molecule(
            mol,
            surface_points=data[f"surf_pos_{i}"],
            electrostatics=data[f"surf_esp_{i}"],
            partial_charges=data[f"charges_{i}"],
            pharm_types=data[f"pharm_types_{i}"],
            pharm_ancs=data[f"pharm_ancs_{i}"],
            pharm_vecs=data[f"pharm_vecs_{i}"],
        ))
    return molecs


def make_pair(i: int = 0, j: int = 1, do_center: bool = False):
    """Fresh ``MoleculePair`` between fixture molecules ``i`` (reference) and ``j`` (fit)."""
    from shepherd_score.container import MoleculePair

    molecs = load_molecules()
    pair = MoleculePair(molecs[i], molecs[j], num_surf_points=NUM_SURF_POINTS, do_center=do_center)
    return pair


def load_baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text())
