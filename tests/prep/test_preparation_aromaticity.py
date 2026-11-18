"""MMFF preparation must preserve the chemical typing seen after an SDF round trip."""
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from shepherd_score.conformer_generation import embed_conformer, generate_conformer_ensemble
from shepherd_score.generate_point_cloud import get_atom_coords, get_electrostatics


@pytest.mark.parametrize('route', ['single', 'ensemble', 'coordinates', 'electrostatics'])
def test_mmff_returns_rdkit_aromaticity(route):
    mol = Chem.MolFromSmiles('O=C1NC(=O)N(C2CC(O)C(CO)O2)C=C1C=CBr')
    if route == 'single':
        result = embed_conformer(mol, MMFF_optimize=True, random_seed=42)
    elif route == 'ensemble':
        result = generate_conformer_ensemble(Chem.AddHs(mol), num_confs=1, num_threads=1)[0]
    elif route == 'coordinates':
        result, _ = get_atom_coords(mol, MMFF_optimize=True)
    else:
        result = embed_conformer(mol, random_seed=42)
        potential = get_electrostatics(result, np.array([[20., 20., 20.]]))
        assert np.isfinite(potential).all()
    back = Chem.MolFromMolBlock(Chem.MolToMolBlock(result), removeHs=False)
    assert sum(a.GetIsAromatic() for a in result.GetAtoms()) == 6
    assert [a.GetIsAromatic() for a in result.GetAtoms()] == [a.GetIsAromatic() for a in back.GetAtoms()]
    np.testing.assert_allclose(rdMolDescriptors._CalcCrippenContribs(result),
                               rdMolDescriptors._CalcCrippenContribs(back), rtol=0, atol=0)
