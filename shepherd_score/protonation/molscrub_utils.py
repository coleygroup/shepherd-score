"""
Script to tautomerize and protonate molecules using `molscrub` package developed by ForliLab.

https://github.com/forlilab/molscrub
"""

from typing import List

from rdkit import Chem
from rdkit.rdBase import BlockLogs

from shepherd_score.protonation.protonate import neutralize_atoms, remove_bad_protomers
from molscrub import Scrub


def tautomerize_molscrub(
    smiles: str,
    pH: float = 7.4,
    neutralize: bool = True,
) -> List[str]:
    """
    Find all tautomers/protomers of a molecule using `molscrub` package.

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule to tautomerize/protonate.
    pH : float (default: 7.4)
        pH value to use for the protonation.
    neutralize : bool (default: True)
        Whether to neutralize the molecule before scrubbing.
    chemaxon_license_path : str | None (default: None)
        Path to the chemaxon license file.
        If ``None``, the ``CHEMAXON_LICENSE_URL`` environment variable is used.

    Returns
    -------
    list[str]
        List of SMILES strings of the tautomers/protomers.
    """
    scrub = Scrub(
        ph_low=pH,
        ph_high=pH,
        skip_gen3d=True,
    )

    _log_blocker = BlockLogs()

    mol = Chem.MolFromSmiles(smiles)
    if neutralize:
        mol = neutralize_atoms(mol)
    tautomers = scrub(mol)

    del _log_blocker

    return remove_bad_protomers([Chem.MolToSmiles(tautomer) for tautomer in tautomers])
