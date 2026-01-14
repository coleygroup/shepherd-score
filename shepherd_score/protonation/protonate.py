"""
Script to protonate SMILES strings using OpenBabel, MolScrub, or ChemAxon.

Requires (dependent on the method chosen):
- openbabel
- chemaxon
- molscrub
"""

import subprocess
import os
from typing import List, Literal
from rdkit import Chem
from copy import deepcopy

from rdkit.Chem.MolStandardize.rdMolStandardize import ChargeParent
from rdkit.rdBase import BlockLogs

# From ZINC22 build pipeline
BAD_CHARGES = [
    '[C-]',
    '[CH-]',
    '[CH2-]',
    '[C+]',
    '[CH+]',
    '[CH2+]',
    '[NH-]',
    '[c-]',
    '[cH-]',
    '[o+]',
    '[OH+]',
]


def neutralize_atoms(mol: Chem.Mol) -> Chem.Mol:
    """
    Attempts to neutralize every atom of a molecule by adding/removing hydrogens.
    It doesn't attempt to keep the formal charge of the original molecule.
    Neutralizes more than the rdMolStandardize.Uncharger.

    Below is copied from:
    https://rdkit.org/docs/Cookbook.html

    This neutralize_atoms() algorithm is adapted from Noel O'Boyle's nocharge code.
    It is a neutralization by atom approach and neutralizes atoms with a +1 or -1
    charge by removing or adding hydrogen where possible. The SMARTS pattern checks
    for a hydrogen in +1 charged atoms and checks for no neighbors with a negative
    charge (for +1 atoms) and no neighbors with a positive charge (for -1 atoms),
    this is to avoid altering molecules with charge separation (e.g., nitro groups).

    The neutralize_atoms() function differs from the `rdMolStandardize.Uncharger` behavior.
    See the MolVS documentation for Uncharger:

    https://molvs.readthedocs.io/en/latest/api.html#molvs-charge

    “This class uncharges molecules by adding and/or removing hydrogens.
    In cases where there is a positive charge that is not neutralizable,
    any corresponding negative charge is also preserved.”

    As an example, rdMolStandardize.Uncharger will not change charges on C[N+](C)(C)CCC([O-])=O,
    as there is a positive charge that is not neutralizable. In contrast, the neutralize_atoms()
    function will attempt to neutralize any atoms it can (in this case to C[N+](C)(C)CCC(=O)O).
    That is, neutralize_atoms() ignores the overall charge on the molecule, and attempts to
    neutralize charges even if the neutralization introduces an overall formal charge on the
    molecule.
    """
    mol = deepcopy(mol)
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol


def force_neutralize(mol: Chem.Mol) -> Chem.Mol:
    """
    Force neutralize a molecule by adding/removing hydrogens.
    Does not attempt to keep the formal charge of the original molecule.
    First runs `neutralize_atoms`, then runs `rdMolStandardize.ChargeParent`.
    """
    with BlockLogs():
        mol = deepcopy(mol)
        mol = neutralize_atoms(mol)
        mol = ChargeParent(mol)
    return mol


def remove_bad_protomers(protomers: List[str]) -> List[str]:
    """
    Remove protomers with bad charges from the rules in ZINC22 build pipeline.

    Parameters
    ----------
    protomers : list[str]
        List of SMILES strings of the protomers.

    Returns
    -------
    list[str]
        List of SMILES strings of the protomers with bad charges removed.
    """
    bad_charges = [c for c in BAD_CHARGES if c != '']
    passing_prots = []
    for smi in protomers:
        has_bad_charge = False
        for c in bad_charges:
            if c in smi:
                has_bad_charge = True
                break
        if not has_bad_charge:
            passing_prots.append(smi)
    return passing_prots


def protonate_smiles(smiles: str,
                     pH: float = 7.4,
                     method: Literal['openbabel', 'molscrub', 'chemaxon'] = 'molscrub',
                     *,
                     path_to_bin: str = '',
                     cxcalc_exe: str | None = None,
                     molconvert_exe: str | None = None,
                     chemaxon_license_path: str | None = None,
                     ) -> List[str]:
    """
    Protonate SMILES string with MolScrub, OpenBabel, or ChemAxon at given pH.

    ChemAxon requires `cxcalc` and `molconvert` executables to be installed with relevant license
    as input or set to the `CHEMAXON_LICENSE_URL` environment variable.

    OpenBabel workflow adapted from DockString:
    https://github.com/dockstring/dockstring/blob/main/dockstring/utils.py#L330

    Parameters
    ----------
    smiles : str
        SMILES string of molecule to be protonated.
    pH : float (default: 7.4)
        pH at which the molecule should be protonated.
    method : Literal['openbabel', 'molscrub', 'chemaxon']
        Method to use for protonation. Defaults to 'molscrub'.
    path_to_bin : str (default: '')
        Path to environment bin containing `mk_prepare_ligand.py`.
    cxcalc_exe : str | None (default: None)
        Path to cxcalc executable.
    molconvert_exe : str | None (default: None)
        Path to molconvert executable.
    chemaxon_license_path : str | None (default: None)
        Path to chemaxon license file.
        If ``None``, the ``CHEMAXON_LICENSE_URL`` environment variable is used.

    Returns
    -------
    list[str]
        List of SMILES strings of tautomers/protomers.
    """
    if method not in ['openbabel', 'molscrub', 'chemaxon']:
        raise ValueError(f'Invalid method: {method}')

    if method == 'openbabel':
        # cmd list format raises errors, therefore one string
        cmd = f'{path_to_bin}obabel -:"{smiles}" -ismi -ocan -p {pH}'
        cmd_return = subprocess.run(cmd, capture_output=True, shell=True)
        output = cmd_return.stdout.decode('utf-8')

        if cmd_return.returncode != 0:
            raise ValueError(f'Could not protonate SMILES: {smiles}')

        return [output.strip()]

    elif method == 'molscrub':
        from shepherd_score.protonation.molscrub_utils import tautomerize_molscrub
        return tautomerize_molscrub(smiles=smiles, pH=pH)

    elif method == 'chemaxon':
        if cxcalc_exe is None or molconvert_exe is None:
            raise ValueError('cxcalc_exe and molconvert_exe are required for chemaxon protonation')

        if os.environ.get('CHEMAXON_LICENSE_URL') is None or chemaxon_license_path is not None:
            raise ValueError('CHEMAXON_LICENSE_URL is not set')

        from shepherd_score.protonation.chemaxon_utils import tautomerize_chemaxon
        return tautomerize_chemaxon(
            smiles=smiles,
            pH=pH,
            cxcalc_exe=cxcalc_exe,
            molconvert_exe=molconvert_exe,
            chemaxon_license_path=chemaxon_license_path,
        )
