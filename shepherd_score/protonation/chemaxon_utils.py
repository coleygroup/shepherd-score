"""
Script to tautomerize and protonate molecules using ChemAxon's package.
REQUIRES ChemAxon license.

`tautomerize_chemaxon` functions are adapted from https://github.com/jenna-fromer/build_3d_py/
"""

import subprocess
import pandas as pd
import io
import os
from typing import List

from rdkit import Chem
from shepherd_score.protonation.protonate import remove_bad_protomers, neutralize_atoms


def tautomerize_chemaxon(
    smiles: str,
    cxcalc_exe: str,
    molconvert_exe: str,
    pH: float = 7.4,
    cutoff: float = 10,
    tautomer_limit: float = 20,
    protomer_limit: float = 20,
    neutralize: bool = True,
    verbose: bool = False,
    chemaxon_license_path: str = None,
) -> List[str]:
    """
    Tautomerize/protonate a molecule using ChemAxon's package.
    Defaults are from the ZINC22 build pipeline.

    Arguments
    ---------
    smiles : str
        SMILES string of the molecule to tautomerize/protonate.
    cxcalc_exe : str
        Path to the cxcalc executable.
    molconvert_exe : str
        Path to the molconvert executable.
    pH : float (default: 7.4)
        pH value to use for the protonation.
    cutoff : float (default: 10)
        Cutoff value to use for the tautomerization/protonation.
    tautomer_limit : float (default: 20)
        Limit for the tautomerization/protonation.
    protomer_limit : float (default: 20)
        Limit for the protomerization.
    neutralize : bool (default: True)
        Whether to neutralize the molecule before tautomerization/protonation.
    verbose : bool (default: False)
        Whether to print verbose output.
    chemaxon_license_path : str | None (default: None)
        Path to the chemaxon license file.
        If ``None``, the ``CHEMAXON_LICENSE_URL`` environment variable is used.

    Returns
    -------
    list[str]
        List of SMILES strings of the tautomers/protomers with bad charges removed.
    """
    if chemaxon_license_path is not None:
        os.environ['CHEMAXON_LICENSE_URL'] = chemaxon_license_path

    if os.environ.get('CHEMAXON_LICENSE_URL') is None:
        raise ValueError('CHEMAXON_LICENSE_URL is not set')

    # Suppress noisy ChemAxon Java logging unless in verbose mode
    _stderr = None if verbose else subprocess.DEVNULL

    if neutralize:
        smiles = Chem.MolToSmiles(neutralize_atoms(Chem.MolFromSmiles(smiles)))

    cmd1 =  f'{cxcalc_exe} -g dominanttautomerdistribution -H {pH} -C false -t tautomer-dist'
    output1 = subprocess.check_output(cmd1, shell=True, input=smiles.encode(), stderr=_stderr)

    cmd2 = f'{molconvert_exe} sdf -g -c "tautomer-dist>={tautomer_limit}" '
    output2 = subprocess.check_output(cmd2, shell=True, input=output1, stderr=_stderr)

    cmd3 = f'{cxcalc_exe} -g microspeciesdistribution -H {pH} -t protomer-dist'
    output3 = subprocess.check_output(cmd3, shell=True, input=output2, stderr=_stderr)

    cmd4 = f'{molconvert_exe} smiles -g -c "protomer-dist>={protomer_limit}" -T name:tautomer-dist:protomer-dist'
    output4 = subprocess.check_output(cmd4, shell=True, input=output3, stderr=_stderr)
    table = pd.read_csv(io.BytesIO(output4), sep='\t')
    if len(table) == 1:
        protomers = list(table['#SMILES'])
    else:
        # remove redundant SMILES, sort by score (highest first)
        table['score'] = table['tautomer-dist']*table['protomer-dist']/100
        prots = {}
        for smi, score in zip(table['#SMILES'], table['score']):
            if smi in prots:
                prots[smi] = max(score, prots[smi])
            elif score > cutoff:
                prots[smi] = score
        protomers = sorted(prots, key = lambda x: prots[x], reverse=True)

    return remove_bad_protomers(protomers)
