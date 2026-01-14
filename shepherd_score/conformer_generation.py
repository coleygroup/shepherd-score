"""
Handles anything related to generating conformers with xTB or MMFF94.

Requires xtb installation with command-line access.
See https://xtb-docs.readthedocs.io/en/latest/setup.html for installation instructions.
"""
import os
from copy import deepcopy
import re
import shutil
import subprocess
import time
from pathlib import Path
from tqdm import tqdm
import uuid
from typing import Optional, List
import contextlib
import multiprocessing

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from rdkit.Chem import rdMolAlign
from rdkit.ML.Cluster import Butina # type: ignore

TMPDIR = os.environ.get('TMPDIR', '')
if TMPDIR == '' and Path('/tmp').is_dir():
    TMPDIR = '/tmp'
TMPDIR = Path(TMPDIR)


@contextlib.contextmanager
def set_thread_limits(num_threads: int):
    """Temporarily set threading environment variables."""
    env_vars = [
        'OMP_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS',
        'NUMEXPR_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS',
    ]
    old_env = {var: os.environ.get(var) for var in env_vars}
    try:
        for var in env_vars:
            os.environ[var] = str(num_threads)
        yield
    finally:
        for var, val in old_env.items():
            if val is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = val


def update_mol_coordinates(mol: Chem.Mol, coordinates) -> Chem.Mol:
    """
    Update the coordinates of a 3D RDKit mol object with a new set of coordinates.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit mol object with 3D coordinates to be replaced.
    coordinates : list or array-like
        List/array of new [x, y, z] coordinates.

    Returns
    -------
    Chem.Mol
        RDKit mol object with updated 3D coordinates.
    """
    mol_new = deepcopy(mol)
    conf = mol_new.GetConformer()
    for i in range(mol_new.GetNumAtoms()):
        x,y,z = coordinates[i]
        conf.SetAtomPosition(i, Point3D(x,y,z))
    return mol_new

def read_multi_xyz_file(file_dir: str):
    """
    Read an xyz file that potentially contains multiple structures.

    Parameters
    ----------
    file_dir : str
        Path to .xyz file.

    Returns
    -------
    all_coordinates : list
        List of lists containing the coordinates of each structure in the xyz file.
    all_elements : list
        List of lists containing the element types of each atom in each structure.
    """
    atom_types = [Chem.GetPeriodicTable().GetElementSymbol(i) for i in range(1, 119)]
    with open(file_dir, 'r') as file:
        lines = file.readlines()
        all_coordinates = []
        all_elements = []

        begin_coord = False
        coordinates = []
        elements = []
        for line in lines:
            stripped_line = (' '.join(line.split()))
            stripped_line_split = stripped_line.split(' ')
            numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", stripped_line)
            if len(stripped_line) == 0:
                continue

            if (len(numbers) == 3) & (stripped_line_split[0] in atom_types):
                begin_coord = True
                x,y,z = numbers
                coordinates.append([float(x),float(y),float(z)])
                elements.append(stripped_line_split[0])
            else:
                if begin_coord:
                    all_coordinates.append(coordinates)
                    all_elements.append(elements)
                    elements = []
                    coordinates = []
                begin_coord = False

    if begin_coord:
        all_coordinates.append(coordinates)
        all_elements.append(elements)

    return all_coordinates, all_elements


def embed_conformer(mol: Chem.Mol, attempts: int=50, MMFF_optimize: bool=False):
    """
    Embed a mol object into a 3D RDKit mol object with ETKDG (and optional MMFF94).

    Parameters
    ----------
    mol : Chem.Mol
        RDKit Mol object.
    attempts : int, optional
        Number of embedding attempts. Default is 50.
    MMFF_optimize : bool, optional
        Whether to optimize embedded conformer with MMFF94. Default is ``False``.

    Returns
    -------
    Chem.Mol or None
        RDKit mol object with 3D coordinates, or ``None`` if embedding fails.
    """
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, maxAttempts = attempts)
        if MMFF_optimize:
            AllChem.MMFFOptimizeMolecule(mol)

        mol.GetConformer() # test whether conformer generation succeeded
    except Exception:
        return None

    return mol


def embed_conformer_from_smiles(smiles: str, attempts: int = 50, MMFF_optimize: bool = False):
    """
    Embed a SMILES into a 3D RDKit mol object with ETKDG (and optionally MMFF94).

    Parameters
    ----------
    smiles : str
        SMILES string of molecule.
    attempts : int, optional
        Number of embedding attempts. Default is 50.
    MMFF_optimize : bool, optional
        Whether to optimize embedded conformer with MMFF94. Default is ``False``.

    Returns
    -------
    Chem.Mol or None
        RDKit mol object with 3D coordinates, or ``None`` if embedding fails.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as e:
        print('Error in SMILES string when embedding molecule:', e)
        return None
    mol = embed_conformer(mol, attempts, MMFF_optimize)
    return mol


def conf_to_mol(mol, conf_id):
    """
    Convert a conformer of a RDKit mol object into its own RDKit mol object.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit mol object with multiple conformers.
    conf_id : int
        ID of conformer to be converted into its own mol object.

    Returns
    -------
    Chem.Mol
        Mol object with only 1 conformer (the selected conformer).
    """
    conf = mol.GetConformer(conf_id)
    new_mol = Chem.Mol(mol)
    new_mol.RemoveAllConformers()
    new_mol.AddConformer(Chem.Conformer(conf))
    return new_mol


def generate_conformer_ensemble(mol_3d: Chem.Mol,
                                num_confs: int=100,
                                num_threads: int = 4,
                                threshold: float = 0.25,
                                num_opt_steps: int = 200):
    """
    Use ETKDG algorithm to embed multiple conformers from a given 3D conformer template.

    Optionally optimizes each embedded conformer with MMFF94.

    Parameters
    ----------
    mol_3d : Chem.Mol
        RDKit mol object with 3D coordinates.
    num_confs : int, optional
        Maximum number of conformers to be embedded with ETKDG. Default is 100.
    num_threads : int, optional
        Number of processors to be used in parallel when embedding conformers. Default is 4.
    threshold : float, optional
        RMSD threshold used to eliminate redundant conformers after ETKDG embedding.
        Default is 0.25.
    num_opt_steps : int, optional
        Number of MMFF94 optimization steps. Default is 200.

    Returns
    -------
    list
        List of mol objects, each containing 1 (unique) conformer.
    """
    mol_3d = deepcopy(mol_3d)

    cids = AllChem.EmbedMultipleConfs(
        mol_3d,
        clearConfs=True,
        numConfs=num_confs,
        pruneRmsThresh = threshold,
        maxAttempts = 50,
        numThreads = num_threads,
    )

    if num_opt_steps > 0:
        for cid in cids:
            AllChem.MMFFOptimizeMolecule(
                mol_3d,
                confId=cid,
                mmffVariant='MMFF94',
                maxIters = num_opt_steps,

            )
    mols = [conf_to_mol(mol_3d, c) for c in cids]

    return mols


def cluster_conformers_butina(conformers, threshold: float = 0.2, num_max_conformers: int = 100):
    """
    Cluster a list of conformers by their pairwise RMSD with Butina Clustering algorithm.

    Parameters
    ----------
    conformers : list
        List of rdkit mol objects containing conformers of a common molecule to be clustered.
    threshold : float, optional
        Initial RMSD threshold for clustering. Default is 0.2.
    num_max_conformers : int, optional
        Maximum number of conformers in the final clustered ensemble. Default is 100.

    Returns
    -------
    list
        List of int indices of the centroids of each cluster, to be indexed into conformers.
    """
    dists = []
    for i, conformer_i in enumerate(conformers):
        for j in range(i):
            mol_i = Chem.RemoveHs(deepcopy(conformer_i))
            mol_j = Chem.RemoveHs(deepcopy(conformers[j]))
            dists.append(rdMolAlign.GetBestRMS(mol_i, mol_j))

    clusts = Butina.ClusterData(dists, len(conformers), threshold, isDistData=True, reordering=True)

    if isinstance(num_max_conformers, int):
        while len(clusts) > num_max_conformers:
            threshold += 0.1
            clusts = Butina.ClusterData(dists, len(conformers),
                                        threshold, isDistData=True, reordering=True)

    select_confs = [c[0] for c in clusts] # picking centroids of each cluster

    return select_confs


def optimize_conformer_with_xtb(conformer: Chem.Mol,
                                solvent: Optional[str] = None,
                                num_cores: int = 1,
                                charge: int = 0,
                                temp_dir: str | Path = TMPDIR):
    """
    Use external calls to GFN2-XTB (command line) to optimize a conformer geometry.

    Parameters
    ----------
    conformer : Chem.Mol
        RDKit mol object containing 3D coordinates.
    solvent : str, optional
        Implicit solvent to be used during optimization. Must be a solvent supported
        by XTB (https://xtb-docs.readthedocs.io/en/latest/gbsa.html). Default is ``None``.
    num_cores : int, optional
        Number of CPU cores to be used in the xTB geometry optimization. Default is 1.
    charge : int, optional
        Molecular charge. Default is 0.
    temp_dir : str or Path, optional
        Temporary directory for I/O. Default is the system temporary directory.

    Returns
    -------
    tuple
        (xtb_mol, energy, charges) - tuple of optimized RDKit mol object,
        xTB energy (in Hartrees), and partial charges (in e-).
    """
    mol = deepcopy(conformer)

    with set_thread_limits(num_cores):
        # rand = str((os.getpid() * int(time.time())) % 123456789)
        rand = str(uuid.uuid4()) + ''.join(str(time.time()).split('.')[1])
        out_dir = Path(f'temp_xtb_opt_{rand}/')
        try:
            if temp_dir != '':
                if isinstance(temp_dir, str):
                    temp_dir = Path(temp_dir)
                out_dir = temp_dir / out_dir
            out_dir.mkdir(exist_ok=True)

            input_file = 'input_mol.xyz'
            Chem.rdmolfiles.MolToXYZFile(mol, str(out_dir/input_file))

            if solvent is not None:
                subprocess.check_call(
                    ['xtb', input_file, '--opt', '--alpb', solvent, '--parallel', str(num_cores), '--chrg', str(charge)],
                    cwd = out_dir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )
            else:
                subprocess.check_call(
                    ['xtb', input_file, '--opt', '--parallel', str(num_cores), '--chrg', str(charge)],
                    cwd = out_dir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )

            xtb_coords_list, xtb_elements_list = read_multi_xyz_file(out_dir/'xtbopt.xyz')
            xtb_coords = xtb_coords_list[0]
            xtb_mol = update_mol_coordinates(mol, xtb_coords)

            with open(out_dir/'xtbopt.xyz', 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if 'energy' in line:
                        numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)
                        energy = float(numbers[0])
                        break

            with open(out_dir/'charges', 'r') as file:
                lines = file.readlines()
                charges = [0]*len(lines)
                for i, line in enumerate(lines):
                    charges[i] = float(line.split()[0])
                    xtb_mol.GetAtomWithIdx(i).SetProp('charge', str(charges[i]))

            xtb_mol.SetProp("energy", str(energy))

        finally:
            if out_dir.exists():
                shutil.rmtree(out_dir)

        return (xtb_mol, energy, charges)


def optimize_conformer_with_xtb_from_xyz_block(xyz_block: str,
                                               solvent: Optional[str] = None,
                                               num_cores: int = 1,
                                               charge: int = 0,
                                               temp_dir: str | Path = TMPDIR):
    """
    Use external calls to GFN2-XTB (command line) to optimize coordinates from an xyz block.

    Parameters
    ----------
    xyz_block : str
        String of an xyz block.
    solvent : str, optional
        Implicit solvent to be used during optimization. Must be a solvent supported
        by XTB (https://xtb-docs.readthedocs.io/en/latest/gbsa.html). Default is ``None``.
    num_cores : int, optional
        Number of CPU cores to be used in the xTB geometry optimization. Default is 1.
    charge : int, optional
        Molecular charge. Default is 0.
    temp_dir : str or Path, optional
        Temporary directory for I/O. Default is the system temporary directory.

    Returns
    -------
    tuple
        (xtb_xyz_block, energy, charges) - tuple of optimized xyz block string,
        xTB energy (in Hartrees), and partial charges (in e-).
    """
    with set_thread_limits(num_cores):
        # rand = str((os.getpid() * int(time.time())) % 123456789)
        rand = str(uuid.uuid4()) + ''.join(str(time.time()).split('.')[1])
        out_dir = Path(f'temp_xtb_opt_{rand}/')
        try:
            if temp_dir != '':
                if isinstance(temp_dir, str):
                    temp_dir = Path(temp_dir)
                out_dir = temp_dir / out_dir
            out_dir.mkdir(exist_ok=True)

            input_file = 'input_mol.xyz'
            with open(out_dir/input_file, 'w') as f:
                f.write(xyz_block)

            if solvent is not None:
                subprocess.check_call(
                    ['xtb', input_file, '--opt', '--alpb', solvent, '--parallel', str(num_cores), '--chrg', str(charge)],
                    cwd = out_dir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )
            else:
                subprocess.check_call(
                    ['xtb', input_file, '--opt', '--parallel', str(num_cores), '--chrg', str(charge)],
                    cwd = out_dir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )

            opt_xyz_path = out_dir/'xtbopt.xyz'
            if opt_xyz_path.is_file():
                with open(out_dir/'xtbopt.xyz', 'r') as file:
                    xtb_xyz_block = file.readlines()
                    for line in xtb_xyz_block:
                        if 'energy' in line:
                            numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)
                            energy = float(numbers[0])
                            break

                if 'energy' in xtb_xyz_block[1]:
                    xtb_xyz_block[1] = '\n' # Replace energy line with blank
                xtb_xyz_block = ''.join(xtb_xyz_block)
            else:
                xtb_xyz_block = None
                energy = None

            with open(out_dir/'charges', 'r') as file:
                lines = file.readlines()
                charges = [0]*len(lines)
                for i, line in enumerate(lines):
                    charges[i] = float(line.split()[0])

        finally:
            if out_dir.exists():
                shutil.rmtree(out_dir)

        return (xtb_xyz_block, energy, charges)


def charges_from_single_point_conformer_with_xtb(conformer: Chem.Mol,
                                                 solvent: Optional[str] = None,
                                                 num_cores: int = 1,
                                                 charge: int = 0,
                                                 temp_dir: str | Path = TMPDIR
                                                 ):
    """
    Compute atomic partial charges from a single point xTB calculation of a provided conformer.

    Uses external calls to GFN2-XTB (command line).

    Parameters
    ----------
    conformer : Chem.Mol
        RDKit mol object containing 3D coordinates.
    solvent : str, optional
        Implicit solvent to be used during calculation. Must be a solvent supported
        by XTB (https://xtb-docs.readthedocs.io/en/latest/gbsa.html). Default is ``None``.
    num_cores : int, optional
        Number of CPU cores to be used in the xTB calculation. Default is 1.
    charge : int, optional
        Molecular charge. Default is 0.
    temp_dir : str or Path, optional
        Temporary directory for I/O. Default is the system temporary directory.

    Returns
    -------
    list
        List of partial charges for each atom (in e-).
    """

    mol = deepcopy(conformer)

    with set_thread_limits(num_cores):
        # rand = str((os.getpid() * int(time.time())) % 123456789)
        rand = str(uuid.uuid4()) + ''.join(str(time.time()).split('.')[1])
        out_dir = Path(f'temp_xtb_opt_{rand}/')
        try:
            if temp_dir != '':
                if isinstance(temp_dir, str):
                    temp_dir = Path(temp_dir)
                out_dir = temp_dir / out_dir
            out_dir.mkdir(exist_ok=True)

            input_file = 'input_mol.xyz'
            Chem.rdmolfiles.MolToXYZFile(mol, str(out_dir/input_file))

            if solvent is not None:
                subprocess.check_call(
                    ['xtb', input_file, '--scc', '--alpb', solvent, '--parallel', str(num_cores), '--chrg', str(charge)],
                    cwd = out_dir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )
            else:
                subprocess.check_call(
                    ['xtb', input_file, '--scc', '--parallel', str(num_cores), '--chrg', str(charge)],
                    cwd = out_dir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )

            with open(out_dir/'charges', 'r') as file:
                lines = file.readlines()
                charges = [0]*len(lines)
                for i, line in enumerate(lines):
                    charges[i] = float(line.split()[0])

        finally:
            if out_dir.exists():
                shutil.rmtree(out_dir)

        return charges


def single_point_xtb_from_xyz(xyz_block: str,
                              solvent: Optional[str] = None,
                              num_cores: int = 1,
                              charge: int = 0,
                              temp_dir: str | Path = TMPDIR):
    """
    Compute energy and atomic partial charges from a single point xTB calculation.

    Uses external calls to GFN2-XTB (command line).

    Parameters
    ----------
    xyz_block : str
        String of xyz block representing a molecule.
    solvent : str, optional
        Implicit solvent to be used during calculation. Must be a solvent supported
        by XTB (https://xtb-docs.readthedocs.io/en/latest/gbsa.html). Default is ``None``.
    num_cores : int, optional
        Number of CPU cores to be used in the xTB calculation. Default is 1.
    charge : int, optional
        Molecular charge. Default is 0.
    temp_dir : str or Path, optional
        Temporary directory for I/O. Default is the system temporary directory.

    Returns
    -------
    energy : float
        xTB energy in Hartrees.
    charges : list
        List of partial charges for each atom (in e-).
    """

    with set_thread_limits(num_cores):
        # rand = str((os.getpid() * int(time.time())) % 123456789)
        rand = str(uuid.uuid4()) + ''.join(str(time.time()).split('.')[1])
        out_dir = Path(f'temp_xtb_opt_{rand}/')
        try:
            if temp_dir != '':
                if isinstance(temp_dir, str):
                    temp_dir = Path(temp_dir)
                out_dir = temp_dir / out_dir
            out_dir.mkdir(exist_ok=True)

            input_file = 'input_mol.xyz'
            with open(out_dir/input_file, 'w') as f:
                f.write(xyz_block)

            if solvent is not None:
                output = subprocess.check_output(
                    ['xtb', input_file, '--scc', '--alpb', solvent, '--parallel', str(num_cores), '--chrg', str(charge)],
                    cwd = out_dir,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    stderr=subprocess.STDOUT,
                )
            else:
                output = subprocess.check_output(
                    ['xtb', input_file, '--scc', '--parallel', str(num_cores), '--chrg', str(charge)],
                    cwd = out_dir,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    stderr=subprocess.STDOUT,
                )

            output = output.split('\n')
            for line in output[::-1]:
                if 'TOTAL ENERGY' in line:
                    numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)
                    energy = float(numbers[0])
                    break

            with open(out_dir/'charges', 'r') as file:
                lines = file.readlines()
                charges = [0]*len(lines)
                for i, line in enumerate(lines):
                    charges[i] = float(line.split()[0])
        finally:
            if out_dir.exists():
                shutil.rmtree(out_dir)

        return energy, charges


def _optimize_conformer_worker(params):
    """Worker function for multiprocessing - must be at module level for pickling"""
    return optimize_conformer_with_xtb(*params)

def optimize_conformer_ensemble_with_xtb(conformers: List[Chem.Mol],
                                       solvent: Optional[str] = None,
                                       num_processes: int = 1,
                                       num_workers: int = 1,
                                       charge: int = 0,
                                       temp_dir: str | Path = TMPDIR,
                                       verbose: bool = False):
    """
    GFN2-XTB geometry optimization for a list of conformers.

    Parameters
    ----------
    conformers : list
        List of RDKit Mol objects (with 3D coordinates) to be optimized.
    solvent : str, optional
        Implicit solvent supported by XTB
        (https://xtb-docs.readthedocs.io/en/latest/gbsa.html). Default is ``None``.
    num_processes : int, optional
        Number of CPU cores used per xTB optimization. Default is 1.
    num_workers : int, optional
        Number of parallel workers (processes) to distribute conformers across.
        Ensure num_workers * num_processes <= available CPUs to avoid oversubscription.
        Default is 1.
    charge : int, optional
        Molecular charge. RDKit will be used to compute the formal charge if
        len(conformers) > 1. Default is 0.
    temp_dir : str or Path, optional
        Temporary directory for I/O. Default is the system temporary directory.
    verbose : bool, optional
        Show a simple progress bar in single-process mode. Default is ``False``.

    Returns
    -------
    tuple
        (conformers_opt, energies_opt, charges_opt) - tuple of lists containing
        optimized conformers, their energies, and partial charges.
    """

    if not conformers:
        return [], [], []

    available_cpus = os.cpu_count() or 1
    total_requested_cpus = max(1, num_workers) * max(1, num_processes)
    if total_requested_cpus > available_cpus:
        raise ValueError(
            f"Requested num_workers * num_processes = {total_requested_cpus} exceeds available CPUs ({available_cpus})."
        )

    # Serial path (single conformer or single worker)
    if len(conformers) == 1 or num_workers <= 1:
        conformers_opt = []
        energies_opt = []
        charges_opt = []

        iterator = tqdm(conformers, total=len(conformers), desc='XTB opt') if verbose else conformers
        for conf in iterator:
            opt_conf, opt_energy, opt_charges = optimize_conformer_with_xtb(
                conformer=conf,
                solvent=solvent,
                num_cores=num_processes,
                charge=charge,
                temp_dir=temp_dir,
            )
            conformers_opt.append(opt_conf)
            energies_opt.append(opt_energy)
            charges_opt.append(opt_charges)

        return conformers_opt, energies_opt, charges_opt

    # Parallel
    ctx = multiprocessing.get_context('spawn')

    args = [
        (conf, solvent, num_processes, Chem.GetFormalCharge(conf), temp_dir) for conf in conformers
    ]

    with ctx.Pool(processes=num_workers) as pool:
        iterator = pool.imap(_optimize_conformer_worker, args)
        if verbose:
            iterator = tqdm(iterator, total=len(args), desc='XTB opt')
        results = list(iterator)

    conformers_opt, energies_opt, charges_opt = zip(*results)
    return list(conformers_opt), list(energies_opt), list(charges_opt)


def generate_opt_conformers_xtb(smiles: str,
                                charge: int = 0,
                                solvent: Optional[str]=None,
                                MMFF_optimize: bool = True,
                                num_processes: int = 1,
                                num_workers: int = 1,
                                temp_dir: str | Path = TMPDIR,
                                verbose: bool = False,
                                num_confs: int = 1000):
    """
    Generate conformer ensemble with RDKit then relax with xTB.

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule.
    charge : int, optional
        Molecular charge. Default is 0.
    solvent : str, optional
        Implicit solvent to be used during optimization. Must be a solvent supported
        by XTB (https://xtb-docs.readthedocs.io/en/latest/gbsa.html). Default is ``None``.
    MMFF_optimize : bool, optional
        Optimize RDKit embedded molecules with MMFF94. Default is ``True``.
    num_processes : int, optional
        Number of CPU cores to be used in the xTB geometry optimization. Default is 1.
    num_workers : int, optional
        Number of parallel workers (processes) to distribute conformers across.
        Ensure num_workers * num_processes <= available CPUs to avoid oversubscription.
        Default is 1.
    temp_dir : str or Path, optional
        Temporary directory for I/O. Default is the system temporary directory.
    verbose : bool, optional
        Toggle tqdm progress bar. Default is ``False``.
    num_confs : int, optional
        Number of conformers to initially generate. Default is 1000.

    Returns
    -------
    clustered_conformers_xtb : list
        List of rdkit conformers after xTB relaxation and clustering.
    clustered_energies_xtb : list
        List of energies for associated conformers.
    clustered_charges_xtb : list
        List of partial charges for associated conformers.
    """
    available_cpus = os.cpu_count() or 1
    total_requested_cpus = max(1, num_workers) * max(1, num_processes)
    if total_requested_cpus > available_cpus:
        raise ValueError(
            f"Requested num_workers * num_processes = {total_requested_cpus} exceeds available CPUs ({available_cpus})."
        )
    mol_3d = embed_conformer_from_smiles(smiles, attempts = 50, MMFF_optimize = MMFF_optimize)

    if mol_3d is None:
        return None, None, None

    conformer_ensemble = generate_conformer_ensemble(
        mol_3d,
        num_confs=num_confs,
        num_threads = 4,
        threshold = 0.05,
        num_opt_steps = 50,
    )
    if verbose:
        print(f'Num conformers generated: \t{len(conformer_ensemble)}')

    clustered_conformers_index = cluster_conformers_butina(
        conformer_ensemble,
        threshold = 0.1,
        num_max_conformers = None,
    )
    clustered_conformers = [conformer_ensemble[c] for c in clustered_conformers_index]
    if verbose:
        print(f'Num conformers remaining: \t{len(clustered_conformers)}')

    # further optimizing each conformer with GFN2-XTB
    optimized_conformers_xtb, optimized_energies_xtb, optimized_charges_xtb = optimize_conformer_ensemble_with_xtb(
        clustered_conformers,
        solvent = solvent,
        num_processes = num_processes,
        num_workers = num_workers,
        charge=charge,
        temp_dir = temp_dir,
        verbose=verbose
    )
    if verbose:
        print(f'Num optimized confomers with xtb: \t{len(optimized_conformers_xtb)}')

    clustered_conformers_xtb_index = cluster_conformers_butina(
        optimized_conformers_xtb,
        threshold = 0.1,
        num_max_conformers = None,
    )
    clustered_conformers_xtb = [optimized_conformers_xtb[c] for c in clustered_conformers_xtb_index]
    clustered_charges_xtb = [optimized_charges_xtb[c] for c in clustered_conformers_xtb_index]
    clustered_energies_xtb = [optimized_energies_xtb[c] for c in clustered_conformers_xtb_index]

    if verbose:
        print(f'Num conformers remaining after xtb opt and cluster: \t{len(clustered_conformers_xtb)}')

    return clustered_conformers_xtb, clustered_energies_xtb, clustered_charges_xtb


def generate_opt_conformers(smiles: str,
                            MMFF_optimize: bool = True,
                            verbose: bool = False,
                            num_confs: int = 1000):
    """
    Generate optimal conformers with RDKit (MMFF94).

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule.
    MMFF_optimize : bool, optional
        Optimize RDKit embedded molecules with MMFF94. Default is ``True``.
    verbose : bool, optional
        Toggle tqdm progress bar. Default is ``False``.
    num_confs : int, optional
        Number of conformers to initially generate. Default is 1000.

    Returns
    -------
    list
        List of clustered rdkit conformers after RDKit embedding and optional MMFF relaxation.
    """
    mol_3d = embed_conformer_from_smiles(smiles, attempts = 50, MMFF_optimize = MMFF_optimize)

    if mol_3d is None:
        return None, None, None

    conformer_ensemble = generate_conformer_ensemble(
        mol_3d,
        num_confs=num_confs,
        num_threads = 4,
        threshold = 0.05,
        num_opt_steps = 50,
    )
    if verbose:
        print(f'Num conformers generated: \t{len(conformer_ensemble)}')

    clustered_conformers_index = cluster_conformers_butina(
        conformer_ensemble,
        threshold = 0.1,
        num_max_conformers = None,
    )
    clustered_conformers = [conformer_ensemble[c] for c in clustered_conformers_index]
    if verbose:
        print(f'Num conformers remaining: \t{len(clustered_conformers)}')

    return clustered_conformers
