"""
Visualize pharmacophores and exit vectors with py3dmol.
"""
from __future__ import annotations
from typing import Union, List, Literal, Optional, TYPE_CHECKING
from pathlib import Path
from copy import deepcopy
import time

import numpy as np
from matplotlib.colors import to_hex

from rdkit import Chem
from rdkit.Chem import AllChem

# drawing
import py3Dmol
from IPython.display import SVG
import matplotlib.colors as mcolors
from rdkit.Chem.Draw import rdMolDraw2D


from shepherd_score.pharm_utils.pharmacophore import feature_colors, get_pharmacophores_dict, get_pharmacophores
from shepherd_score.evaluations.utils.convert_data import get_xyz_content_with_dummy
from shepherd_score.score.constants import P_TYPES

if TYPE_CHECKING:
    from shepherd_score.container import Molecule

P_TYPES_LWRCASE = tuple(map(str.lower, P_TYPES))
P_IND2TYPES = {i : p for i, p in enumerate(P_TYPES)}


def __draw_arrow(view, color, anchor_pos, rel_unit_vec, flip: bool = False, opacity: float = 1.0):
    """
    Add arrow
    """
    keys = ['x', 'y', 'z']
    if flip:
        flip = -1.
    else:
        flip = 1.

    view.addArrow({
        'start' : {k: float(anchor_pos[i]) for i, k in enumerate(keys)},
        'end' : {k: float(flip*2*rel_unit_vec[i] + anchor_pos[i]) for i, k in enumerate(keys)},
        'radius': .1,
        'radiusRatio':2.5,
        'mid':0.7,
        'color':to_hex(color),
        'opacity': opacity
    })


def draw(mol: Union[Chem.Mol, str],
         feats: dict = {},
         pharm_types: Union[np.ndarray, None] = None,
         pharm_ancs: Union[np.ndarray, None] = None,
         pharm_vecs: Union[np.ndarray, None] = None,
         point_cloud = None,
         esp = None,
         dummy_atom_pos = None,
         ev_pos = None,
         ev_vecs = None,
         add_SAS = False,
         view = None,
         removeHs = False,
         opacity = 1.0,
         opacity_features = 0.9,
         color_scheme: Optional[str] = None,
         custom_carbon_color: Optional[str] = None,
         width = 800,
         height = 400
):
    """
    Draw molecule with pharmacophore features and point cloud on surface accessible surface and electrostatics.

    Parameters
    ----------
    mol : Chem.Mol | str
        The molecule to draw. Either an RDKit Mol object or a string of the molecule in XYZ format.
        The XYZ string does not need to be a valid molecular structure.

    Optional Parameters
    -------------------
    feats : dict
        The pharmacophores to draw in a dictionary format with features as keys.
    pharm_types : np.ndarray (N,)
        The pharmacophores types
    pharm_ancs : np.ndarray (N, 3)
        The pharmacophores positions / anchor points.
    pharm_vecs : np.ndarray (N, 3)
        The pharmacophores vectors / directions.
    point_cloud : np.ndarray (N, 3)
        The point cloud positions.
    esp : np.ndarray (N,)
        The electrostatics values.
    add_SAS : bool
        Whether to add the SAS surface computed by py3Dmol.
    view : py3Dmol.view
        The view to draw the molecule to. If None, a new view will be created.
    removeHs : bool (default: False)
        Whether to remove the hydrogen atoms.
    color_scheme : str (default: None)
        Provide a py3Dmol color scheme string.
        Example: 'whiteCarbon'
    custom_carbon_color : str (default: None)
        Provide hex color of the carbon atoms. Programmed are 'dark slate grey' and 'light steel blue'.
    opacity : float (default: 1.0)
        The opacity of the molecule.
    opacity_features : float (default: 1.0)
        The opacity of the pharmacophore features.
    width : int (default: 800)
        The width of the view.
    height : int (default: 400)
        The height of the view.
    """
    if esp is not None:
        esp_colors = np.zeros((len(esp), 3))
        esp_colors[:,2] = np.where(esp < 0, 0, esp/np.max((np.max(-esp), np.max(esp)))).squeeze()
        esp_colors[:,0] = np.where(esp >= 0, 0, -esp/np.max((np.max(-esp), np.max(esp)))).squeeze()

    if view is None:
        view = py3Dmol.view(width=width, height=height)
        view.removeAllModels()
    if removeHs:
        mol = Chem.RemoveHs(mol)

    if isinstance(mol, Chem.Mol):
        mb = Chem.MolToMolBlock(mol)
        view.addModel(mb, 'sdf')
    else:
        view.addModel(mol, 'xyz')

    if color_scheme is not None:
        view.setStyle({'model': -1}, {'stick': {'colorscheme':color_scheme, 'opacity': opacity}})
    elif custom_carbon_color is not None:
        if custom_carbon_color == 'dark slate grey':
            custom_carbon_color = '#2F4F4F'
        elif custom_carbon_color == 'light steel blue':
            custom_carbon_color = '#B0C4DE'
        elif custom_carbon_color.startswith('#'):
            pass
        else:
            raise ValueError(f'Expects hex code for custom_carbon_color, got "{custom_carbon_color}"')
        view.setStyle({'model': -1, 'elem':'C'},{'stick':{'color':custom_carbon_color, 'opacity': opacity}})
        view.setStyle({'model': -1, 'not':{'elem':'C'}},{'stick':{'opacity': opacity}})
    else:
        view.setStyle({'model': -1}, {'stick': {'opacity': opacity}})
    keys = ['x', 'y', 'z']

    if feats:
        for fam in feats: # cycle through pharmacophores
            clr = feature_colors.get(fam, (.5,.5,.5))

            num_points = len(feats[fam]['P'])
            for i in range(num_points):
                pos = feats[fam]['P'][i]
                view.addSphere({'center':{keys[k]: float(pos[k]) for k in range(3)},
                                'radius':.5,'color':to_hex(clr), 'opacity': opacity_features})

                if fam not in ('Aromatic', 'Donor', 'Acceptor', 'Halogen'):
                    continue

                vec = feats[fam]['V'][i]
                __draw_arrow(view, clr, pos, vec, flip=False, opacity=opacity_features)

                if fam == 'Aromatic':
                    __draw_arrow(view, clr, pos, vec, flip=True, opacity=opacity_features)

    if feats == {} and pharm_types is not None and pharm_ancs is not None and pharm_vecs is not None:
        for i, ptype in enumerate(pharm_types):
            # Skip invalid pharmacophore type indices (like -1)
            if ptype < 0 or ptype >= len(P_TYPES):
                continue

            fam = P_IND2TYPES[ptype]
            clr = feature_colors.get(fam, (.5,.5,.5))
            view.addSphere({'center':{keys[k]: float(pharm_ancs[i][k]) for k in range(3)},
                            'radius':.5,'color':to_hex(clr), 'opacity': opacity_features})
            if fam not in ('Aromatic', 'Donor', 'Acceptor', 'Halogen', 'Dummy'):
                continue

            vec = pharm_vecs[i]
            __draw_arrow(view, clr, pharm_ancs[i], vec, flip=False, opacity=opacity_features)

            if fam == 'Aromatic':
                __draw_arrow(view, clr, pharm_ancs[i], vec, flip=True, opacity=opacity_features)

    if dummy_atom_pos is not None:
        for i, pos in enumerate(dummy_atom_pos):
            clr = (.8, .6, 1.)
            view.addSphere({'center':{keys[k]: float(pos[k]) for k in range(3)},
                            'radius':.45,'color':to_hex(clr), 'opacity': 0.9})

    if ev_pos is not None:
        for i, pos in enumerate(ev_pos):
            clr = (0., 0., 0.)
            if ev_vecs is not None:
                vec = ev_vecs[i]
                __draw_arrow(view, clr, pos, vec, flip=False, opacity=0.9)

    if point_cloud is not None:
        clr = np.zeros(3)
        if isinstance(point_cloud, np.ndarray):
            point_cloud = point_cloud.tolist()
        for i, pc in enumerate(point_cloud):
            if esp is not None:
                if np.sqrt(np.sum(np.square(esp_colors[i]))) < 0.3:
                    clr = np.ones(3)
                else:
                    clr = esp_colors[i]
            else:
                esp_colors = np.ones((len(point_cloud), 3))
            view.addSphere({'center':{'x':float(pc[0]), 'y':float(pc[1]), 'z':float(pc[2])}, 'radius':.1,'color':to_hex(clr), 'opacity':0.5})

    if add_SAS:
        view.addSurface(py3Dmol.SAS, {'opacity':0.5})
    view.zoomTo()
    # return view.show() # view.show() to save memory
    return view


def _process_generated_sample(
        generated_sample: dict,
        model_type: Literal['all', 'x2', 'x3', 'x4'] = 'all'
    ) -> tuple[str, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:

    if 'x1' not in generated_sample or 'atoms' not in generated_sample['x1'] or 'positions' not in generated_sample['x1']:
        raise ValueError('Generated sample does not contain atoms and positions in expected dict.')

    if model_type not in ['all', 'x2', 'x3', 'x4']:
        raise ValueError(f'Invalid model type: {model_type}')

    xyz_block, dummy_atom_pos = get_xyz_content_with_dummy(generated_sample['x1']['atoms'], generated_sample['x1']['positions'])

    surf_pos = generated_sample['x3']['positions'] if model_type in ['all', 'x3'] else None
    if model_type == 'x2':
        surf_pos = generated_sample['x2']['positions']

    surf_esp = generated_sample['x3']['charges'] if model_type in ['all', 'x3'] else None

    pharm_types = generated_sample['x4']['types'] if model_type in ['all', 'x4'] else None
    pharm_ancs = generated_sample['x4']['positions'] if model_type in ['all', 'x4'] else None
    pharm_vecs = generated_sample['x4']['directions'] if model_type in ['all', 'x4'] else None

    return xyz_block, dummy_atom_pos, surf_pos, surf_esp, pharm_types, pharm_ancs, pharm_vecs


def draw_sample(
    generated_sample: dict,
    ref_mol = None,
    only_atoms = False,
    model_type: Literal['all', 'x2', 'x3', 'x4'] = 'all',
    opacity = 0.6,
    view = None,
    color_scheme: Optional[str] = None,
    custom_carbon_color: Optional[str] = None,
    width = 800,
    height = 400,
):
    """
    Draw generated ShEPhERD sample with pharmacophore features and point cloud.

    Draws on surface accessible surface and electrostatics, optionally overlaid
    on the reference molecule.

    Parameters
    ----------
    generated_sample : dict
        The generated sample dictionary. Note that it does NOT use x2 and assumes
        shape positions are in x3. Expected format::

            {'x1': {'atoms': np.ndarray, 'positions': np.ndarray},
             'x2': {'positions': np.ndarray},
             'x3': {'charges': np.ndarray, 'positions': np.ndarray},
             'x4': {'types': np.ndarray, 'positions': np.ndarray,
                    'directions': np.ndarray}}

    ref_mol : Chem.Mol, optional
        The reference molecule with a conformer. Default is ``None``.
    only_atoms : bool, optional
        Whether to only draw the atoms and ignore the interaction profiles.
        Default is ``False``.
    model_type : str, optional
        One of 'all', 'x2', 'x3', 'x4'. Default is 'all'.
    opacity : float, optional
        The opacity of the reference molecule. Default is 0.6.
    view : py3Dmol.view, optional
        The view to draw the molecule to. If ``None``, a new view will be created.
    color_scheme : str, optional
        Provide a py3Dmol color scheme string (e.g., 'whiteCarbon').
    custom_carbon_color : str, optional
        Provide hex color of the carbon atoms. Programmed are 'dark slate grey'
        and 'light steel blue'.
    width : int, optional
        The width of the view. Default is 800.
    height : int, optional
        The height of the view. Default is 400.
    """
    xyz_block, dummy_atom_pos, surf_pos, surf_esp, pharm_types, pharm_ancs, pharm_vecs = _process_generated_sample(generated_sample, model_type)

    if view is None:
        view = py3Dmol.view(width=width, height=height)
        view.removeAllModels()

    if ref_mol is not None:
        mb = Chem.MolToMolBlock(ref_mol)
        view.addModel(mb, 'sdf')
        view.setStyle({'model': -1}, {'stick': {'opacity': opacity}})

    view = draw(xyz_block,
                feats={},
                pharm_types=pharm_types if not only_atoms else None,
                pharm_ancs=pharm_ancs if not only_atoms else None,
                pharm_vecs=pharm_vecs if not only_atoms else None,
                point_cloud=surf_pos if not only_atoms else None,
                esp=surf_esp if not only_atoms else None,
                dummy_atom_pos=dummy_atom_pos,
                view=view,
                color_scheme=color_scheme,
                custom_carbon_color=custom_carbon_color if color_scheme is None else None)
    # return view.show() # view.show() to save memory
    return view


def draw_molecule(molecule: Molecule,
                  dummy_atom_pos = None,
                  add_SAS = False,
                  view = None,
                  removeHs = False,
                  color_scheme: Optional[str] = None,
                  custom_carbon_color: Optional[str] = None,
                  opacity: float = 1.0,
                  opacity_features: float = 1.0,
                  no_surface_points: bool = False,
                  width = 800,
                  height = 400):
    view = draw(molecule.mol,
                pharm_types=molecule.pharm_types,
                pharm_ancs=molecule.pharm_ancs,
                pharm_vecs=molecule.pharm_vecs,
                point_cloud=molecule.surf_pos if not no_surface_points else None,
                esp=molecule.surf_esp if not no_surface_points else None,
                dummy_atom_pos=dummy_atom_pos,
                add_SAS=add_SAS,
                view=view,
                width=width,
                height=height,
                removeHs=removeHs,
                color_scheme=color_scheme,
                custom_carbon_color=custom_carbon_color if color_scheme is None else None,
                opacity=opacity,
                opacity_features=opacity_features)
    return view


def draw_pharmacophores(mol, view=None, width=800, height=400, opacity=1.0, opacity_features=1.0):
    """
    Generate the pharmacophores and visualize them.
    """
    draw(mol,
         feats = get_pharmacophores_dict(mol),
         view = view,
         width = width,
         height = height,
         opacity=opacity,
         opacity_features=opacity_features)


def chimera_from_mol(mol: Chem.Mol,
                     mol_id: Union[str, int],
                     surf_pos = None,
                     surf_esp = None,
                     ev_pos = None,
                     ev_vecs = None,
                     save_dir: str = './',
                     verbose: bool = True,
                     ) -> None:
    """
    Create SDF file for atoms (x1_{mol_id}.sdf) and BILD file for pharmacophores (x4_{mol_id}.bild).
    Drag and drop into ChimeraX to visualize.
    """
    save_dir_ = Path(save_dir)
    if not save_dir_.is_dir():
        save_dir_.mkdir(parents=True, exist_ok=True)

    pharm_types, pharm_pos, pharm_direction = get_pharmacophores(
        mol,
        multi_vector = False,
        check_access = False,
    )

    if ev_pos is not None and ev_vecs is not None:
        pharm_pos = np.concatenate([ev_pos, pharm_pos], axis=0)
        pharm_direction = np.concatenate([ev_vecs, pharm_direction], axis=0)
        pharm_types = np.concatenate([np.zeros((len(ev_pos)), dtype=int) + 10, pharm_types], axis=0)

    if surf_pos is not None and surf_esp is not None:
        surf_bild = _chimera_shape_esp_file(surf_pos, surf_esp)
        with open(save_dir_ / f'{mol_id}_x3.bild', 'w') as f:
            f.write(surf_bild)
        if verbose:
            print(f'Wrote ESP file to {save_dir_ / f"{mol_id}_x3.bild"}')

    pharm_types = pharm_types + 1 # Accomodate virtual node at idx=0

    bild = _chimera_pharmacophore_file(pharm_types, pharm_pos, pharm_direction)
    with open(save_dir_ / f'{mol_id}_x4.bild', 'w') as f:
        f.write(bild)
    if verbose:
        print(f'Wrote pharmacophore file to {save_dir_ / f"{mol_id}_x4.bild"}')

    Chem.MolToMolFile(mol, save_dir_ / f'{mol_id}_x1.sdf')
    if verbose:
        print(f'Wrote mol file to {save_dir_ / f"{mol_id}_x1.sdf"}')


def _chimera_pharmacophore_file(pharm_types: np.ndarray, pharm_pos: np.ndarray, pharm_direction: np.ndarray) -> str:
    pharmacophore_colors = {
        0: (None, (0,0,0), 0.0, 0.0), # virtual node type
        1: ('Acceptor', (0.62,0.03,0.35), 0.3, 0.5),
        2: ('Donor', (0,0.55,0.55), 0.3, 0.5),
        3: ('Aromatic', (.85,.5,.0), 0.5, 0.5),
        4: ('Hydrophobe', (0.2,0.2,0.2), 0.5, 0.5),
        5: ('Halogen', (0.,1.,0), 0.5, 0.5),
        6: ('Cation', (0,0,1.), 0.5, 0.5),
        7: ('Anion', (1.,0,0), 0.5, 0.5),
        8: ('ZnBinder', (1.,.5,.5), 0.5, 0.5),
        9: ('Dummy', feature_colors['Dummy'], 0.5, 0.5),
        10: ('Dummy atom', (0.8, 0.6, 1.), 0.5, 0.5),
        11: ('Exit vector', (0., 0., 0.), 0.5, 0.5),
    }

    bild = ''
    for i in range(len(pharm_types)):
        pharm_type = int(pharm_types[i])
        pharm_name = pharmacophore_colors[pharm_type][0]
        p = pharm_pos[i]
        v = pharm_direction[i] * 2.0 # scaling size of vector

        bild += f'.color {pharmacophore_colors[pharm_type][1][0]} {pharmacophore_colors[pharm_type][1][1]} {pharmacophore_colors[pharm_type][1][2]}\n'
        bild += f'.transparency {pharmacophore_colors[pharm_type][3]}\n'
        if pharm_name not in ['Aromatic', 'Acceptor', 'Donor', 'Halogen', 'Exit vector']:
            bild += f'.sphere {p[0]} {p[1]} {p[2]} {pharmacophore_colors[pharm_type][2]}\n'
        if np.linalg.norm(v) > 0.0:
            bild += f'.arrow {p[0]} {p[1]} {p[2]} {p[0] + v[0]} {p[1] + v[1]} {p[2] + v[2]} 0.1 0.2\n'
        if pharm_name == 'Aromatic':
            bild += f'.arrow {p[0]} {p[1]} {p[2]} {p[0] - v[0]} {p[1] - v[1]} {p[2] - v[2]} 0.1 0.2\n'
    return bild


def _chimera_shape_esp_file(surf_pos: np.ndarray,
                            surf_esp: np.ndarray,
                            norm_factor: float = 2.0,
                            ) -> str:
    esp = surf_esp * 4.0
    esp_pos = surf_pos

    esp_colors = np.zeros((len(esp), 3))
    esp_colors[:,2] = np.where(esp < 0, 0, esp/norm_factor).squeeze()
    esp_colors[:,0] = np.where(esp >= 0, 0, -esp/norm_factor).squeeze()

    bild = ''
    for i in range(len(esp_pos)):
        esp_color = esp_colors[i]
        p = esp_pos[i]
        bild += f'.color {esp_color[0]} {esp_color[1]} {esp_color[2]}\n'
        if (esp[i] ** 2.0) ** 0.5 < 0.5:
            bild += f'.transparency {0.9}\n'
        else:
            bild += f'.transparency {0.0}\n'
        bild += f'.sphere {p[0]} {p[1]} {p[2]} 0.05\n'

    return bild


def chimera_from_sample(generated_sample: dict,
                        mol_id: str | int,
                        save_dir: str,
                        model_type: Literal['all', 'x2', 'x3', 'x4'] = 'all',
                        esp_norm_factor: float = 2.0,
                        verbose: bool = True,
                        ) -> None:
    """
    Create SDF file for atoms (x1_{mol_id}.sdf) and BILD file for pharmacophores (x4_{mol_id}.bild).
    Drag and drop into ChimeraX to visualize.
    """
    path_ = Path(save_dir)
    if not path_.is_dir():
        path_.mkdir(parents=True, exist_ok=True)

    xyz_block, dummy_atom_pos, surf_pos, surf_esp, pharm_types, pharm_ancs, pharm_vecs = _process_generated_sample(generated_sample, model_type)

    if xyz_block is not None:
        with open(path_ / f'{mol_id}_x1.xyz', 'w') as f:
            f.write(xyz_block)
        if verbose:
            print(f'Wrote xyz file to {path_ / f"{mol_id}_x1.xyz"}')

    if dummy_atom_pos is not None:
        if pharm_ancs is not None:
            pharm_ancs = np.concatenate([dummy_atom_pos, pharm_ancs], axis=0)
        else:
            pharm_ancs = dummy_atom_pos
        if pharm_vecs is not None:
            pharm_vecs = np.concatenate([np.zeros((len(dummy_atom_pos), 3)), pharm_vecs], axis=0)
        else:
            pharm_vecs = np.zeros((len(dummy_atom_pos), 3))
        if pharm_types is not None:
            pharm_types = np.concatenate([np.zeros((len(dummy_atom_pos))) + 9, pharm_types], axis=0)
        else:
            pharm_types = np.zeros((len(dummy_atom_pos))) + 9

    if surf_pos is not None and surf_esp is not None:
        esp_bild = _chimera_shape_esp_file(surf_pos, surf_esp, norm_factor=esp_norm_factor)
        with open(path_ / f'{mol_id}_x3.bild', 'w') as f:
            f.write(esp_bild)
        if verbose:
            print(f'Wrote ESP file to {path_ / f"{mol_id}_x3.bild"}')

    if pharm_types is not None and pharm_ancs is not None and pharm_vecs is not None:
        pharm_types = pharm_types + 1 # Accomodate virtual node at idx=0
        pharm_bild = _chimera_pharmacophore_file(pharm_types, pharm_ancs, pharm_vecs)
        with open(path_ / f'{mol_id}_x4.bild', 'w') as f:
            f.write(pharm_bild)
        if verbose:
            print(f'Wrote pharmacophore file to {path_ / f"{mol_id}_x4.bild"}')


def draw_2d_valid(ref_mol: Chem.Mol,
                  mols: List[Chem.Mol | None],
                  mols_per_row: int = 5,
                  use_svg: bool = True,
                  find_atomic_overlap: bool = True,
                  ):
    """
    Draw 2D grid image of the reference molecule and a list of corresponding molecules.
    It will align the molecules to the reference molecule using the MCS and highlight
    the maximum common substructure between the reference molecule and the other molecules.

    Parameters
    ----------
    ref_mol : Chem.Mol
        The reference molecule to align the other molecules to.
    mols : List[Chem.Mol | None]
        The list of molecules to draw.
    mols_per_row : int
        The number of molecules to draw per row.
    use_svg : bool
        Whether to use SVG for the image.

    Returns
    -------
    MolsToGridImage
        The image of the molecules.

    Credit
    ------
    https://github.com/PatWalters/practical_cheminformatics_tutorials/
    """
    from rdkit.Chem import rdFMCS, AllChem
    temp_mol = Chem.MolFromSmiles(Chem.MolToSmiles(ref_mol))
    valid_mols = [Chem.MolFromSmiles(Chem.MolToSmiles(m)) for m in mols if m is not None]
    if (len(valid_mols) == 1 and valid_mols[0] is None) or len(valid_mols) == 0:
        return Chem.Draw.MolToImage(temp_mol, useSVG=True, legend='Target | Found no valid molecules')

    valid_inds = [i for i in range(len(mols)) if mols[i] is not None]
    if find_atomic_overlap:
        params = rdFMCS.MCSParameters()
        params.BondCompareParameters.CompleteRingsOnly = True
        params.AtomCompareParameters.CompleteRingsOnly = True
        # find the MCS
        mcs = rdFMCS.FindMCS([temp_mol] + valid_mols, params)
        # get query molecule from the MCS, we will use this as a template for alignment
        qmol = mcs.queryMol
        # generate coordinates for the template
        AllChem.Compute2DCoords(qmol)
        # generate coordinates for the molecules using the template
        [AllChem.GenerateDepictionMatching2DStructure(m, qmol) for m in valid_mols]

    return Chem.Draw.MolsToGridImage(
        [temp_mol]+ valid_mols,
        highlightAtomLists=[temp_mol.GetSubstructMatch(mcs.queryMol)]+[m.GetSubstructMatch(mcs.queryMol) for m in valid_mols] if find_atomic_overlap else None,
        molsPerRow=mols_per_row,
        legends=['Target'] + [f'Sample {i}' for i in valid_inds],
        useSVG=use_svg)


def draw_2d_highlight(mol: Chem.Mol,
                      atom_sets: List[List[int]],
                      colors: Optional[List[str]] = None,
                      label: Optional[Literal['atomLabel', 'molAtomMapNumber', 'atomNote']] = None,
                      compute_2d_coords: bool = True,
                      add_stereo_annotation: bool = True,
                      width: int = 800,
                      height: int = 600,
                      embed_display: bool = True
                      ) -> SVG:
    """
    Create an SVG representation of the molecule with highlighted atom sets.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to draw.
    atom_sets : List[List[int]]
        The list of atom sets to highlight.
    colors : List[str]
        The list of colors to use for the atom sets.
    label : Literal['atomLabel', 'molAtomMapNumber', 'atomNote']
        The label to use for the atom indices.
    width : int
        The width of the SVG image.
    height : int
        The height of the SVG image.

    Returns
    -------
    SVG: The SVG representation of the molecule with highlighted atom sets.
    """
    if colors is None:
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']

    non_empty_sets = [s for s in atom_sets if s]

    highlight_atoms = {}
    highlight_colors = {}

    for set_idx, atom_set in enumerate(non_empty_sets):
        color_rgb = mcolors.to_rgb(colors[set_idx % len(colors)])
        for atom_id in atom_set:
            highlight_atoms[atom_id] = color_rgb
            highlight_colors[atom_id] = color_rgb

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)

    opts = drawer.drawOptions()
    opts.addStereoAnnotation = add_stereo_annotation

    if label is not None:
        mol_copy = mol_with_atom_index(mol, label=label)
    else:
        mol_copy = deepcopy(mol)

    if compute_2d_coords:
        AllChem.Compute2DCoords(mol_copy)

    drawer.DrawMolecule(mol_copy,
                        highlightAtoms=list(highlight_atoms.keys()),
                        highlightAtomColors=highlight_colors)

    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    if embed_display:
        return SVG(svg)
    else:
        return svg


def mol_with_atom_index(mol: Chem.Mol, label: Literal['atomLabel', 'molAtomMapNumber', 'atomNote']='atomLabel'):
    mol_label = deepcopy(mol)
    for atom in mol_label.GetAtoms():
        atom.SetProp(label, str(atom.GetIdx()))
    return mol_label


def view_sample_trajectory(generated_sample, trajectory: Literal['x', 'x0']='x', frame_sleep: float=0.05,
                           ref_mol = None,
                           only_atoms = True,
                           opacity = 0.6,
                           color_scheme: Optional[str] = None,
                           custom_carbon_color: Optional[str] = None,
                           width = 800,
                           height = 400,
                           ):
    """
    View the trajectory of the generated sample.
    Must set store_trajectory=True or store_trajectory_x0=True in the `generate` function.
    """
    view = py3Dmol.view(width=width, height=height)
    suffix = f'_{trajectory}' if trajectory == 'x0' else ''
    for i in range(len(generated_sample['trajectories' + suffix])):
        view.clear()
        view = draw_sample(generated_sample['trajectories' + suffix][i],
                           only_atoms=only_atoms, view = view,
                           ref_mol=ref_mol,
                           opacity=opacity,
                           color_scheme=color_scheme,
                           custom_carbon_color=custom_carbon_color)
        view.update()
        time.sleep(frame_sleep)
    return view
