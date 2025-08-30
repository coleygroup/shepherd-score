"""
Visualize pharmacophores and exit vectors with py3dmol.
"""
from typing import Union, List, Literal, Optional
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
from shepherd_score.evaluations.utils.convert_data import get_xyz_content
from shepherd_score.score.constants import P_TYPES
P_TYPES_LWRCASE = tuple(map(str.lower, P_TYPES))
P_IND2TYPES = {i : p for i, p in enumerate(P_TYPES)}
from shepherd_score.container import Molecule


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
         add_SAS = False,
         view = None,
         removeHs = False,
         opacity = 1.0,
         opacity_features = 1.0,
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
        mb = Chem.MolToMolBlock(mol, confId=0)
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
            if fam not in ('Aromatic', 'Donor', 'Acceptor', 'Halogen'):
                continue

            vec = pharm_vecs[i]
            __draw_arrow(view, clr, pharm_ancs[i], vec, flip=False, opacity=opacity_features)

            if fam == 'Aromatic':
                __draw_arrow(view, clr, pharm_ancs[i], vec, flip=True, opacity=opacity_features)

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


def draw_sample(
    generated_sample: dict,
    ref_mol = None,
    only_atoms = False,
    opacity = 0.6,
    view = None,
    color_scheme: Optional[str] = None,
    custom_carbon_color: Optional[str] = None,
    width = 800,
    height = 400,
):
    """
    Draw generated ShEPhERD sample with pharmacophore features and point cloud on surface
    accessible surface and electrostatics optionally overlaid on the reference molecule.

    Parameters
    ----------
    generated_sample : dict
        The generated sample with the following format.
        Note that it does NOT use x2 and assumes shape positions are in x3:
        {
            'x1': {
                'atoms': np.ndarray (N,),
                'positions': np.ndarray (N, 3)
            },
            'x2': {
                'positions': np.ndarray (N, 3),
            },
            'x3': {
                'charges': np.ndarray (N,),
                'positions': np.ndarray (N, 3),
            },
            'x4': {
                'types': np.ndarray (N,),
                'positions': np.ndarray (N, 3),
                'directions': np.ndarray (N, 3)
            }
        }
    ref_mol : Chem.Mol (default: None)
        The reference molecule with a conformer.
    only_atoms : bool (default: False)
        Whether to only draw the atoms and ignore the interaction profiles.
    opacity : float (default: 0.6)
        The opacity of the reference molecule.
    view : py3Dmol.view (default: None)
        The view to draw the molecule to. If None, a new view will be created.
    color_scheme : str (default: None)
        Provide a py3Dmol color scheme string.
        Example: 'whiteCarbon'
    custom_carbon_color : str (default: 'dark slate grey')
        Provide hex color of the carbon atoms. Programmed are 'dark slate grey' and 'light steel blue'.
    width : int (default: 800)
        The width of the view.
    height : int (default: 400)
        The height of the view.
    """
    if 'x1' not in generated_sample or 'atoms' not in generated_sample['x1'] or 'positions' not in generated_sample['x1']:
        raise ValueError('Generated sample does not contain atoms and positions in expected dict.')
    
    if view is None:
        view = py3Dmol.view(width=width, height=height)
        view.removeAllModels()
    
    if ref_mol is not None:
        mb = Chem.MolToMolBlock(ref_mol, confId=0)
        view.addModel(mb, 'sdf')
        view.setStyle({'model': -1}, {'stick': {'opacity': opacity}})

    xyz_block = get_xyz_content(generated_sample['x1']['atoms'], generated_sample['x1']['positions'])
    if xyz_block is None:
        return view

    view = draw(xyz_block,
                feats={},
                pharm_types=generated_sample['x4']['types'] if not only_atoms else None,
                pharm_ancs=generated_sample['x4']['positions'] if not only_atoms else None,
                pharm_vecs=generated_sample['x4']['directions'] if not only_atoms else None,
                point_cloud=generated_sample['x3']['positions'] if not only_atoms else None,
                esp=generated_sample['x3']['charges'] if not only_atoms else None,
                view=view,
                color_scheme=color_scheme,
                custom_carbon_color=custom_carbon_color if color_scheme is None else None)
    # return view.show() # view.show() to save memory
    return view


def draw_molecule(molecule: Molecule,
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


def create_pharmacophore_file_for_chimera(mol: Chem.Mol,
                                          id: Union[str, int],
                                          save_dir: str
                                          ) -> None:
    """
    Create SDF file for atoms (x1_{id}.sdf) and BILD file for pharmacophores (x4_{id}.bild).
    Drag and drop into ChimeraX to visualize.
    """
    save_dir_ = Path(save_dir)
    if not save_dir_.is_dir():
        save_dir_.mkdir(parents=True, exist_ok=True)

    pharm_types, pharm_pos, pharm_direction = get_pharmacophores(
        mol, 
        multi_vector = True, 
        check_access = False,
    )

    pharm_types = pharm_types + 1 # Accomodate virtual node at idx=0

    pharmacophore_colors = {
        0: (None, (0,0,0), 0.0, 0.0), # virtual node type
        1: ('Acceptor', (0.62,0.03,0.35), 0.3, 0.5),
        2: ('Donor', (0,0.55,0.55), 0.3, 0.5),
        3: ('Aromatic', (1.,.1,.0), 0.5, 0.5),
        4: ('Hydrophobe', (0.2,0.2,0.2), 0.5, 0.5),
        5: ('Halogen', (0.,1.,0), 0.5, 0.5),
        6: ('Cation', (0,0,1.), 0.1, 0.5),
        7: ('Anion', (1.,0,0), 0.1, 0.5),
        8: ('ZnBinder', (1.,.5,.5), 0.5, 0.5),
    }

    bild = ''
    for i in range(len(pharm_types)):
        pharm_type = int(pharm_types[i])
        pharm_name = pharmacophore_colors[pharm_type][0]
        p = pharm_pos[i]
        v = pharm_direction[i] * 2.0 # scaling size of vector
        
        bild += f'.color {pharmacophore_colors[pharm_type][1][0]} {pharmacophore_colors[pharm_type][1][1]} {pharmacophore_colors[pharm_type][1][2]}\n'
        bild += f'.transparency {pharmacophore_colors[pharm_type][3]}\n'
        if pharm_name not in ['Aromatic', 'Acceptor', 'Donor', 'Halogen']: 
            bild += f'.sphere {p[0]} {p[1]} {p[2]} {pharmacophore_colors[pharm_type][2]}\n'
        if np.linalg.norm(v) > 0.0:
            bild += f'.arrow {p[0]} {p[1]} {p[2]} {p[0] + v[0]} {p[1] + v[1]} {p[2] + v[2]} 0.1 0.2\n'
    # write pharmacophores
    with open(save_dir_ / f'x4_{id}.bild', 'w') as f:
        f.write(bild)
    # write mol
    Chem.MolToMolFile(mol, save_dir_ / f'x1_{id}.sdf')


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
