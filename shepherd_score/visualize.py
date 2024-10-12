"""
Visualize pharmacophores and exit vectors with py3dmol.
"""
from typing import Union
from pathlib import Path

import numpy as np
from matplotlib.colors import to_hex

from rdkit import Chem

# drawing
from rdkit.Chem.Draw import IPythonConsole
import py3Dmol

from shepherd_score.pharm_utils.pharmacophore import feature_colors, get_pharmacophores_dict, get_pharmacophores
from shepherd_score.score.constants import P_TYPES
P_TYPES_LWRCASE = tuple(map(str.lower, P_TYPES))
P_IND2TYPES = {i : p for i, p in enumerate(P_TYPES)}
from shepherd_score.container import Molecule


def __draw_arrow(view, color, anchor_pos, rel_unit_vec, flip: bool = False):
    """
    Add arrow
    """
    keys = ['x', 'y', 'z']
    if flip:
        flip = -1.
    else:
        flip = 1.
        
    view.addArrow({
        'start' : {k: anchor_pos[i] for i, k in enumerate(keys)},
        'end' : {k: flip*2*rel_unit_vec[i] + anchor_pos[i] for i, k in enumerate(keys)},
        'radius': .1,
        'radiusRatio':2.5,
        'mid':0.7,
        'color':to_hex(color)
    })


def draw(mol: Chem.rdchem.Mol,
         feats: dict = {},
         pharm_types: Union[np.ndarray, None] = None,
         pharm_ancs: Union[np.ndarray, None] = None,
         pharm_vecs: Union[np.ndarray, None] = None,
         point_cloud = None,
         esp = None,
         add_SAS = False,
         view = None,
         removeHs = False,
         width = 800,
         height = 400):
    """
    Draw molecule with pharmacophore features and point cloud on surface accessible surface and electrostatics.
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
    IPythonConsole.addMolToView(mol, view ,confId=0)
    keys = ['x', 'y', 'z']

    if feats:
        for fam in feats: # cycle through pharmacophores
            clr = feature_colors.get(fam, (.5,.5,.5))

            num_points = len(feats[fam]['P'])
            for i in range(num_points):
                pos = feats[fam]['P'][i]
                view.addSphere({'center':{keys[k]: pos[k] for k in range(3)},'radius':.5,'color':to_hex(clr)})

                if fam not in ('Aromatic', 'Donor', 'Acceptor', 'Halogen'):
                    continue

                vec = feats[fam]['V'][i]
                __draw_arrow(view, clr, pos, vec, flip=False)

                if fam == 'Aromatic':
                    __draw_arrow(view, clr, pos, vec, flip=True)
    
    if feats == {} and pharm_types is not None and pharm_ancs is not None and pharm_vecs is not None:
        for i, ptype in enumerate(pharm_types):
            fam = P_IND2TYPES[ptype]
            clr = feature_colors.get(fam, (.5,.5,.5))
            view.addSphere({'center':{keys[k]: pharm_ancs[i][k] for k in range(3)},'radius':.5,'color':to_hex(clr)})
            if fam not in ('Aromatic', 'Donor', 'Acceptor', 'Halogen'):
                continue

            vec = pharm_vecs[i]
            __draw_arrow(view, clr, pharm_ancs[i], vec, flip=False)

            if fam == 'Aromatic':
                __draw_arrow(view, clr, pharm_ancs[i], vec, flip=True)

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
            view.addSphere({'center':{'x':pc[0], 'y':pc[1], 'z':pc[2]}, 'radius':.1,'color':to_hex(clr), 'opacity':0.5})
    if add_SAS:
        view.addSurface(py3Dmol.SAS, {'opacity':0.5})
    view.zoomTo()
    # return view.show() # view.show() to save memory
    return view


def draw_molecule(molecule: Molecule,
                  add_SAS = False,
                  view = None,
                  removeHs = False,
                  width = 800,
                  height = 400):
    view = draw(molecule.mol,
                pharm_types=molecule.pharm_types,
                pharm_ancs=molecule.pharm_ancs,
                pharm_vecs=molecule.pharm_vecs,
                point_cloud=molecule.surf_pos,
                esp=molecule.surf_esp,
                add_SAS=add_SAS,
                view=view,
                width=width,
                height=height,
                removeHs=removeHs)
    return view


def draw_pharmacophores(mol, view=None, width=800, height=400):
    """
    Generate the pharmacophores and visualize them.
    """
    draw(mol,
         feats = get_pharmacophores_dict(mol),
         view = view,
         width = width,
         height = height)


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
