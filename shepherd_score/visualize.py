"""
Visualize pharmacophores and exit vectors with py3dmol.
"""
from __future__ import annotations
from typing import Union, List, Literal, Optional, Sequence, TYPE_CHECKING
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


from shepherd_score.pharm_utils.pharmacophore import (
    Pharmacophore,
    feature_colors,
    get_pharmacophores_dict,
    get_pharmacophores,
)
from shepherd_score.container.profiles import Surface
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


_COORD_KEYS = ('x', 'y', 'z')


def _ensure_view(view=None, width: int = 800, height: int = 400):
    if view is None:
        view = py3Dmol.view(width=width, height=height)
        view.removeAllModels()
    return view


def _compute_esp_colors(esp: np.ndarray) -> np.ndarray:
    esp_colors = np.zeros((len(esp), 3))
    norm = np.max((np.max(-esp), np.max(esp)))
    esp_colors[:, 2] = np.where(esp < 0, 0, esp / norm).squeeze()
    esp_colors[:, 0] = np.where(esp >= 0, 0, -esp / norm).squeeze()
    return esp_colors


def _resolve_custom_carbon_color(custom_carbon_color: str) -> str:
    if custom_carbon_color == 'dark slate grey':
        return '#2F4F4F'
    if custom_carbon_color == 'light steel blue':
        return '#B0C4DE'
    if custom_carbon_color.startswith('#'):
        return custom_carbon_color
    raise ValueError(f'Expects hex code for custom_carbon_color, got "{custom_carbon_color}"')


def draw_mol(mol: Union[Chem.Mol, str],
             removeHs: bool = False,
             opacity: float = 1.0,
             color_scheme: Optional[str] = None,
             custom_carbon_color: Optional[str] = None,
             highlight_atoms: Optional[List[int]] = None,
             dummy_atom_pos=None,
             add_SAS: bool = False,
             width: int = 800,
             height: int = 400,
             view=None):
    """
    Draw a molecule (RDKit mol or XYZ string) into a py3Dmol view.

    Parameters
    ----------
    mol : Chem.Mol | str
        RDKit molecule or XYZ-format string (need not be a valid structure).
    removeHs : bool
        Whether to strip hydrogens before drawing.
    opacity : float
        Stick opacity for the molecule.
    color_scheme : str, optional
        py3Dmol color scheme (e.g. ``'whiteCarbon'``).
    custom_carbon_color : str, optional
        Hex color or preset name for carbon atoms.
    highlight_atoms : list of int, optional
        Atom serial indices to highlight in purple.
    dummy_atom_pos : array-like (N, 3), optional
        Positions of dummy atoms to render as lavender spheres.
    add_SAS : bool
        Whether to add a solvent-accessible surface.
    width, height : int
        View dimensions when creating a new view.
    view : py3Dmol.view, optional
        Existing view; if ``None``, a new view is created and zoomed.
    """
    created = view is None
    view = _ensure_view(view, width=width, height=height)
    if removeHs:
        mol = Chem.RemoveHs(mol)

    if isinstance(mol, Chem.Mol):
        view.addModel(Chem.MolToMolBlock(mol), 'sdf')
    else:
        view.addModel(mol, 'xyz')

    if highlight_atoms is not None:
        view.setStyle({'serial': highlight_atoms}, {'stick': {'color': 'purple'}})

    if color_scheme is not None:
        view.setStyle({'model': -1}, {'stick': {'colorscheme': color_scheme, 'opacity': opacity}})
    elif custom_carbon_color is not None:
        carbon_color = _resolve_custom_carbon_color(custom_carbon_color)
        view.setStyle({'model': -1, 'elem': 'C'}, {'stick': {'color': carbon_color, 'opacity': opacity}})
        view.setStyle({'model': -1, 'not': {'elem': 'C'}}, {'stick': {'opacity': opacity}})
    else:
        view.setStyle({'model': -1}, {'stick': {'opacity': opacity}})

    if dummy_atom_pos is not None:
        clr = (.8, .6, 1.)
        for pos in dummy_atom_pos:
            view.addSphere({
                'center': {_COORD_KEYS[k]: float(pos[k]) for k in range(3)},
                'radius': .45,
                'color': to_hex(clr),
                'opacity': 0.9,
            })

    if add_SAS:
        view.addSurface(py3Dmol.SAS, {'opacity': 0.5})
    if created:
        view.zoomTo()
    return view


def draw_pharm(pharmacophore: Union[Pharmacophore, None] = None,
               feats: Optional[dict] = None,
               pharm_types: Union[np.ndarray, None] = None,
               pharm_ancs: Union[np.ndarray, None] = None,
               pharm_vecs: Union[np.ndarray, None] = None,
               ev_pos=None,
               ev_vecs=None,
               opacity_features: float = 0.9,
               width: int = 800,
               height: int = 400,
               view=None):
    """
    Draw pharmacophore features into a py3Dmol view.

    Accepts a :class:`~shepherd_score.pharm_utils.pharmacophore.Pharmacophore`
    container, a legacy ``feats`` dict from :func:`get_pharmacophores_dict`, or
    separate ``pharm_types`` / ``pharm_ancs`` / ``pharm_vecs`` arrays.

    Parameters
    ----------
    pharmacophore : Pharmacophore, optional
        Pharmacophore profile container.
    feats : dict, optional
        Feature-family dict with ``'P'`` positions and ``'V'`` vectors per type.
    pharm_types, pharm_ancs, pharm_vecs : np.ndarray, optional
        Flat pharmacophore arrays (used when ``feats`` is empty and no
        ``Pharmacophore`` is given).
    ev_pos, ev_vecs : array-like, optional
        Exit-vector anchor positions and unit vectors.
    opacity_features : float
        Opacity of pharmacophore spheres and arrows.
    width, height : int
        View dimensions when creating a new view.
    view : py3Dmol.view, optional
        Existing view; if ``None``, a new view is created and zoomed.
    """
    created = view is None
    view = _ensure_view(view, width=width, height=height)
    if feats is None:
        feats = {}

    if isinstance(pharmacophore, Pharmacophore):
        pharm_types = pharmacophore.types
        pharm_ancs = pharmacophore.positions
        pharm_vecs = pharmacophore.vectors

    if feats:
        for fam in feats:
            clr = feature_colors.get(fam, (.5, .5, .5))
            num_points = len(feats[fam]['P'])
            for i in range(num_points):
                pos = feats[fam]['P'][i]
                view.addSphere({
                    'center': {_COORD_KEYS[k]: float(pos[k]) for k in range(3)},
                    'radius': .5,
                    'color': to_hex(clr),
                    'opacity': opacity_features,
                })
                if fam not in ('Aromatic', 'Donor', 'Acceptor', 'Halogen'):
                    continue
                vec = feats[fam]['V'][i]
                __draw_arrow(view, clr, pos, vec, flip=False, opacity=opacity_features)
                if fam == 'Aromatic':
                    __draw_arrow(view, clr, pos, vec, flip=True, opacity=opacity_features)
    elif pharm_types is not None and pharm_ancs is not None and pharm_vecs is not None:
        for i, ptype in enumerate(pharm_types):
            if ptype < 0 or ptype >= len(P_TYPES):
                continue
            fam = P_IND2TYPES[ptype]
            clr = feature_colors.get(fam, (.5, .5, .5))
            view.addSphere({
                'center': {_COORD_KEYS[k]: float(pharm_ancs[i][k]) for k in range(3)},
                'radius': .5,
                'color': to_hex(clr),
                'opacity': opacity_features,
            })
            if fam not in ('Aromatic', 'Donor', 'Acceptor', 'Halogen', 'Dummy'):
                continue
            vec = pharm_vecs[i]
            __draw_arrow(view, clr, pharm_ancs[i], vec, flip=False, opacity=opacity_features)
            if fam == 'Aromatic':
                __draw_arrow(view, clr, pharm_ancs[i], vec, flip=True, opacity=opacity_features)

    if ev_pos is not None:
        for i, pos in enumerate(ev_pos):
            clr = (0., 0., 0.)
            if ev_vecs is not None:
                __draw_arrow(view, clr, pos, ev_vecs[i], flip=False, opacity=0.9)
    if created:
        view.zoomTo()
    return view


def draw_surface(surface: Union[Surface, None] = None,
                 point_cloud=None,
                 esp: Union[np.ndarray, None] = None,
                 opacity: float = 0.5,
                 radius: float = 0.1,
                 width: int = 800,
                 height: int = 400,
                 view=None):
    """
    Draw a molecular surface point cloud into a py3Dmol view.

    Accepts a :class:`~shepherd_score.container.profiles.Surface` container or
    separate ``point_cloud`` / ``esp`` arrays.

    Parameters
    ----------
    surface : Surface, optional
        Surface profile with ``positions`` and optional ``esp``.
    point_cloud : array-like (N, 3), optional
        Surface point positions (overrides ``surface.positions`` when given).
    esp : np.ndarray (N,), optional
        Electrostatic potential per point (overrides ``surface.esp`` when given).
    opacity : float
        Sphere opacity for surface points.
    radius : float
        Sphere radius for surface points.
    width, height : int
        View dimensions when creating a new view.
    view : py3Dmol.view, optional
        Existing view; if ``None``, a new view is created and zoomed.
    """
    created = view is None
    view = _ensure_view(view, width=width, height=height)
    if isinstance(surface, Surface):
        if point_cloud is None:
            point_cloud = surface.positions
        if esp is None:
            esp = surface.esp

    if point_cloud is None:
        if created:
            view.zoomTo()
        return view

    esp_colors = _compute_esp_colors(esp) if esp is not None else None
    if isinstance(point_cloud, np.ndarray):
        point_cloud = point_cloud.tolist()

    clr = np.zeros(3)
    for i, pc in enumerate(point_cloud):
        if esp_colors is not None:
            if np.sqrt(np.sum(np.square(esp_colors[i]))) < 0.3:
                clr = np.ones(3)
            else:
                clr = esp_colors[i]
        else:
            clr = np.ones(3)
        view.addSphere({
            'center': {'x': float(pc[0]), 'y': float(pc[1]), 'z': float(pc[2])},
            'radius': radius,
            'color': to_hex(clr),
            'opacity': opacity,
        })
    if created:
        view.zoomTo()
    return view


def draw(mol: Union[Chem.Mol, str],
         pharmacophore: Union[Pharmacophore, None] = None,
         surface: Union[Surface, None] = None,
         feats: dict = {},
         pharm_types: Union[np.ndarray, None] = None,
         pharm_ancs: Union[np.ndarray, None] = None,
         pharm_vecs: Union[np.ndarray, None] = None,
         point_cloud=None,
         esp=None,
         dummy_atom_pos=None,
         ev_pos=None,
         ev_vecs=None,
         add_SAS=False,
         view=None,
         removeHs=False,
         opacity=1.0,
         opacity_features=0.9,
         color_scheme: Optional[str] = None,
         custom_carbon_color: Optional[str] = None,
         highlight_atoms: Optional[List[int]] = None,
         width=800,
         height=400):
    """
    Draw molecule with pharmacophore features and surface point cloud.

    Convenience wrapper around :func:`draw_mol`, :func:`draw_pharm`, and
    :func:`draw_surface`.

    Parameters
    ----------
    mol : Chem.Mol | str
        RDKit molecule or XYZ-format string.
    pharmacophore : Pharmacophore, optional
        Pharmacophore profile container. Takes precedence over ``feats`` and
        flat ``pharm_*`` arrays when given.
    surface : Surface, optional
        Surface profile container. ``point_cloud`` / ``esp`` override container
        fields when explicitly provided.
    feats : dict, optional
        Legacy pharmacophore dict from :func:`get_pharmacophores_dict`.
    pharm_types, pharm_ancs, pharm_vecs : np.ndarray, optional
        Flat pharmacophore arrays.
    point_cloud : array-like (N, 3), optional
        Surface point positions.
    esp : np.ndarray (N,), optional
        Electrostatic potential per surface point.
    view : py3Dmol.view, optional
        Existing view; if ``None``, a new view is created.
    """
    view = draw_mol(
        mol,
        removeHs=removeHs,
        opacity=opacity,
        color_scheme=color_scheme,
        custom_carbon_color=custom_carbon_color,
        highlight_atoms=highlight_atoms,
        dummy_atom_pos=dummy_atom_pos,
        add_SAS=add_SAS,
        width=width,
        height=height,
        view=view,
    )
    view = draw_pharm(
        pharmacophore=pharmacophore,
        feats=feats,
        pharm_types=pharm_types,
        pharm_ancs=pharm_ancs,
        pharm_vecs=pharm_vecs,
        ev_pos=ev_pos,
        ev_vecs=ev_vecs,
        opacity_features=opacity_features,
        view=view,
    )
    view = draw_surface(
        surface=surface,
        point_cloud=point_cloud,
        esp=esp,
        view=view,
    )
    view.zoomTo()
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
                  highlight_atoms: Optional[List[int]] = None,
                  width = 800,
                  height = 400):
    view = draw_mol(
        molecule.mol,
        removeHs=removeHs,
        opacity=opacity,
        color_scheme=color_scheme,
        custom_carbon_color=custom_carbon_color if color_scheme is None else None,
        highlight_atoms=highlight_atoms,
        dummy_atom_pos=dummy_atom_pos,
        add_SAS=add_SAS,
        width=width,
        height=height,
        view=view,
    )
    if molecule.pharmacophore is not None:
        view = draw_pharm(
            pharmacophore=molecule.pharmacophore,
            opacity_features=opacity_features,
            view=view,
        )
    if not no_surface_points and molecule.surface.positions is not None:
        view = draw_surface(surface=molecule.surface, view=view)
    view.zoomTo()
    return view


def draw_pharmacophores(mol, view=None, width=800, height=400, opacity=1.0, opacity_features=1.0):
    """
    Generate the pharmacophores and visualize them.
    """
    view = draw_mol(
        mol,
        width=width,
        height=height,
        opacity=opacity,
        view=view,
    )
    view = draw_pharm(feats=get_pharmacophores_dict(mol), opacity_features=opacity_features, view=view)
    view.zoomTo()
    return view


def draw_atom_sample(
    generated_sample: List,
    ref_mol = None,
    feats: dict = {},
    pharm_types: Union[np.ndarray, None] = None,
    pharm_ancs: Union[np.ndarray, None] = None,
    pharm_vecs: Union[np.ndarray, None] = None,
    point_cloud = None,
    esp = None,
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
    generated_sample : list
        Expects [atoms, positions] where atoms is a list of atomic numbers and positions is a list of 3D coordinates.

    ref_mol : Chem.Mol, optional
        The reference molecule with a conformer. Default is ``None``.
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
    xyz_block, dummy_atom_pos = get_xyz_content_with_dummy(generated_sample[0], generated_sample[1])

    if view is None:
        view = py3Dmol.view(width=width, height=height)
        view.removeAllModels()

    if ref_mol is not None:
        mb = Chem.MolToMolBlock(ref_mol)
        view.addModel(mb, 'sdf')
        view.setStyle({'model': -1}, {'stick': {'opacity': opacity}})

    view = draw(xyz_block,
                feats=feats,
                pharm_types=pharm_types,
                pharm_ancs=pharm_ancs,
                pharm_vecs=pharm_vecs,
                point_cloud=point_cloud,
                esp=esp,
                dummy_atom_pos=dummy_atom_pos,
                view=view,
                color_scheme=color_scheme,
                custom_carbon_color=custom_carbon_color if color_scheme is None else None)
    # return view.show() # view.show() to save memory
    return view


def _normalize_chimera_outputs(
    outputs: Sequence[Literal['x1', 'x2', 'x3', 'x4']] | None,
) -> set[str]:
    valid = {'x1', 'x2', 'x3', 'x4'}
    if outputs is None:
        return valid
    selected = set(outputs)
    invalid = selected - valid
    if invalid:
        raise ValueError(f'Invalid chimera outputs: {sorted(invalid)}. Expected subset of {sorted(valid)}.')
    return selected


def _merge_dummy_into_pharm(
    dummy_atom_pos: np.ndarray | None,
    pharm_types: np.ndarray | None,
    pharm_ancs: np.ndarray | None,
    pharm_vecs: np.ndarray | None,
    dummy_type: int = 9,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Prepend dummy-atom markers into pharmacophore arrays (type index before +1 shift)."""
    if dummy_atom_pos is None:
        return pharm_types, pharm_ancs, pharm_vecs

    n_dummy = len(dummy_atom_pos)
    dummy_types = np.zeros(n_dummy) + dummy_type
    dummy_vecs = np.zeros((n_dummy, 3))

    if pharm_ancs is not None:
        pharm_ancs = np.concatenate([dummy_atom_pos, pharm_ancs], axis=0)
    else:
        pharm_ancs = dummy_atom_pos

    if pharm_vecs is not None:
        pharm_vecs = np.concatenate([dummy_vecs, pharm_vecs], axis=0)
    else:
        pharm_vecs = dummy_vecs

    if pharm_types is not None:
        pharm_types = np.concatenate([dummy_types, pharm_types], axis=0)
    else:
        pharm_types = dummy_types

    return pharm_types, pharm_ancs, pharm_vecs


def _chimera_pharmacophore_file(pharm_types: np.ndarray, pharm_pos: np.ndarray, pharm_direction: np.ndarray, pharm_transparency: float = 0.5) -> str:
    pharmacophore_colors = {
        0: (None, (0,0,0), 0.0, 0.0), # virtual node type
        1: ('Acceptor', (0.62,0.03,0.35), 0.3, pharm_transparency),
        2: ('Donor', (0,0.55,0.55), 0.3, pharm_transparency),
        3: ('Aromatic', (.85,.5,.0), 0.5, pharm_transparency),
        4: ('Hydrophobe', (0.2,0.2,0.2), 0.5, pharm_transparency),
        5: ('Halogen', (0.,1.,0), 0.5, pharm_transparency),
        6: ('Cation', (0,0,1.), 0.5, pharm_transparency),
        7: ('Anion', (1.,0,0), 0.5, pharm_transparency),
        8: ('ZnBinder', (1.,.5,.5), 0.5, pharm_transparency),
        9: ('Dummy', feature_colors['Dummy'], 0.5, pharm_transparency),
        10: ('Dummy atom', (0.8, 0.6, 1.), 0.5, pharm_transparency),
        11: ('Exit vector', (0., 0., 0.), 0.5, pharm_transparency),
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
                            surf_esp: np.ndarray | None = None,
                            norm_factor: float = 2.0,
                            surf_point_size: float = 0.05,
                            transparency_charged: float = 0.0,
                            neutral_color_threshold: float = 0.3,
                            transparency_neutral: float = 0.9,
                            ) -> str:
    if surf_esp is None:
        bild = ''
        for i in range(len(surf_pos)):
            p = surf_pos[i]
            bild += f'.color {0.0} {0.0} {0.0}\n'
            bild += f'.transparency {transparency_neutral}\n'
            bild += f'.sphere {p[0]} {p[1]} {p[2]} {surf_point_size}\n'
        return bild

    esp = surf_esp * 4.0
    esp_pos = surf_pos

    esp_colors = np.zeros((len(esp), 3))
    esp_colors[:,2] = np.where(esp < 0, 0, esp/norm_factor).squeeze()
    esp_colors[:,0] = np.where(esp >= 0, 0, -esp/norm_factor).squeeze()

    # Low ESP defaults to black (0,0,0); transparency alone still looks dark in ChimeraX.
    # Use white for neutral points, matching draw()'s py3Dmol ESP cloud behavior.
    color_magnitude = np.sqrt(np.sum(np.square(esp_colors), axis=1))
    neutral_mask = color_magnitude < neutral_color_threshold
    esp_colors[neutral_mask] = 1.0

    bild = ''
    for i in range(len(esp_pos)):
        esp_color = esp_colors[i]
        p = esp_pos[i]
        bild += f'.color {esp_color[0]} {esp_color[1]} {esp_color[2]}\n'
        if neutral_mask[i]:
            bild += f'.transparency {transparency_neutral}\n'
        else:
            bild += f'.transparency {transparency_charged}\n'
        bild += f'.sphere {p[0]} {p[1]} {p[2]} {surf_point_size}\n'

    return bild


def _write_chimera_files(
    mol_id: str | int,
    save_dir: str | Path,
    *,
    outputs: Sequence[Literal['x1', 'x2', 'x3', 'x4']] | None = None,
    mol: Chem.Mol | None = None,
    xyz_block: str | None = None,
    output_mol_file: Literal['sdf', 'xyz'] = 'sdf',
    surf_pos: np.ndarray | None = None,
    surf_esp: np.ndarray | None = None,
    pharm_types: np.ndarray | None = None,
    pharm_ancs: np.ndarray | None = None,
    pharm_vecs: np.ndarray | None = None,
    dummy_atom_pos: np.ndarray | None = None,
    pharm_transparency: float = 0.5,
    esp_norm_factor: float = 2.0,
    esp_transparency_charged: float = 0.9,
    esp_neutral_color_threshold: float = 0.3,
    esp_transparency_neutral: float = 0.9,
    surf_point_size: float = 0.05,
    verbose: bool = True,
) -> None:
    """
    Write ChimeraX-compatible files for available representations.

    Only representations listed in ``outputs`` that are also available are written:
    - ``x1``: atoms (``.sdf`` or ``.xyz``)
    - ``x2``: shape surface points (``.bild``)
    - ``x3``: ESP surface points (``.bild``)
    - ``x4``: pharmacophores (``.bild``)
    """
    selected = _normalize_chimera_outputs(outputs)
    path_ = Path(save_dir)
    if not path_.is_dir():
        path_.mkdir(parents=True, exist_ok=True)

    if 'x1' in selected:
        if output_mol_file == 'sdf' and mol is not None:
            out_path = path_ / f'{mol_id}_x1.sdf'
            with Chem.SDWriter(out_path) as w:
                w.write(mol)
            if verbose:
                print(f'Wrote mol file to {out_path}')
        elif xyz_block is not None:
            out_path = path_ / f'{mol_id}_x1.xyz'
            with open(out_path, 'w') as f:
                f.write(xyz_block)
            if verbose:
                print(f'Wrote xyz file to {out_path}')

    if 'x2' in selected and surf_pos is not None:
        shape_bild = _chimera_shape_esp_file(
            surf_pos,
            None,
            norm_factor=esp_norm_factor,
            surf_point_size=surf_point_size,
            transparency_charged=esp_transparency_charged,
            neutral_color_threshold=esp_neutral_color_threshold,
            transparency_neutral=esp_transparency_neutral,
        )
        out_path = path_ / f'{mol_id}_x2.bild'
        with open(out_path, 'w') as f:
            f.write(shape_bild)
        if verbose:
            print(f'Wrote shape file to {out_path}')

    if 'x3' in selected and surf_pos is not None and surf_esp is not None:
        esp_bild = _chimera_shape_esp_file(
            surf_pos,
            surf_esp,
            norm_factor=esp_norm_factor,
            surf_point_size=surf_point_size,
            transparency_charged=esp_transparency_charged,
            neutral_color_threshold=esp_neutral_color_threshold,
            transparency_neutral=esp_transparency_neutral,
        )
        out_path = path_ / f'{mol_id}_x3.bild'
        with open(out_path, 'w') as f:
            f.write(esp_bild)
        if verbose:
            print(f'Wrote ESP file to {out_path}')

    if 'x4' in selected:
        pharm_types, pharm_ancs, pharm_vecs = _merge_dummy_into_pharm(
            dummy_atom_pos, pharm_types, pharm_ancs, pharm_vecs
        )
        if pharm_types is not None and pharm_ancs is not None and pharm_vecs is not None:
            # Accommodate virtual node at idx=0
            pharm_bild = _chimera_pharmacophore_file(
                pharm_types + 1, pharm_ancs, pharm_vecs, pharm_transparency=pharm_transparency
            )
            out_path = path_ / f'{mol_id}_x4.bild'
            with open(out_path, 'w') as f:
                f.write(pharm_bild)
            if verbose:
                print(f'Wrote pharmacophore file to {out_path}')


def chimera_from_mol(mol: Chem.Mol,
                     mol_id: Union[str, int],
                     surf_pos = None,
                     surf_esp = None,
                     ev_pos = None,
                     ev_vecs = None,
                     save_dir: str = './',
                     outputs: Sequence[Literal['x1', 'x2', 'x3', 'x4']] | None = None,
                     pharm_transparency: float = 0.5,
                     esp_norm_factor: float = 2.0,
                     esp_transparency_charged: float = 0.9,
                     esp_neutral_color_threshold: float = 0.3,
                     esp_transparency_neutral: float = 0.9,
                     surf_point_size: float = 0.05,
                     verbose: bool = True,
                     ) -> None:
    """
    Write ChimeraX files from an RDKit mol (and optional surface / exit-vector data).

    Parameters
    ----------
    outputs : sequence of {'x1','x2','x3','x4'}, optional
        Representations to write. Default writes all that are available.
    """
    selected = _normalize_chimera_outputs(outputs)

    pharm_types = pharm_ancs = pharm_vecs = None
    if 'x4' in selected:
        pharm_types, pharm_ancs, pharm_vecs = get_pharmacophores(
            mol,
            multi_vector=False,
            check_access=False,
        )
        if ev_pos is not None and ev_vecs is not None:
            # Exit vectors use type index 10 before the +1 virtual-node shift
            pharm_ancs = np.concatenate([ev_pos, pharm_ancs], axis=0)
            pharm_vecs = np.concatenate([ev_vecs, pharm_vecs], axis=0)
            pharm_types = np.concatenate(
                [np.zeros(len(ev_pos), dtype=int) + 10, pharm_types], axis=0
            )

    _write_chimera_files(
        mol_id,
        save_dir,
        outputs=selected,
        mol=mol,
        output_mol_file='sdf',
        surf_pos=surf_pos,
        surf_esp=surf_esp,
        pharm_types=pharm_types,
        pharm_ancs=pharm_ancs,
        pharm_vecs=pharm_vecs,
        pharm_transparency=pharm_transparency,
        esp_norm_factor=esp_norm_factor,
        esp_transparency_charged=esp_transparency_charged,
        esp_neutral_color_threshold=esp_neutral_color_threshold,
        esp_transparency_neutral=esp_transparency_neutral,
        surf_point_size=surf_point_size,
        verbose=verbose,
    )


def chimera_from_sample(generated_sample: dict,
                        mol_id: str | int,
                        save_dir: str,
                        model_type: Literal['all', 'x2', 'x3', 'x4'] = 'all',
                        outputs: Sequence[Literal['x1', 'x2', 'x3', 'x4']] | None = None,
                        pharm_transparency: float = 0.5,
                        esp_norm_factor: float = 2.0,
                        esp_transparency_charged: float = 0.9,
                        esp_neutral_color_threshold: float = 0.3,
                        esp_transparency_neutral: float = 0.9,
                        surf_point_size: float = 0.05,
                        verbose: bool = True,
                        ) -> None:
    """
    Write ChimeraX files from a generated ShEPhERD sample dict.

    Parameters
    ----------
    outputs : sequence of {'x1','x2','x3','x4'}, optional
        Representations to write. Default writes all that are available.
    """
    xyz_block, dummy_atom_pos, surf_pos, surf_esp, pharm_types, pharm_ancs, pharm_vecs = (
        _process_generated_sample(generated_sample, model_type)
    )
    _write_chimera_files(
        mol_id,
        save_dir,
        outputs=outputs,
        xyz_block=xyz_block,
        output_mol_file='xyz',
        surf_pos=surf_pos,
        surf_esp=surf_esp,
        pharm_types=pharm_types,
        pharm_ancs=pharm_ancs,
        pharm_vecs=pharm_vecs,
        dummy_atom_pos=dummy_atom_pos,
        pharm_transparency=pharm_transparency,
        esp_norm_factor=esp_norm_factor,
        esp_transparency_charged=esp_transparency_charged,
        esp_neutral_color_threshold=esp_neutral_color_threshold,
        esp_transparency_neutral=esp_transparency_neutral,
        surf_point_size=surf_point_size,
        verbose=verbose,
    )


def chimera_from_atom_sample(
    generated_sample: List,
    mol_id: str | int,
    save_dir: str,
    only_dummy_atoms: bool = False,
    outputs: Sequence[Literal['x1', 'x2', 'x3', 'x4']] | None = None,
    pharm_transparency: float = 0.5,
    verbose: bool = True,
) -> None:
    """
    Write ChimeraX files from an atom-only sample ``[atomic_numbers, positions]``.

    Parameters
    ----------
    outputs : sequence of {'x1','x2','x3','x4'}, optional
        Representations to write. Default is ``('x1', 'x4')`` (what this input can provide).
        If ``only_dummy_atoms=True``, ``x1`` is omitted.
    """
    if outputs is None:
        outputs = ('x1', 'x4')
    selected = _normalize_chimera_outputs(outputs)
    if only_dummy_atoms:
        selected.discard('x1')

    xyz_block, dummy_atom_pos = get_xyz_content_with_dummy(generated_sample[0], generated_sample[1])
    _write_chimera_files(
        mol_id,
        save_dir,
        outputs=selected,
        xyz_block=xyz_block,
        output_mol_file='xyz',
        dummy_atom_pos=dummy_atom_pos,
        pharm_transparency=pharm_transparency,
        verbose=verbose,
    )


def chimera_from_molecule(molec: Molecule,
                          mol_id: str | int,
                          save_dir: str,
                          outputs: Sequence[Literal['x1', 'x2', 'x3', 'x4']] | None = None,
                          pharm_transparency: float = 0.5,
                          esp_norm_factor: float = 2.0,
                          esp_transparency_charged: float = 0.9,
                          esp_neutral_color_threshold: float = 0.3,
                          esp_transparency_neutral: float = 0.9,
                          surf_point_size: float = 0.05,
                          dummy_atom_pos: Optional[np.ndarray] = None,
                          output_mol_file: Literal['sdf', 'xyz'] = 'sdf',
                          verbose: bool = True,
                          ) -> None:
    """
    Write ChimeraX files from a ``Molecule`` object.

    Parameters
    ----------
    outputs : sequence of {'x1','x2','x3','x4'}, optional
        Representations to write. Default writes all that are available.
    """
    selected = _normalize_chimera_outputs(outputs)

    xyz_block = None
    if dummy_atom_pos is None or ('x1' in selected and output_mol_file == 'xyz'):
        xyz_block, extracted_dummy = get_xyz_content_with_dummy(
            atomic_numbers=np.array([a.GetAtomicNum() for a in molec.mol.GetAtoms()]),
            positions=molec.mol.GetConformer().GetPositions(),
        )
        if dummy_atom_pos is None:
            dummy_atom_pos = extracted_dummy

    _write_chimera_files(
        mol_id,
        save_dir,
        outputs=selected,
        mol=molec.mol,
        xyz_block=xyz_block,
        output_mol_file=output_mol_file,
        surf_pos=molec.surf_pos,
        surf_esp=molec.surf_esp,
        pharm_types=molec.pharm_types,
        pharm_ancs=molec.pharm_ancs,
        pharm_vecs=molec.pharm_vecs,
        dummy_atom_pos=dummy_atom_pos,
        pharm_transparency=pharm_transparency,
        esp_norm_factor=esp_norm_factor,
        esp_transparency_charged=esp_transparency_charged,
        esp_neutral_color_threshold=esp_neutral_color_threshold,
        esp_transparency_neutral=esp_transparency_neutral,
        surf_point_size=surf_point_size,
        verbose=verbose,
    )



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
