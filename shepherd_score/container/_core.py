"""
Molecule class to hold molecule geometries and extract interaction profiles.
MoleculePair class facilitates alignment with interaction profiles.
"""
from typing import Union, List, Optional, Tuple, Iterable, Literal
from copy import deepcopy
from functools import partial
from dataclasses import dataclass, field

import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Geometry.rdGeometry import Point3D
import torch

from shepherd_score.score.constants import COULOMB_SCALING, LAM_SCALING, ALPHA  # noqa: F401
from shepherd_score.generate_point_cloud import get_atom_coords, get_atomic_vdw_radii, get_molecular_surface, get_molecular_surface_const_density
from shepherd_score.score.gaussian_overlap_np import get_overlap_np
from shepherd_score.score.gaussian_overlap import get_overlap
from shepherd_score.score.electrostatic_scoring import get_overlap_esp
from shepherd_score.score.electrostatic_scoring_np import get_overlap_esp_np
from shepherd_score.pharm_utils.pharmacophore import get_pharmacophores, Pharmacophore
from shepherd_score.score.pharmacophore_scoring_np import get_overlap_pharm_np
from shepherd_score.score.pharmacophore_scoring import _SIM_TYPE, get_overlap_pharm
from shepherd_score.score.similarity import Similarity, tversky_weights
from shepherd_score.alignment._optimize import optimize_se3, tree_map
from shepherd_score.modes import (Mode, get_mode, Vol, VolAvoid, VolESP, Surf, SurfESP, VolAndSurfESP,
                                  Pharm, vol_color, vol_color_avoid, vol_lipo, vol_mr, vol_fukui, vol_atomtype)
from shepherd_score.alignment.utils.se3_np import apply_SE3_transform_np, apply_SO3_transform_np
from shepherd_score.accel import batch as _ba
from shepherd_score.accel._modes import (
    MODE_ATTRS as _MODE_ATTRS,
    CANONICAL_MODES as _CANONICAL_MODES,
    LEGACY_MODE_ALIASES as _LEGACY_MODE_ALIASES,
)
from shepherd_score.container.profiles import Surface


_CHARGE_MODELS = ('xtb_water', 'xtb', 'mmff')
_FEATURE_SETS = ('shepherd', 'rdkit_base')
_SURFACE_METHODS = ('mesh', 'smooth_sdf')

# Alignment modes tracked by MoleculePair (one AlignmentResult each). ``esp``/``esp_combo`` are
# now ``surf_esp``/``vol_and_surf_esp``; the legacy names remain as delegating properties on
# MoleculePair. The bare ``vol``/``vol_esp`` keys hold the with-hydrogen results, ``*_noH`` the
# heavy-atom results.
_ALIGN_KEYS = (
    'vol', 'vol_noH', 'vol_esp', 'vol_esp_noH',
    'surf', 'surf_esp', 'vol_and_surf_esp', 'pharm', 'vol_color', 'vol_tversky',
    'vol_lipo', 'vol_esp_tversky',
    # Experimental modes: directional-pharmacophore, atom-identity and molar-refractivity
    # blends, and the Tversky variants of the remaining modes.
    'vol_pharm', 'vol_atomtype', 'vol_mr',
    'surf_tversky', 'surf_esp_tversky', 'vol_and_surf_esp_tversky',
    'vol_color_tversky', 'vol_lipo_tversky', 'pharm_tversky',
    # shape + condensed-Fukui reactivity field (reuses the vol_lipo overlap; f+ - f- dual descriptor)
    'vol_fukui',
    # shape minus a linear hard-sphere excluded-volume penalty against a fixed avoid-point cloud
    'vol_avoid',
    # vol_color minus the same excluded-volume penalty, and its Tversky variant
    'vol_color_avoid', 'vol_color_avoid_tversky',
)


def _require_jax():
    """
    Import and return ``jax.numpy``, raising a clear error if JAX is unavailable.
    """
    try:
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "The 'jax' backend needs JAX. Install it with: pip install \"shepherd-score[jax]\", "
            "or use backend='analytical' or 'torch' (use_jax=False).") from exc
    return jnp


@dataclass
class AlignmentResult:
    """
    Result of a single alignment mode: the optimal similarity score and SE(3) transform.

    Attributes
    ----------
    score : np.ndarray or None
        Optimally aligned similarity score. ``None`` until an alignment is run.
    transform : np.ndarray
        SE(3) transformation matrix, shape (4, 4). Defaults to the identity.
    aligned : dict or tuple, optional
        The mode's fit inputs under ``transform`` (returned by ``MoleculePair.align``; not kept
        in the per-mode ``transform_*`` / ``sim_aligned_*`` slots).
    """
    score: Optional[np.ndarray] = None
    transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    aligned: Optional[Union[dict, tuple]] = None


def _legacy_backend(use_jax: bool = False, use_analytical: bool = True) -> str:
    """``MoleculePair.align`` backend for the ``use_jax`` / ``use_analytical`` flags."""
    if use_jax:
        return 'jax'
    return 'auto' if use_analytical else 'torch'


def _similarity_slot(base: str, similarity: Similarity, slot: Optional[str] = None) -> str:
    """Result slot for ``align_with_<base>(similarity=...)``: ``<base>_tversky`` for a Tversky
    similarity, as ``MoleculePairBatch`` stores it; otherwise ``slot`` (default ``base``)."""
    if tversky_weights(similarity) is not None and base + '_tversky' in _ALIGN_KEYS:
        return base + '_tversky'
    return slot or base


def _alignment_property(key: str, field_name: str) -> property:
    """Build a property delegating to ``self._alignments[key].<field_name>``."""
    def getter(self):
        return getattr(self._alignments[key], field_name)

    def setter(self, value) -> None:
        setattr(self._alignments[key], field_name, value)

    return property(getter, setter)


def update_mol_coordinates(mol: Chem.Mol, coordinates: Union[List, np.ndarray]) -> Chem.Mol:
    """
    Updates the coordinates of a 3D RDKit mol object with a new set of coordinates

    Parameters
    ----------
    mol : Chem.Mol
        RDKit mol object with 3D coordinates to be replaced
    coordinates : Union[List, np.ndarray]
        List/array of new [x,y,z] coordinates

    Returns
    -------
    mol_new : Chem.Mol
        deep-copied RDKit mol object with updated 3D coordinates
    """
    mol_new = deepcopy(mol)
    conf = mol_new.GetConformer()
    for i in range(mol_new.GetNumAtoms()):
        x,y,z = coordinates[i]
        conf.SetAtomPosition(i, Point3D(x,y,z))
    return mol_new


def _with_bonded_hydrogens(mol: Chem.Mol, atom_indices: np.ndarray) -> np.ndarray:
    """
    Return ``atom_indices`` unioned with the directly-bonded hydrogen neighbor
    of every atom in ``atom_indices``.
    """
    h_idx = set()
    for i in atom_indices.tolist():
        for nbr in mol.GetAtomWithIdx(int(i)).GetNeighbors():
            if nbr.GetAtomicNum() == 1:
                h_idx.add(nbr.GetIdx())
    if not h_idx:
        return atom_indices
    return np.array(sorted(set(atom_indices.tolist()) | h_idx), dtype=np.int64)


class Molecule:
    """
    Molecule contains ways to hold/generate molecule geometries
    """
    def __init__(self,
                 mol: Chem.rdchem.Mol,
                 num_surf_points: Optional[int] = None,
                 density: Optional[float] = None,
                 probe_radius: Optional[float] = None,
                 surface_points: Optional[np.ndarray] = None,
                 partial_charges : Optional[np.ndarray] = None,
                 charge_model: Literal['xtb_water', 'xtb', 'mmff'] = 'xtb_water',
                 electrostatics: Optional[np.ndarray] = None,
                 pharm_multi_vector: Optional[bool] = None,
                 pharm_types: Optional[np.ndarray] = None,
                 pharm_ancs: Optional[np.ndarray] = None,
                 pharm_vecs: Optional[np.ndarray] = None,
                 feature_set: Literal['shepherd', 'rdkit_base'] = 'shepherd',
                 directionless: bool = False,
                 surface_method: Literal['mesh', 'smooth_sdf'] = 'mesh',
                 fukui: Optional[np.ndarray] = None
                 ):
        """
        Molecule constructor to extract interaction profiles.

        If `partial_charges` are not provided, they are generated lazily (only when first needed)
        using `charge_model` -- gfn2-xTB in implicit water by default, MMFF94 as a fallback.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
        num_surf_points : Optional[int]
            Number of surface points to sample.
            If ``None``, the surface point cloud is not generated. More efficient if only doing volumentric.
        density : Optional[float]
            Density of points to sample on molecular surface.
            If ``None``, the surface point cloud is not generated. More efficient if only doing volumentric.
            If both ``num_surf_points`` and ``density`` are not ``None``, ``num_surf_points`` supercedes ``density``.
        surface_points : Optional[np.ndarray]
            Surface points if they were previously generated. Shape: (M,3).
        probe_radius : Optional[float]
            The radius of a probe atom to act as a "solvent accessible surface".
            Default is 1.2 if ``None`` is passed.
        partial_charges : Optional[np.ndarray]
            Partial charges for each atom. Shape: (N,).
            If ``None``, charges are generated lazily from ``charge_model`` the first time they are
            needed (an ESP mode, or the surface ESP built when a surface is generated).
        charge_model : {'xtb_water', 'xtb', 'mmff'}
            Charge model used when ``partial_charges`` is ``None``: ``'xtb_water'`` (default) runs
            a gfn2-xTB single point in ALPB water, ``'xtb'`` runs it in vacuum, and both fall back
            to MMFF94 with a warning if ``xtb`` is unavailable; ``'mmff'`` uses MMFF94 directly.
            xTB uses the molecule's formal charge. Charges are computed lazily either way.
        electrostatics : Optional[np.ndarray]
            Electrostatic potential if they were previously generated. Shape: (M,).
        pharm_multi_vector : Optional[bool]
            If ``None``, don't generate pharmacophore, else generate
            pharmacophore with/without (``True``/``False``) multi-vectors.
        pharm_types : Optional[np.ndarray]
            Types of pharmacophore. Shape: (P,).
        pharm_ancs : Optional[np.ndarray]
            Anchor positions of pharmacophore. Shape: (P,3).
        pharm_vecs : Optional[np.ndarray]
            Unit vectors relative to anchor positions of pharmacophore. Shape: (P,3).
        feature_set : {'shepherd', 'rdkit_base'}
            Pharmacophore feature definition used when pharmacophores are generated:
            ``'shepherd'`` (default) is the local ``smarts_features.fdef`` (8 types);
            ``'rdkit_base'`` is RDKit's ``BaseFeatures.fdef`` reduced to the 6 ROCS color types.
        directionless : bool
            When ``True``, generate isotropic (zero-vector) "color" pharmacophores for all
            families, overriding ``pharm_multi_vector``. Default ``False``. Only used when
            pharmacophores are generated.
        surface_method : {'mesh', 'smooth_sdf'}
            How to generate the surface point cloud. ``'mesh'`` (default) uses Open3D ball
            pivoting plus Poisson-disk resampling; ``'smooth_sdf'`` uses the Open3D-free
            ``generate_point_cloud.get_molecular_surface_smooth_sdf`` and requires
            ``num_surf_points`` rather than ``density``. A model trained on the mesh surface sees
            the smooth surface as a distribution shift.
        fukui : Optional[np.ndarray]
            Per-atom condensed Fukui dual descriptor (f+ - f-) for the ``vol_fukui`` mode, in the
            same with-H order as ``partial_charges``. Shape: (N,). If ``None`` it is generated
            lazily on first access from three gfn2-xTB single points (in ALPB water when
            ``charge_model='xtb_water'``).
        """
        self.mol = mol
        self.atom_pos = Chem.RemoveHs(mol).GetConformer().GetPositions()
        if surface_points is None:
            self.num_surf_points = num_surf_points
        else:
            self.num_surf_points = len(surface_points)
        self.density = density
        if surface_method not in _SURFACE_METHODS:
            raise ValueError(
                f"surface_method must be one of {_SURFACE_METHODS}, got {surface_method!r}")
        if feature_set not in _FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {_FEATURE_SETS}, got {feature_set!r}")
        self.surface_method = surface_method

        if isinstance(partial_charges, list):
            partial_charges = np.array(partial_charges)

        # Charges are generated lazily by the ``partial_charges`` property, so shape-only
        # molecules never pay for them.
        self._charge_model = str(charge_model).lower()
        if self._charge_model not in _CHARGE_MODELS:
            raise ValueError(f"charge_model must be one of {_CHARGE_MODELS}, got {charge_model!r}")
        if isinstance(partial_charges, np.ndarray):
            self._partial_charges = partial_charges
        else:
            self._partial_charges = None                       # deferred
        # Fukui field, generated lazily by the ``fukui`` property like ``partial_charges``.
        if isinstance(fukui, list):
            fukui = np.array(fukui)
        self._fukui = fukui if isinstance(fukui, np.ndarray) else None   # deferred
        # Per-atom Crippen logP and molar-refractivity contributions, (N,) in with-H order like
        # ``partial_charges`` (the ``vol_lipo`` and ``vol_mr`` channels).
        self.lipophilicity = self.get_lipophilicity_contribs()
        self.molar_refractivity = self.get_molar_refractivity_contribs()
        self.radii = get_atomic_vdw_radii(mol)

        self._surface = Surface(
            positions=None,
            esp=None,
            probe_radius=probe_radius if probe_radius is not None else 1.2,
        )
        if surface_points is None:
            if isinstance(num_surf_points, int):
                self.surf_pos = self.get_pc()
            elif isinstance(density, float):
                self.surf_pos = self.get_pc(use_density=True)
            # else: no point cloud (surf_pos/surf_esp stay None)
        else:
            self.surf_pos = surface_points

        if self.surf_pos is not None and self.partial_charges is not None:
            if not isinstance(electrostatics, np.ndarray):
                self.surf_esp = self.get_electrostatic_potential()
            else:
                self.surf_esp = electrostatics

        # Indices for atoms that aren't hydrogens
        self._nonH_atoms_idx = np.array([a.GetIdx() for a in self.mol.GetAtoms() if a.GetAtomicNum() != 1])

        self.pharm_multi_vector = pharm_multi_vector
        if isinstance(pharm_types, np.ndarray) and isinstance(pharm_ancs, np.ndarray) and isinstance(pharm_vecs, np.ndarray):
            self._pharmacophore = Pharmacophore(types=pharm_types,
                                                  positions=pharm_ancs,
                                                  vectors=pharm_vecs)
        else:
            self._pharmacophore = None
            if self.pharm_multi_vector is not None:
                self.get_pharmacophore(
                    multi_vector=self.pharm_multi_vector,
                    exclude=[],
                    check_access=False,
                    scale=1.,
                    feature_set=feature_set,
                    directionless=directionless
                )


    def __setstate__(self, state):
        """
        Restore a pickled Molecule, upgrading the older flat layout.

        Older pickles carry ``surf_pos`` / ``surf_esp`` / ``probe_radius`` / ``pharm_types`` /
        ``pharm_ancs`` / ``pharm_vecs`` directly in ``__dict__``. Those names are now data
        descriptors forwarding to ``self._surface`` / ``self._pharmacophore``, and a descriptor
        takes precedence over the instance dict, so without this hook every such read on an old
        pickle raises AttributeError.
        """
        state = dict(state)

        if "_surface" not in state:
            state["_surface"] = Surface(
                positions=state.pop("surf_pos", None),
                esp=state.pop("surf_esp", None),
                probe_radius=state.pop("probe_radius", 1.2),
            )
        else:
            for _k in ("surf_pos", "surf_esp", "probe_radius"):
                state.pop(_k, None)

        if "_pharmacophore" not in state:
            _t = state.pop("pharm_types", None)
            _a = state.pop("pharm_ancs", None)
            _v = state.pop("pharm_vecs", None)
            state["_pharmacophore"] = (
                None if _t is None and _a is None and _v is None
                else Pharmacophore(types=_t, positions=_a, vectors=_v)
            )
        else:
            for _k in ("pharm_types", "pharm_ancs", "pharm_vecs"):
                state.pop(_k, None)

        # Attributes added after the flat layout, defaulted to the constructor's values so an
        # old pickle does not raise on first read.
        state.setdefault("surface_method", "mesh")
        if "partial_charges" in state:
            state.setdefault("_partial_charges", state.pop("partial_charges"))
        state.setdefault("_partial_charges", None)
        state.setdefault("_charge_model", "xtb_water")
        state.setdefault("_fukui", None)

        self.__dict__.update(state)

    # Interaction-profile accessors (backwards-compatible with the loose
    # ``surf_pos``/``surf_esp``/``probe_radius`` and ``pharm_*`` attributes)
    @property
    def surface(self) -> Surface:
        """The :class:`Surface` holding surface positions, ESP, and probe radius."""
        return self._surface

    @property
    def surf_pos(self) -> Optional[np.ndarray]:
        return self._surface.positions

    @surf_pos.setter
    def surf_pos(self, value: Optional[np.ndarray]) -> None:
        self._surface.positions = value

    @property
    def surf_esp(self) -> Optional[np.ndarray]:
        return self._surface.esp

    @surf_esp.setter
    def surf_esp(self, value: Optional[np.ndarray]) -> None:
        self._surface.esp = value

    @property
    def probe_radius(self) -> float:
        return self._surface.probe_radius

    @probe_radius.setter
    def probe_radius(self, value: float) -> None:
        self._surface.probe_radius = value

    @property
    def pharmacophore(self) -> Optional[Pharmacophore]:
        """The :class:`Pharmacophore` container, or ``None`` if not generated."""
        return self._pharmacophore

    def _ensure_pharm_container(self) -> Pharmacophore:
        """Lazily create an empty :class:`Pharmacophore` so setters can populate it."""
        if self._pharmacophore is None:
            self._pharmacophore = Pharmacophore(types=None, positions=None, vectors=None)
        return self._pharmacophore

    @property
    def pharm_types(self) -> Optional[np.ndarray]:
        return None if self._pharmacophore is None else self._pharmacophore.types

    @pharm_types.setter
    def pharm_types(self, value: Optional[np.ndarray]) -> None:
        self._ensure_pharm_container().types = value

    @property
    def pharm_ancs(self) -> Optional[np.ndarray]:
        return None if self._pharmacophore is None else self._pharmacophore.positions

    @pharm_ancs.setter
    def pharm_ancs(self, value: Optional[np.ndarray]) -> None:
        self._ensure_pharm_container().positions = value

    @property
    def pharm_vecs(self) -> Optional[np.ndarray]:
        return None if self._pharmacophore is None else self._pharmacophore.vectors

    @pharm_vecs.setter
    def pharm_vecs(self, value: Optional[np.ndarray]) -> None:
        self._ensure_pharm_container().vectors = value

    @property
    def partial_charges(self) -> np.ndarray:
        """Per-atom partial charges (all atoms, with-H order), computed lazily on first access.

        Generated by ``charge_model`` (gfn2-xTB in water by default, falling back to MMFF94 with a warning
        if ``xtb`` is unavailable or fails) and cached. Pass ``charge_model='mmff'`` or explicit
        ``partial_charges`` to skip xTB.
        """
        if self._partial_charges is None:
            self._partial_charges = self._generate_partial_charges()
        return self._partial_charges

    @partial_charges.setter
    def partial_charges(self, value) -> None:
        self._partial_charges = value

    def _generate_partial_charges(self) -> np.ndarray:
        """Generate charges per ``self._charge_model`` (default 'xtb_water'); MMFF94 fallback on failure."""
        if self._charge_model in ("xtb", "xtb_water"):
            try:
                from shepherd_score.conformer_generation import (
                    charges_from_single_point_conformer_with_xtb)
                return np.asarray(
                    charges_from_single_point_conformer_with_xtb(
                        self.mol, solvent=self._xtb_solvent,
                        charge=int(Chem.GetFormalCharge(self.mol))),
                    dtype=np.float32)
            except Exception as e:
                import warnings
                warnings.warn(
                    f"xTB partial charges unavailable ({type(e).__name__}: {e}); falling back to "
                    "MMFF94. Install xtb or pass charge_model='mmff' to silence this.",
                    RuntimeWarning)
        return self.get_partial_charges()

    def get_partial_charges(self) -> np.ndarray:
        """
        Get the partial charges on each atom using MMFF.
        """
        mol_copy = deepcopy(self.mol)
        molec_props = Chem.AllChem.MMFFGetMoleculeProperties(mol_copy)
        if molec_props is None:
            # MMFF94 cannot parameterize this molecule (e.g. H2 or a lone noble-gas atom); fail
            # with an actionable message instead of an AttributeError on None.
            raise ValueError(
                "MMFF94 could not parameterize this molecule, so partial charges cannot be "
                "computed (this happens for inputs MMFF does not cover, e.g. H2 or a single "
                "noble-gas atom). Pass partial_charges=... explicitly if you need to score it."
            )
        charges = np.array([molec_props.GetMMFFPartialCharge(i) for i, _ in enumerate(mol_copy.GetAtoms())])
        return charges.astype(np.float32)


    @property
    def fukui(self) -> np.ndarray:
        """Per-atom condensed Fukui dual descriptor ``f+ - f-`` (all atoms, with-H order), computed
        lazily on first access from three gfn2-xTB single points and cached. Positive at sites susceptible to
        nucleophilic attack and negative at sites susceptible to electrophilic attack. Pass ``fukui=...`` to the constructor to
        skip xTB.
        """
        if self._fukui is None:
            self._fukui = self._generate_fukui()
        return self._fukui

    @fukui.setter
    def fukui(self, value) -> None:
        self._fukui = value

    @property
    def _xtb_solvent(self) -> Optional[str]:
        """ALPB solvent for xTB calls: ``'water'`` under ``charge_model='xtb_water'``."""
        return "water" if self._charge_model == "xtb_water" else None

    def _generate_fukui(self) -> np.ndarray:
        """Condensed Fukui dual descriptor (f+ - f-) per atom from three gfn2-xTB single points
        (the molecular charge and its adjacent charge states) at the same geometry. Unlike charges there is no non-QM fallback,
        so any xTB failure (e.g. a non-converging ion) propagates.
        """
        from shepherd_score.conformer_generation import fukui_from_single_point_conformer_with_xtb
        fpm = fukui_from_single_point_conformer_with_xtb(
            self.mol, solvent=self._xtb_solvent, charge=int(Chem.GetFormalCharge(self.mol)))   # (N,3) = [f+, f-, f0]
        return (fpm[:, 0] - fpm[:, 1]).astype(np.float32)           # signed dual descriptor f+ - f-


    def get_positions(self, no_H: bool = True) -> np.ndarray:
        """
        Get atom coordinates with or without hydrogens.

        Parameters
        ----------
        no_H : bool, optional
            If ``True`` (default) return the cached heavy-atom coordinates (``atom_pos``).
            If ``False`` return all-atom coordinates from the conformer (including H).

        Returns
        -------
        np.ndarray
            Atom coordinates. Shape: (N, 3).
        """
        if no_H:
            return self.atom_pos
        return self.mol.GetConformer().GetPositions()


    def get_charges(self, no_H: bool = True) -> np.ndarray:
        """
        Get partial charges with or without hydrogens.

        This slices the already-computed ``partial_charges``; it does not recompute them
        (see :meth:`get_partial_charges` for MMFF computation).

        Parameters
        ----------
        no_H : bool, optional
            If ``True`` (default) return charges for heavy atoms only.
            If ``False`` return charges for all atoms (including H).

        Returns
        -------
        np.ndarray
            Partial charges. Shape: (N,).
        """
        if no_H:
            return self.partial_charges[self._nonH_atoms_idx]
        return self.partial_charges


    def get_lipophilicity_contribs(self) -> np.ndarray:
        """
        Get the per-atom Crippen atomic logP contribution for each atom.

        Uses ``rdMolDescriptors._CalcCrippenContribs`` (one ``(logP, MR)`` tuple per atom in
        with-H order) and keeps the ``logP`` element, in the same order as
        :attr:`partial_charges`.

        Returns
        -------
        np.ndarray
            Per-atom Crippen logP contributions. Shape: (N,).
        """
        contribs = rdMolDescriptors._CalcCrippenContribs(self.mol)
        return np.array([c[0] for c in contribs], dtype=np.float32)


    def get_lipophilicity(self, no_H: bool = True) -> np.ndarray:
        """
        Get per-atom Crippen logP contributions with or without hydrogens.

        Slices the cached ``lipophilicity``; mirrors :meth:`get_charges`.

        Parameters
        ----------
        no_H : bool, optional
            If ``True`` (default) return contributions for heavy atoms only.
            If ``False`` return contributions for all atoms (including H).

        Returns
        -------
        np.ndarray
            Per-atom Crippen logP contributions. Shape: (N,).
        """
        if no_H:
            return self.lipophilicity[self._nonH_atoms_idx]
        return self.lipophilicity


    def get_lipo_positions(self) -> np.ndarray:
        """Get the heavy-atom coordinates that carry the per-atom Crippen logP (the ``vol_lipo``
        channel's centers).

        Indexes the with-H conformer by ``_nonH_atoms_idx`` so the positions stay 1:1 with
        :meth:`get_lipophilicity`. ``atom_pos`` is not used because ``Chem.RemoveHs`` retains
        isotope-labelled hydrogens, which would misalign it with the heavy-atom logP.

        Returns
        -------
        np.ndarray
            Heavy-atom coordinates. Shape: (N_heavy, 3).
        """
        return self.mol.GetConformer().GetPositions()[self._nonH_atoms_idx]


    def get_molar_refractivity_contribs(self) -> np.ndarray:
        """
        Get the per-atom Crippen atomic molar-refractivity (MR) contribution for each atom.

        Uses ``rdMolDescriptors._CalcCrippenContribs`` and keeps the ``MR`` element, in the same
        with-H order as :attr:`partial_charges`. Atomic MR is a size/polarizability descriptor.

        Returns
        -------
        np.ndarray
            Per-atom Crippen molar-refractivity contributions. Shape: (N,).
        """
        contribs = rdMolDescriptors._CalcCrippenContribs(self.mol)
        return np.array([c[1] for c in contribs], dtype=np.float32)


    def get_molar_refractivity(self, no_H: bool = True) -> np.ndarray:
        """
        Get per-atom Crippen molar-refractivity contributions with or without hydrogens.

        Slices the cached ``molar_refractivity``; mirrors :meth:`get_lipophilicity`.

        Parameters
        ----------
        no_H : bool, optional
            If ``True`` (default) return contributions for heavy atoms only.
            If ``False`` return contributions for all atoms (including H).

        Returns
        -------
        np.ndarray
            Per-atom Crippen molar-refractivity contributions. Shape: (N,).
        """
        if no_H:
            return self.molar_refractivity[self._nonH_atoms_idx]
        return self.molar_refractivity


    def get_mr_positions(self) -> np.ndarray:
        """Get the heavy-atom coordinates that carry the per-atom Crippen molar refractivity (the
        ``vol_mr`` channel's centers); same basis as :meth:`get_lipo_positions`."""
        return self.mol.GetConformer().GetPositions()[self._nonH_atoms_idx]


    def get_fukui(self, no_H: bool = True) -> np.ndarray:
        """Get the per-atom Fukui dual-descriptor field with or without hydrogens.

        Slices the cached :attr:`fukui`; mirrors :meth:`get_charges`.

        Parameters
        ----------
        no_H : bool, optional
            If ``True`` (default) return the field for heavy atoms only; else all atoms (with H).

        Returns
        -------
        np.ndarray
            Per-atom Fukui dual descriptor. Shape: (N,).
        """
        if no_H:
            return self.fukui[self._nonH_atoms_idx]
        return self.fukui


    def get_fukui_positions(self) -> np.ndarray:
        """Get the heavy-atom coordinates that carry the per-atom Fukui field (the ``vol_fukui``
        channel's centers); same basis as :meth:`get_lipo_positions`."""
        return self.mol.GetConformer().GetPositions()[self._nonH_atoms_idx]


    def get_atomtype_positions(self) -> np.ndarray:
        """Get the heavy-atom coordinates that carry the per-atom element labels (the
        ``vol_atomtype`` channel's centers); 1:1 with :meth:`get_atomic_numbers` and the same basis
        as :meth:`get_lipo_positions`."""
        return self.mol.GetConformer().GetPositions()[self._nonH_atoms_idx]


    def get_atomic_numbers(self, no_H: bool = True) -> np.ndarray:
        """
        Get per-atom atomic numbers as a float32 array, the categorical label the ``vol_atomtype``
        channel matches on.

        Sliced with the same ``_nonH_atoms_idx`` as the charges, so the labels stay 1:1 with
        :meth:`get_atomtype_positions`. Returned as float32 to sit alongside the other per-atom
        channels in a batch tensor.

        Parameters
        ----------
        no_H : bool, optional
            If ``True`` (default) return atomic numbers for heavy atoms only.
            If ``False`` return atomic numbers for all atoms (including H).

        Returns
        -------
        np.ndarray
            Per-atom atomic numbers. Shape: (N,).
        """
        Z = np.array([a.GetAtomicNum() for a in self.mol.GetAtoms()], dtype=np.float32)
        if no_H:
            return Z[self._nonH_atoms_idx]
        return Z


    def get_pc(self,
              use_density=False,
              atom_indices: Optional[Iterable[int]] = None,
              radial_buffer: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Gets the point cloud positions.

        Parameters
        ----------
        use_density : bool, optional
            Whether to sample at a fixed point density instead of a fixed count. Default ``False``.
        atom_indices : Optional[Iterable[int]]
            If given, restricts surface generation to only these atom indices
            (original-mol atom-index space, same as ``partial_charges``).
            ``None`` (default) uses all atoms, preserving prior behavior.
        radial_buffer : float, optional
            Flat padding (in Angstroms) added to every atom's vdW radius before
            surface generation. Default ``0.0`` (no change).
        """
        self.mol, centers = get_atom_coords(self.mol, MMFF_optimize=False)
        surface_method = self.surface_method
        radii = self.radii
        if atom_indices is not None:
            idx = np.asarray(sorted({int(i) for i in atom_indices}), dtype=np.int64)
            centers, radii = centers[idx], radii[idx]
        if radial_buffer:
            radii = radii + radial_buffer
        if use_density:
            if surface_method != 'mesh':
                raise ValueError(
                    f"surface_method={surface_method!r} does not support density-based surfaces; "
                    "the mesh-free path needs num_surf_points. Use surface_method='mesh' with density, "
                    "or pass num_surf_points instead of density."
                )
            positions = get_molecular_surface_const_density(centers,
                                                            radii,
                                                            self.density,
                                                            probe_radius=self.probe_radius,
                                                            num_samples_per_atom=25)
        else:
            # num_samples_per_atom is left to each method's default.
            positions = get_molecular_surface(centers,
                                              radii,
                                              num_points=self.num_surf_points,
                                              probe_radius=self.probe_radius,
                                              method=surface_method)
        return positions.astype(np.float32)


    def get_electrostatic_potential(self,
                                    atom_indices: Optional[Iterable[int]] = None,
                                    surf_pos: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get the electrostatic potential at each surface point.

        Parameters
        ----------
        atom_indices : Optional[Iterable[int]]
            If given, restricts the atoms (and their charges) contributing to
            the potential to only these indices (original-mol atom-index
            space, same as ``partial_charges``). ``None`` (default) uses all
            atoms, preserving prior behavior.
        surf_pos : Optional[np.ndarray]
            Surface points to evaluate the potential at. Defaults to
            ``self.surf_pos``; pass an explicit array when evaluating against
            a freshly regenerated surface not yet assigned to ``self``.
        """
        centers = self.mol.GetConformer().GetPositions()
        charges = self.partial_charges
        if atom_indices is not None:
            idx = np.asarray(sorted({int(i) for i in atom_indices}), dtype=np.int64)
            centers, charges = centers[idx], charges[idx]
        surf_pos = self.surf_pos if surf_pos is None else surf_pos
        distances = np.linalg.norm(surf_pos[:, np.newaxis] - centers, axis=2)
        # Calculate the potentials
        E_pot = np.dot(charges, 1 / distances.T) * COULOMB_SCALING
        # Ensure that invalid distances (where distance is 0) are handled
        E_pot[np.isinf(E_pot)] = 0
        return E_pot.astype(np.float32)


    def center_to(self, xyz_means: np.ndarray) -> None:
        """
        If you want to center the molecule with respect to a certain coordinate frame.
        """
        self.atom_pos -= xyz_means
        trans = np.eye(4)
        trans[:3,3] = -xyz_means
        Chem.rdMolTransforms.TransformConformer(self.mol.GetConformer(), trans)
        if self.surf_pos is not None:
            self.surf_pos -= xyz_means
        if self.pharm_ancs is not None:
            self.pharm_ancs -= xyz_means


    def get_pharmacophore(self,
                          multi_vector: bool = True,
                          exclude: List[int] = [],
                          check_access: bool = False,
                          scale: float = 1,
                          feature_set: Literal['shepherd', 'rdkit_base'] = 'shepherd',
                          directionless: bool = False,
                          return_atom_ids: bool = False,
                          priority_atoms: Optional[Iterable[int]] = None,
                          min_ring_priority_atoms: int = 3):
        """
        Get the pharmacophore of the molecule.

        Stores the full :class:`~shepherd_score.pharm_utils.pharmacophore.Pharmacophore`
        container on ``self`` (accessible via :attr:`pharmacophore`). The
        ``pharm_types``/``pharm_ancs``/``pharm_vecs`` properties delegate to it, so
        existing usage is unchanged.

        Parameters
        ----------
        multi_vector : bool, optional
            Whether to represent pharmacophore with multiple vectors. Default ``True``.
        exclude : list, optional
            Hydrogen indices to not include as a HBD. Default ``[]``.
        check_access : bool, optional
            Check if HBD/HBA are accessible to the molecular surface. Default ``False``.
        scale : float, optional
            Length of a pharmacophore vector in Angstroms. Default 1.
        feature_set : {'shepherd', 'rdkit_base'}, optional
            ``'shepherd'`` (default, 8 types) or ``'rdkit_base'`` (6 ROCS color types).
            See :func:`get_pharmacophores`.
        directionless : bool, optional
            When ``True``, emit isotropic zero-vector "color" pharmacophores, overriding
            ``multi_vector``. Default ``False``.
        return_atom_ids : bool, optional
            Retain per-pharmacophore atom-id sets on ``self.pharmacophore.atom_ids``,
            enabling ``self.pharmacophore.priority_labels(...)``. Default ``False``.
        priority_atoms : iterable of int, optional
            When provided, priority labels are computed and stored on
            ``self.pharmacophore.labels``. Default ``None``.
        min_ring_priority_atoms : int, optional
            Only used when ``priority_atoms`` is provided. See
            :func:`~shepherd_score.pharm_utils.pharmacophore.get_pharmacophore`.
            Default 3.
        """
        self._pharmacophore = get_pharmacophores(
            self.mol,
            multi_vector=multi_vector,
            exclude=exclude,
            check_access=check_access,
            scale=scale,
            feature_set=feature_set,
            directionless=directionless,
            return_atom_ids=return_atom_ids,
            priority_atoms=priority_atoms,
            min_ring_priority_atoms=min_ring_priority_atoms,
        )


    def select_atoms(self,
                     atom_indices: Iterable[int],
                     surface_atom_indices: Optional[Iterable[int]] = None,
                     expand_pharm_consistent: bool = True,
                     min_ring_priority_atoms: int = 1,
                     restrict_esp: bool = True,
                     radial_buffer: float = 0.0,
                     include_h: bool = False) -> 'Molecule':
        """
        Return a new :class:`Molecule` whose surface, ESP, and pharmacophore
        data are restricted to a chosen atom subset.

        Parameters
        ----------
        atom_indices : Iterable[int]
            Seed atom indices (original-mol atom-index space, same as
            ``partial_charges`` and ``Pharmacophore.atom_ids``) that drive the
            pharmacophore subselection. Also used for the surface/ESP
            restriction when ``surface_atom_indices`` is ``None``.
        surface_atom_indices : Optional[Iterable[int]]
            If given, used in place of ``atom_indices`` to restrict the
            surface/ESP -- decoupling which atoms define the retained
            pharmacophores from which atoms the surface/ESP are regenerated
            over. Expanded via ``expand_pharm_consistent`` the same way
            ``atom_indices`` is independently.
        expand_pharm_consistent : bool, optional
            Whether to expand ``atom_indices`` (and, independently,
            ``surface_atom_indices`` when given) to the pharmacophore-consistent
            superset via
            :meth:`~shepherd_score.pharm_utils.pharmacophore.Pharmacophore.expand_atom_selection`
            before restricting the pharmacophore / surface / ESP.
            Default ``True``.
        min_ring_priority_atoms : int, optional
            Forwarded to the underlying :class:`Pharmacophore` expansion/filtering
            calls. Default ``1`` (see
            :meth:`~shepherd_score.pharm_utils.pharmacophore.Pharmacophore.expand_atom_selection`).
        restrict_esp : bool, optional
            If ``True`` (default), the electrostatic potential is recomputed
            using only the selected atoms' charges. If ``False``, ESP is still
            recomputed at the restricted surface but using all atoms' charges.
        radial_buffer : float, optional
            Flat padding (in Angstroms) added to every retained atom's vdW
            radius before regenerating the surface. Default ``0.0`` (no change).
        include_h : bool, optional
            If ``True``, the directly-bonded hydrogen neighbors of every atom
            in the surface/ESP atom set are added before regenerating the
            surface/ESP. Default ``False``.

        Returns
        -------
        Molecule
            A new ``Molecule`` instance; ``self`` is not modified.
        """
        pharm_idx = np.asarray(sorted({int(i) for i in atom_indices}), dtype=np.int64)

        pharm = self.pharmacophore
        if pharm is not None and pharm.atom_ids is None:
            self.get_pharmacophore(multi_vector=self.pharm_multi_vector, return_atom_ids=True)
            pharm = self.pharmacophore

        if expand_pharm_consistent and pharm is not None:
            pharm_idx = pharm.expand_atom_selection(pharm_idx, min_ring_priority_atoms=min_ring_priority_atoms)

        if surface_atom_indices is None:
            surf_idx = pharm_idx
        else:
            surf_idx = np.asarray(sorted({int(i) for i in surface_atom_indices}), dtype=np.int64)
            if expand_pharm_consistent and pharm is not None:
                surf_idx = pharm.expand_atom_selection(surf_idx, min_ring_priority_atoms=min_ring_priority_atoms)

        if include_h:
            surf_idx = _with_bonded_hydrogens(self.mol, surf_idx)

        new_surf_pos = None
        if self.surf_pos is not None:
            new_surf_pos = self.get_pc(atom_indices=surf_idx, radial_buffer=radial_buffer)

        new_esp = None
        if self.surf_esp is not None and new_surf_pos is not None:
            esp_idx = surf_idx if restrict_esp else None
            new_esp = self.get_electrostatic_potential(atom_indices=esp_idx, surf_pos=new_surf_pos)

        new_pharm = (
            pharm.subset_to_atoms(pharm_idx, min_ring_priority_atoms=min_ring_priority_atoms)
            if pharm is not None else None
        )

        return Molecule(
            mol=self.mol,
            probe_radius=self.probe_radius,
            surface_points=new_surf_pos,
            partial_charges=self.partial_charges,
            electrostatics=new_esp,
            pharm_multi_vector=self.pharm_multi_vector,
            pharm_types=None if new_pharm is None else new_pharm.types,
            pharm_ancs=None if new_pharm is None else new_pharm.positions,
            pharm_vecs=None if new_pharm is None else new_pharm.vectors,
            # Carry the remaining instance state so regeneration on the subset matches the
            # source; feature_set/directionless are call-time arguments, not instance state.
            charge_model=self._charge_model,
            surface_method=self.surface_method,
            fukui=self._fukui,
        )


def _bind_batch_aligners(cls):
    """Bind ``accel.batch._align_batch_<mode>`` onto ``cls`` as static methods, one per registry
    mode plus the legacy aliases (esp -> surf_esp, esp_combo -> vol_and_surf_esp), so
    ``MoleculePair._align_batch_vol(pairs, ...)`` resolves and adding a mode needs no edit here."""
    for _m in _CANONICAL_MODES:
        setattr(cls, "_align_batch_" + _m, staticmethod(getattr(_ba, "_align_batch_" + _m)))
    for _legacy, _canon in _LEGACY_MODE_ALIASES.items():
        setattr(cls, "_align_batch_" + _legacy, staticmethod(getattr(_ba, "_align_batch_" + _canon)))
    return cls


@_bind_batch_aligners
class MoleculePair:
    """ Pair of Molecule objects to facilitate alignment. """

    def __init__(self,
                 ref_mol: Union[Chem.rdchem.Mol, Molecule],
                 fit_mol: Union[Chem.rdchem.Mol, Molecule],
                 num_surf_points: Optional[int] = None,
                 density: Optional[float] = None,
                 do_center: bool = False,
                 device = -1):
        """
        A pair of molecules. A refence molecule and a fit molecule that can be aligned to the fit.
        There are a number of alignments that can be done:

        - Volumetric (with and without hydrogens)
        - Volumetric with partial charge weighting (with and without hydrogens)
        - Surface
        - Surface with electrostatic potential weighting
        - ShaEP scoring (esp-combo)
        - Pharmacophore (with various settings for using extended points rather than vectors)

        Similarly, you can score with surface, Surf+ESP, and pharmacophore

        Parameters
        ----------
        ref_mol : Union[rdkit.Chem.rdchem.Mol, container.Molecule]
            Reference molecule.
            If a RDKit Mol object is provided, it will be converted to a Molecule
            object. If a Molecule object is given, it will NOT regenerate the surface.
        fit_mol : Union[rdkit.Chem.rdchem.Mol, container.Molecule]
            Molecule to fit to the reference.
            If a RDKit Mol object is provided, it will be converted to a Molecule
            object. If a Molecule object is given, it will NOT regenerate the surface.
        num_surf_points : Optional[int] (default = None)
            Number of surface points to sample if rdkit Mol objects are given.
            MUST provide a value for surface or ESP alignment.
        density : Optional[float] (default = None)
            Density of points to sample if rdkit Mol objects are given.
            An integer intput for num_surf_points supercedes the density call.
        do_center : bool (default = False)
            THIS IS CRUCIAL
            Whether to initially align molecule centers together. For global optimizations, set to
            True. For scoring of current alignment or local alignment set to False.
        device : pytorch Device (default = -1)
            Device to use if you want to align with PyTorch downstream.
            Default places alignment computation on CPU.
        """
        # Generate surfaces if not a Molecule object
        if not isinstance(ref_mol, Chem.rdchem.Mol):
            self.ref_molec = ref_mol
        else:
            self.ref_molec = Molecule(ref_mol, num_surf_points=num_surf_points, density=density)
        if not isinstance(fit_mol, Chem.rdchem.Mol):
            self.fit_molec = fit_mol
        else:
            self.fit_molec = Molecule(fit_mol, num_surf_points=num_surf_points, density=density)

        self.num_surf_points = num_surf_points
        self.density = density
        if density is not None and num_surf_points is None:
            self.num_surf_points = True
        if not isinstance(device, torch.device):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Center to origin; ref_shift records how far the reference moved. An avoid cloud is
        # never shifted for you: give it in this (centered) frame, e.g. ``cloud - pair.ref_shift``.
        self.ref_shift = np.zeros(3)
        if do_center:
            self.ref_shift = self.ref_molec.atom_pos.mean(0)
            self.ref_molec.center_to(self.ref_shift)
            self.fit_molec.center_to(self.fit_molec.atom_pos.mean(0))

        # Atomic coordinates as float32 tensors on the target device.
        self._ref_xyz_t = torch.as_tensor(self.ref_molec.atom_pos,
                                          dtype=torch.float32,
                                          device=device)          # (N,3)
        self._fit_xyz_t = torch.as_tensor(self.fit_molec.atom_pos,
                                          dtype=torch.float32,
                                          device=device)          # (M,3)

        # One AlignmentResult per mode (score defaults to None, transform to identity).
        # transform_<mode>/sim_aligned_<mode> are properties delegating to this dict.
        self._alignments = {key: AlignmentResult() for key in _ALIGN_KEYS}


    def __setstate__(self, state):
        """
        Restore a pickled MoleculePair, upgrading the older flat layout.

        Same hazard as :meth:`Molecule.__setstate__`: ``transform_<mode>`` / ``sim_aligned_<mode>``
        are now data descriptors delegating to ``self._alignments``, so an old pickle would raise
        AttributeError on first read. Flat keys (including the legacy ``esp`` / ``esp_combo``
        names) are remapped into AlignmentResult entries when ``_alignments`` is absent, and
        stripped either way so a stale duplicate cannot shadow the canonical entry.
        """
        state = dict(state)
        state.setdefault('ref_shift', np.zeros(3))
        aliases = {'esp': 'surf_esp', 'esp_combo': 'vol_and_surf_esp'}

        if '_alignments' not in state:
            alignments = {key: AlignmentResult() for key in _ALIGN_KEYS}
            for flat, canon in [(k, k) for k in _ALIGN_KEYS] + list(aliases.items()):
                t = state.get(f'transform_{flat}')
                sc = state.get(f'sim_aligned_{flat}')
                if t is not None:
                    alignments[canon].transform = t
                if sc is not None:
                    alignments[canon].score = sc
            state['_alignments'] = alignments

        for flat in list(_ALIGN_KEYS) + list(aliases):
            state.pop(f'transform_{flat}', None)
            state.pop(f'sim_aligned_{flat}', None)

        self.__dict__.update(state)


    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """Convert a numpy array to a float32 tensor on this pair's device."""
        return torch.from_numpy(arr).to(torch.float32).to(self.device)

    # Alignment-result accessors (backwards-compatible with the loose
    # ``transform_*`` / ``sim_aligned_*`` attributes).
    for _key in _ALIGN_KEYS:
        locals()[f'transform_{_key}'] = _alignment_property(_key, 'transform')
        locals()[f'sim_aligned_{_key}'] = _alignment_property(_key, 'score')
    del _key

    # Legacy result-attribute aliases (esp -> surf_esp, esp_combo -> vol_and_surf_esp).
    transform_esp = _alignment_property('surf_esp', 'transform')
    sim_aligned_esp = _alignment_property('surf_esp', 'score')
    transform_esp_combo = _alignment_property('vol_and_surf_esp', 'transform')
    sim_aligned_esp_combo = _alignment_property('vol_and_surf_esp', 'score')

    # Batched aligners (``_align_batch_<mode>``) live in ``accel.batch`` and are bound as static
    # methods by ``@_bind_batch_aligners``.

    def _as_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """Tensor on this pair's device; floating arrays become float32, others keep their dtype."""
        t = torch.from_numpy(np.ascontiguousarray(arr))
        return (t.to(torch.float32) if t.is_floating_point() else t).to(self.device)

    def _mode_inputs(self, mode: Mode, convert) -> tuple:
        # A Combined mode has no class-level name; its label ('vol_color', ...) identifies it
        what = mode.name or getattr(mode, 'label', None) or type(mode).__name__
        for req in mode.requires:
            if req == 'surface' and self.num_surf_points is None:
                raise ValueError(f'{what} needs surfaces, but the Molecule '
                                 'objects were initialized with no surface points.')
            if req == 'pharmacophores' and (self.ref_molec.pharm_types is None
                                            or self.fit_molec.pharm_types is None):
                raise ValueError(f'{what} needs pharmacophores on both '
                                 'Molecule objects (build them with `pharm_multi_vector` set).')
        return (tree_map(convert, mode.inputs(self.ref_molec)),
                tree_map(convert, mode.inputs(self.fit_molec)))

    def align(self,
              mode: Union[str, Mode],
              *,
              num_repeats: int = 50,
              trans_init: bool = False,
              lr: float = 0.1,
              max_num_steps: int = 200,
              backend: str = 'auto',
              verbose: bool = False,
              **params) -> AlignmentResult:
        """
        Align fit_molec to ref_molec under ``mode`` and return the result.

        ``mode`` is a registered name (see ``shepherd_score.modes.MODES``) with its parameters
        passed as keywords, or a ``Mode`` instance::

            res = pair.align('surf_esp', lam=0.3, similarity='tversky')
            res.score, res.transform, res.aligned['points']

        Unlike the ``align_with_*`` methods, this does not write ``transform_*`` /
        ``sim_aligned_*``.

        Parameters
        ----------
        mode : str or Mode
            Alignment mode.
        num_repeats : int, optional
            Number of SE(3) seeds. Default is 50.
        trans_init : bool, optional
            Also seed by translating the fit's center of mass to each reference atom (or
            pharmacophore), with 10 rotations each. Default is ``False``.
        lr : float, optional
            Learning rate. Default is 0.1.
        max_num_steps : int, optional
            Maximum optimization steps. Default is 200.
        backend : str, optional
            ``'torch'`` (generic autograd), ``'analytical'`` or ``'jax'`` where the mode provides
            them, or ``'auto'`` (default): ``'analytical'`` if available, else ``'torch'``.
        verbose : bool, optional
            Print optimization progress. Default is ``False``.
        **params
            Parameters of a named mode, e.g. ``alpha``, ``lam``, ``similarity``. An
            ``avoid_points`` cloud is used as-is, in the frame the molecules are scored in
            (centered when ``do_center=True``; see :attr:`ref_shift`).

        Returns
        -------
        AlignmentResult
            ``score``, ``transform`` (4, 4) and ``aligned`` (the fit's mode inputs under the
            optimal transform).
        """
        mode = get_mode(mode, **params)
        backends = mode.backends()
        if backend == 'auto':
            backend = 'analytical' if 'analytical' in backends else 'torch'
        if backend != 'torch' and backend not in backends:
            raise ValueError(f'Backend {backend!r} is not available for {mode}. '
                             f"Available: {', '.join(['torch', *backends])}.")

        if backend.startswith('jax'):
            jnp = _require_jax()
            ref, fit = self._mode_inputs(mode, jnp.array)
            trans_centers = mode.trans_centers(self.ref_molec) if trans_init else None
        else:
            ref, fit = self._mode_inputs(mode, self._as_tensor)
            trans_centers = self._to_tensor(mode.trans_centers(self.ref_molec)) if trans_init else None

        optimizer = partial(optimize_se3, mode) if backend == 'torch' else backends[backend]
        aligned, se3_transform, score = optimizer(ref, fit, num_repeats=num_repeats,
                                                  trans_centers=trans_centers, lr=lr,
                                                  max_num_steps=max_num_steps, verbose=verbose)
        to_np = np.asarray if backend.startswith('jax') else (lambda x: x.cpu().numpy())
        return AlignmentResult(score=to_np(score), transform=to_np(se3_transform),
                               aligned=tree_map(to_np, aligned))

    def score(self, mode: Union[str, Mode], **params) -> np.ndarray:
        """
        Score fit_molec against ref_molec under ``mode`` in their current poses (PyTorch).

        ``mode`` and ``params`` are as in :meth:`align`, e.g. ``pair.score('vol', similarity='tversky')``.
        """
        mode = get_mode(mode, **params)
        ref, fit = self._mode_inputs(mode, self._as_tensor)
        with torch.no_grad():
            return np.asarray(mode.score(ref, fit).cpu())

    def _align_and_store(self, slot: str, mode: Mode, **opt):
        """Run :meth:`align` for an ``align_with_*`` method and store the result in the method's
        ``transform_*`` / ``sim_aligned_*`` slot. Returns the aligned fit inputs."""
        res = self.align(mode, **opt)
        self._alignments[slot].score = res.score
        self._alignments[slot].transform = res.transform
        return res.aligned

    def align_with_vol(self,
                       no_H: bool = True,
                       num_repeats: int = 50,
                       trans_init: bool = False,
                       lr: float = 0.1,
                       max_num_steps: int = 200,
                       use_jax: bool = False,
                       use_analytical: bool = True,
                       verbose: bool = False,
                       similarity: Similarity = 'tanimoto') -> np.ndarray:
        """
        Align fit_molec to ref_molec using volumetric similarity.

        Optimally aligned score found in ``self.sim_aligned_vol`` and the optimal SE(3)
        transformation is at ``self.transform_vol``. If ``no_H`` is ``True``, append '_noH' to them.

        Parameters
        ----------
        no_H : bool
            Whether to not include hydrogens in volumetric similarity. Default is ``True``.
        num_repeats : int, optional
            Number of different random initializations of SO(3) transformation parameters.
            Default is 50.
        trans_init : bool, optional
            Apply translation initializiation for alignment. ``fit_molec``'s center of mass (COM) is translated to
            each ``ref_molec``'s atoms, with 10 rotations for each translation. So the
            number of initializations scales as (# translation centers * 10 + 5) where 5 is from
            the identity and 4 PCA with aligned COMs. If ``None``, then ``num_repeats``
            rotations are done with aligned COMs.
        lr : float, optional
            Learning rate or step-size for optimization. Default is 0.1.
        max_num_steps : int, optional
            Maximum number of steps to optimize over.
            Default is 200.
        use_jax : bool, optional
            Whether to use Jax instead of PyTorch. Default is ``False``.
        use_analytical : bool, optional
            Whether to use analytical gradients instead of PyTorch autograd. Ignored if
            ``use_jax=True``. Default is ``True``.
        verbose : bool, optional
            Print initial and final similarity scores with scores every 100 steps. Default is ``False``.
        similarity : str or (float, float), optional
            ``'tanimoto'`` (default), ``'tversky'``, ``'tversky_ref'``, ``'tversky_fit'`` or
            Tversky weights ``(a, b)``. A Tversky similarity stores the result in
            ``self.sim_aligned_vol_tversky`` / ``self.transform_vol_tversky`` and
            runs the generic PyTorch optimizer.

        Returns
        -------
        aligned_fit_points : np.ndarray
            Coordinates of transformed atoms. Shape: (N, 3).
        """
        slot = _similarity_slot('vol', similarity, 'vol_noH' if no_H else 'vol')
        return self._align_and_store(slot, Vol(no_H=no_H, similarity=similarity),
                                     num_repeats=num_repeats, trans_init=trans_init, lr=lr,
                                     max_num_steps=max_num_steps, verbose=verbose,
                                     backend=_legacy_backend(use_jax, use_analytical))['points']


    def align_with_vol_avoid(self,
                             avoid_points: np.ndarray,
                             avoid_weight: float = 1.0,
                             avoid_min_dist: float = 2.0,
                             no_H: bool = True,
                             num_repeats: int = 50,
                             trans_init: bool = False,
                             lr: float = 0.1,
                             max_num_steps: int = 200,
                             verbose: bool = False,
                             use_analytical: bool = True) -> np.ndarray:
        """Align fit_molec to ref_molec by volumetric similarity minus a linear hard-sphere
        excluded-volume penalty that keeps the fit out of a fixed cloud of ``avoid_points``:

            score = shape_Tanimoto - avoid_weight * sum relu((avoid_min_dist - d)/avoid_min_dist)

        ``avoid_points`` is an arbitrary cloud relative to the reference (a pocket wall, a region
        to grow away from), not one of the two molecules. It is used as-is, in the frame the
        molecules are scored in: with ``do_center=True`` that is the centered reference frame,
        so pass ``cloud - self.ref_shift`` for a cloud in the original coordinates. The penalty is evaluated on the fit shape
        atoms. Optimally aligned score found in ``self.sim_aligned_vol_avoid`` and the SE(3)
        transformation in ``self.transform_vol_avoid``.

        Parameters
        ----------
        avoid_points : np.ndarray (K,3)
            Points to penalize the fit for coming within ``avoid_min_dist`` of.
        avoid_weight : float, optional
            Weight of the subtracted penalty term. Default 1.0.
        avoid_min_dist : float, optional
            Hard-sphere cutoff/scale (Angstrom): the penalty ramps linearly from 1 at coincidence to
            0 at this distance, and is 0 beyond. Default 2.0.
        no_H : bool, optional
            Score the shape channel without hydrogens. Default ``True``.
        num_repeats, trans_init, lr, max_num_steps, verbose
            As in :meth:`align_with_vol`.
        use_analytical : bool, optional
            Whether to use analytical gradients instead of PyTorch autograd. Default is ``True``.

        Returns
        -------
        aligned_fit_points : np.ndarray (M,3)
        """
        mode = VolAvoid(no_H=no_H, avoid_points=avoid_points, avoid_weight=avoid_weight,
                        avoid_min_dist=avoid_min_dist)
        return self._align_and_store('vol_avoid', mode, num_repeats=num_repeats,
                                     trans_init=trans_init, lr=lr, max_num_steps=max_num_steps,
                                     verbose=verbose,
                                     backend=_legacy_backend(False, use_analytical))['points']


    def align_with_vol_esp(self,
                           lam: float = 0.1,
                           no_H: bool = True,
                           num_repeats: int = 50,
                           trans_init: bool = False,
                           lr: float = 0.1,
                           max_num_steps: int = 200,
                           use_jax: bool = False,
                           use_analytical: bool = True,
                           verbose: bool = False,
                           similarity: Similarity = 'tanimoto') -> np.ndarray:
        """
        Align fit_molec to ref_molec using volume similarity weighted by partial charge
        Toggle ``no_H`` parameter for scoring with or without hydrogens.

        Typically ``lam=0.1`` is used (the default). ``lam`` is a width, not a weight: the ESP
        term is ``exp(-esp_diff_sq / lam)``, so a smaller lam penalises charge differences more
        sharply. It is used raw here, whereas ``align_with_surf_esp`` scales its lam by
        ``LAM_SCALING`` internally, so the two are not interchangeable.
        Optimally aligned score found in ``self.sim_aligned_vol_esp`` and the optimal SE(3)
        transformation is at ``self.transform_vol_esp``. If ``no_H`` is ``True``, append '_noH' to them.

        Parameters
        ----------
        lam : float
            Partial charge weighting parameter.
        no_H : bool
            Whether to not include hydrogens in volumetric similarity. Default is ``True``.
        num_repeats : int, optional
            Number of different random initializations of SO(3) transformation parameters.
            Default is 50.
        trans_init : bool, optional
            Apply translation initializiation for alignment. ``fit_molec``'s center of mass
            (COM) is translated to each ``ref_molec``'s atoms, with 10 rotations for each translation.
            So the number of initializations scales as (# translation centers * 10 + 5) where 5 is
            from the identity and 4 PCA with aligned COMs. If ``None``, then ``num_repeats``
            rotations are done with aligned COMs. Default is ``False``.
        lr : float, optional
            Learning rate or step-size for optimization. Default is 0.1.
        max_num_steps : int, optional
            Maximum number of steps to optimize over.
            Default is 200.
        use_jax : bool, optional
            Whether to use Jax instead of PyTorch. Default is ``False``.
        verbose : bool, optional
            Print initial and final similarity scores with scores every 100 steps.
            Default is ``False``.
        similarity : str or (float, float), optional
            ``'tanimoto'`` (default), ``'tversky'``, ``'tversky_ref'``, ``'tversky_fit'`` or
            Tversky weights ``(a, b)``. A Tversky similarity stores the result in
            ``self.sim_aligned_vol_esp_tversky`` / ``self.transform_vol_esp_tversky`` and
            runs the generic PyTorch optimizer.

        Returns
        -------
        aligned_fit_points : np.ndarray
            Coordinates of transformed atoms. Shape: (N, 3).
        """
        slot = _similarity_slot('vol_esp', similarity,
                                          'vol_esp_noH' if no_H else 'vol_esp')
        return self._align_and_store(slot, VolESP(lam=lam, no_H=no_H, similarity=similarity),
                                     num_repeats=num_repeats, trans_init=trans_init,
                                     lr=lr, max_num_steps=max_num_steps, verbose=verbose,
                                     backend=_legacy_backend(use_jax, use_analytical))['points']


    def align_with_surf(self,
                        alpha: float,
                        num_repeats: int = 50,
                        trans_init: bool = False,
                        lr: float = 0.1,
                        max_num_steps: int = 200,
                        use_jax: bool = False,
                        use_analytical: bool = True,
                        verbose: bool = False,
                        similarity: Similarity = 'tanimoto') -> np.ndarray:
        """
        Align fit_molec to ref_molec using surface similarity.

        Optimally aligned score found in ``self.sim_aligned_surf`` and the optimal SE(3)
        transformation is at ``self.transform_surf``.

        Parameters
        ----------
        alpha : float
            Gaussian width parameter for overlap.
        num_repeats : int, optional
            Number of different random initializations of SO(3) transformation parameters.
            Default is 50.
        trans_init : bool, optional
            Apply translation initializiation for alignment. ``fit_molec``'s center of mass
            (COM) is translated to each ``ref_molec``'s atoms, with 10 rotations for each
            translation. So the number of initializations scales as
            (# translation centers * 10 + 5) where 5 is from the identity and 4 PCA with
            aligned COMs. If ``None``, then ``num_repeats`` rotations are done with aligned COMs.
            Default is ``False``.
        lr : float, optional
            Learning rate or step-size for optimization. Default is 0.1.
        max_num_steps : int, optional
            Maximum number of steps to optimize over.
            Default is 200.
        use_jax : bool, optional
            Whether to use Jax instead of PyTorch. Default is ``False``.
        use_analytical : bool, optional
            Whether to use analytical gradients instead of PyTorch autograd. Ignored if
            ``use_jax=True``. Default is ``True``.
        verbose : bool, optional
            Print initial and final similarity scores with scores every 100 steps. Default is ``False``.
        similarity : str or (float, float), optional
            ``'tanimoto'`` (default), ``'tversky'``, ``'tversky_ref'``, ``'tversky_fit'`` or
            Tversky weights ``(a, b)``. A Tversky similarity stores the result in
            ``self.sim_aligned_surf_tversky`` / ``self.transform_surf_tversky`` and
            runs the generic PyTorch optimizer.

        Returns
        -------
        aligned_fit_points : np.ndarray
            Coordinates of transformed atoms. Shape: (N, 3).
        """
        slot = _similarity_slot('surf', similarity)
        return self._align_and_store(slot, Surf(alpha=alpha, similarity=similarity),
                                     num_repeats=num_repeats, trans_init=trans_init, lr=lr,
                                     max_num_steps=max_num_steps, verbose=verbose,
                                     backend=_legacy_backend(use_jax, use_analytical))['points']


    def align_with_surf_esp(self,
                       alpha: float,
                       lam: float = 0.3,
                       num_repeats: int = 50,
                       trans_init: bool = False,
                       lr: float = 0.1,
                       max_num_steps: int = 200,
                       use_jax: bool = False,
                       use_analytical: bool = True,
                       verbose: bool = False,
                       similarity: Similarity = 'tanimoto') -> np.ndarray:
        """
        Align fit_molec to ref_molec using surface-ESP similarity (formerly ``align_with_esp``,
        which is kept as an alias).
        ``lam`` is scaled by ``(1e4/(4*55.263*np.pi))**2`` for correct units.

        Typically, ``lam=0.3`` is used and is scaled internally.

        Optimally aligned score found in ``self.sim_aligned_surf_esp`` and the optimal SE(3)
        transformation is at ``self.transform_surf_esp``.

        Parameters
        ----------
        alpha : float
            Gaussian width parameter for overlap.
        lam : float, optional
            Weighting factor for ESP scoring. Scaled internally. Default is 0.3.
        num_repeats : int, optional
            Number of different random initializations of SO(3) transformation parameters.
            Default is 50.
        trans_init : bool, optional
            Apply translation initializiation for alignment. ``fit_molec``'s COM is translated to
            each ``ref_molecs``'s atoms, with 10 rotations for each translation. So the
            number of initializations scales as (# translation centers * 10 + 5) where 5 is from
            the identity and 4 PCA with aligned COM's. If None, then num_repeats rotations are done
            with aligned COM's. Default is ``False``.
        lr : float, optional
            Learning rate or step-size for optimization. Default is 0.1.
        max_num_steps : int, optional
            Maximum number of steps to optimize over.
            Default is 200.
        use_jax : bool, optional
            Whether to use Jax instead of PyTorch. Default is ``False``.
        verbose : bool, optional
            Print initial and final similarity scores with scores every 100 steps.
            Default is ``False``.
        similarity : str or (float, float), optional
            ``'tanimoto'`` (default), ``'tversky'``, ``'tversky_ref'``, ``'tversky_fit'`` or
            Tversky weights ``(a, b)``. A Tversky similarity stores the result in
            ``self.sim_aligned_surf_esp_tversky`` / ``self.transform_surf_esp_tversky`` and
            runs the generic PyTorch optimizer.

        Returns
        -------
        aligned_fit_points : np.ndarray
            Coordinates of transformed atoms. Shape: (N, 3).
        """
        slot = _similarity_slot('surf_esp', similarity)
        return self._align_and_store(slot, SurfESP(alpha=alpha, lam=lam, similarity=similarity),
                                     num_repeats=num_repeats, trans_init=trans_init, lr=lr,
                                     max_num_steps=max_num_steps, verbose=verbose,
                                     backend=_legacy_backend(use_jax, use_analytical))['points']


    def align_with_vol_and_surf_esp(self,
                             alpha: float,
                             lam: float = 0.001,
                             probe_radius: float = 1.0,
                             esp_weight: float = 0.5,
                             num_repeats: int = 50,
                             trans_init: bool = False,
                             lr: float = 0.1,
                             max_num_steps: int = 200,
                             use_jax: bool = False,
                             verbose: bool = False,
                             similarity: Similarity = 'tanimoto'):
        """
        Align using ShaEP similarity score (formerly ``align_with_esp_combo``, which is kept as
        an alias).
        If alpha is 0.81, then it automatically uses volumetric shape similarity.
        Otherwise, it uses surface shape similarity.

        Optimally aligned score found in ``self.sim_aligned_vol_and_surf_esp`` and the optimal
        SE(3) transformation is at ``self.transform_vol_and_surf_esp``.

        Parameters
        ----------
        alpha : float
            Gaussian width parameter for overlap.
        lam : float, optional
            ESP weighting parameter. Default is 0.001.
        probe_radius : float, optional
            Surface points found within vdW radii + probe radius will be masked out.
            Surface generation uses a probe radius of 1.2 by default (radius of hydrogen)
            so we use a slightly lower radius for be more tolerant. Default is 1.0.
        esp_weight : float, optional
            How much to weight shape vs esp_combo similarity ([0,1]). Default is 0.5.
        num_repeats : int, optional
            Number of different random initializations of SO(3) transformation parameters.
            Default is 50.
        trans_init : bool, optional
            Apply translation initializiation for alignment. ``fit_molec``'s COM is translated
            to each ``ref_molecs``'s atoms, with 10 rotations for each translation. So the
            number of initializations scales as (# translation centers * 10 + 5) where 5 is
            from the identity and 4 PCA with aligned COM's. If ``None``, then ``num_repeats``
            rotations are done with aligned COM's. Default is ``False``.
        lr : float, optional
            Learning rate or step-size for optimization. Default is 0.1.
        max_num_steps : int, optional
            Maximum number of steps to optimize over.
            Default is 200.
        use_jax : bool, optional
            Whether to use Jax instead of PyTorch. Default is ``False``.
        verbose : bool, optional
            Print initial and final similarity scores with scores every 100 steps.
            Default is ``False``.
        similarity : str or (float, float), optional
            ``'tanimoto'`` (default), ``'tversky'``, ``'tversky_ref'``, ``'tversky_fit'`` or
            Tversky weights ``(a, b)``. A Tversky similarity stores the result in
            ``self.sim_aligned_vol_and_surf_esp_tversky`` /
            ``self.transform_vol_and_surf_esp_tversky`` and runs the generic PyTorch optimizer.
            Tversky applies to the shape term; the ESP agreement is unchanged.

        Returns
        -------
        aligned_fit_points : np.ndarray (N, 3)
            Coordinates of transformed atoms. Shape: (N, 3).
        """
        mode = VolAndSurfESP(alpha=alpha, lam=lam, probe_radius=probe_radius, esp_weight=esp_weight,
                             similarity=similarity)
        slot = _similarity_slot('vol_and_surf_esp', similarity)
        return self._align_and_store(slot, mode, num_repeats=num_repeats,
                                     trans_init=trans_init, lr=lr, max_num_steps=max_num_steps,
                                     verbose=verbose, backend=_legacy_backend(use_jax))['points']


    # legacy method aliases (esp -> surf_esp, esp_combo -> vol_and_surf_esp)
    align_with_esp = align_with_surf_esp
    align_with_esp_combo = align_with_vol_and_surf_esp


    def align_with_vol_color(self,
                             color_weight: float = 0.5,
                             alpha: float = 0.81,
                             similarity: Similarity = 'tanimoto',
                             directionless: bool = True,
                             extended_points: bool = False,
                             only_extended: bool = False,
                             num_repeats: int = 50,
                             trans_init: bool = False,
                             lr: float = 0.1,
                             max_num_steps: int = 200,
                             verbose: bool = False,
                             color_similarity: Optional[Similarity] = None) -> np.ndarray:
        """
        Align using a ROCS-style combined atom-centered Gaussian *shape* (volume) and
        directionless *color* (pharmacophore) overlay (a TanimotoCombo analogue).

        The optimized objective is
        ``(1 - color_weight) * shape_Tanimoto + color_weight * color_Tanimoto``, or the Tversky
        form of both under ``similarity``. By default the color channel is directionless
        (isotropic point Gaussians); pass ``directionless=False`` to keep the orientation-vector
        weighting. For ROCS feature parity, build the ``Molecule`` objects with
        ``feature_set='rdkit_base'``.

        Optimally aligned score is stored in ``self.sim_aligned_vol_color`` and the optimal
        SE(3) transformation in ``self.transform_vol_color``.

        Parameters
        ----------
        color_weight : float, optional
            Weight of the color channel in [0, 1]; shape gets ``1 - color_weight``.
            Default is 0.5.
        alpha : float, optional
            Gaussian width for the shape overlap. Default is 0.81 (volumetric, heavy atoms).
        similarity : str or (float, float), optional
            Similarity for both channels: ``'tanimoto'`` (default), ``'tversky'``,
            ``'tversky_ref'``, ``'tversky_fit'`` or Tversky weights ``(a, b)``. A Tversky
            similarity stores the result in ``self.sim_aligned_vol_color_tversky`` /
            ``self.transform_vol_color_tversky``.
        directionless : bool, optional
            ``True`` (default) scores color as isotropic point Gaussians; ``False`` uses the
            orientation-vector cosine weighting. Same polarity as the extraction-side
            ``directionless`` on :meth:`Molecule.get_pharmacophore`.
        extended_points, only_extended : bool, optional
            Forwarded to the color scorer (ignored when ``directionless=True``).
        num_repeats : int, optional
            Number of SE(3) initializations. Default is 50.
        trans_init : bool, optional
            Translation-seeded initialization from the reference atoms. Default is ``False``.
        lr : float, optional
            Learning rate. Default is 0.1.
        max_num_steps : int, optional
            Maximum optimization steps. Default is 200.
        verbose : bool, optional
            Print progress. Default is ``False``.
        color_similarity : str or (float, float), optional
            Similarity for the color channel alone. Default (``None``) is ``similarity``.

        Returns
        -------
        aligned_fit_centers : np.ndarray (M, 3)
            Transformed fit atom (heavy-atom) coordinates.
        """
        mode = vol_color(color_weight=color_weight, alpha=alpha, similarity=similarity,
                         color_similarity=color_similarity, directionless=directionless,
                         extended_points=extended_points, only_extended=only_extended)
        slot = _similarity_slot('vol_color', similarity)
        return self._align_and_store(slot, mode, num_repeats=num_repeats,
                                     trans_init=trans_init, lr=lr, max_num_steps=max_num_steps,
                                     verbose=verbose)[0]['points']


    def align_with_vol_color_avoid(self,
                                   avoid_points: np.ndarray,
                                   color_weight: float = 0.5,
                                   avoid_weight: float = 1.0,
                                   avoid_min_dist: float = 2.0,
                                   alpha: float = 0.81,
                                   similarity: Similarity = 'tanimoto',
                                   directionless: bool = True,
                                   extended_points: bool = False,
                                   only_extended: bool = False,
                                   no_H: bool = True,
                                   num_repeats: int = 50,
                                   trans_init: bool = False,
                                   lr: float = 0.1,
                                   max_num_steps: int = 200,
                                   verbose: bool = False,
                                   color_similarity: Optional[Similarity] = None) -> np.ndarray:
        """Align by shape + color (as :meth:`align_with_vol_color`) minus the excluded-volume
        penalty of :meth:`align_with_vol_avoid`:

            score = (1 - color_weight) * shape + color_weight * color
                    - avoid_weight * sum relu((avoid_min_dist - d) / avoid_min_dist)

        The penalty is evaluated between the fit's shape atoms and ``avoid_points``, a fixed
        (K, 3) cloud relative to the reference (see :meth:`align_with_vol_avoid`). Runs on the
        generic PyTorch optimizer. Score and
        transform are stored in ``self.sim_aligned_vol_color_avoid`` /
        ``self.transform_vol_color_avoid``, or in ``self.sim_aligned_vol_color_avoid_tversky`` /
        ``self.transform_vol_color_avoid_tversky`` for a Tversky ``similarity``.

        Parameters
        ----------
        avoid_points : np.ndarray (K,3)
            Points to penalize the fit for coming within ``avoid_min_dist`` of.
        color_weight, alpha, similarity, directionless, extended_points, only_extended, color_similarity
            As in :meth:`align_with_vol_color`. ``similarity`` does not affect the penalty.
        avoid_weight, avoid_min_dist
            As in :meth:`align_with_vol_avoid`.
        no_H : bool, optional
            Score shape and penalty without hydrogens. Default ``True``.
        num_repeats, trans_init, lr, max_num_steps, verbose
            As in :meth:`align_with_vol`.

        Returns
        -------
        aligned_fit_points : np.ndarray (M,3)
            Transformed fit shape-atom coordinates.
        """
        mode = vol_color_avoid(avoid_points=avoid_points, color_weight=color_weight,
                               avoid_weight=avoid_weight, avoid_min_dist=avoid_min_dist,
                               alpha=alpha, similarity=similarity, directionless=directionless,
                               extended_points=extended_points, only_extended=only_extended,
                               color_similarity=color_similarity, no_H=no_H)
        slot = _similarity_slot('vol_color_avoid', similarity)
        return self._align_and_store(slot, mode, num_repeats=num_repeats,
                                     trans_init=trans_init, lr=lr, max_num_steps=max_num_steps,
                                     verbose=verbose)[0]['points']


    def align_with_vol_lipo(self,
                            lipo_weight: float = 0.5,
                            alpha: float = 0.81,
                            lam: float = 0.1,
                            num_repeats: int = 50,
                            trans_init: bool = False,
                            lr: float = 0.1,
                            max_num_steps: int = 200,
                            verbose: bool = False,
                            similarity: Similarity = 'tanimoto') -> np.ndarray:
        """
        Align using a combined atom-centered Gaussian *shape* (volume) + *lipophilicity* overlay.

        The optimized objective is
        ``(1 - lipo_weight) * shape_Tanimoto + lipo_weight * lipo_Tanimoto`` (or the Tversky form
        of both under ``similarity``). The shape channel is
        the heavy-atom Gaussian volume overlap of the ``vol`` mode; the lipophilicity channel
        overlays the per-atom Crippen logP contributions at the heavy-atom centers like a
        partial-charge field (``get_overlap_esp`` with logP as the charge), so hydrophobic
        overlaps hydrophobic. Both fit point sets move under the same SE(3) pose.

        Optimally aligned score is stored in ``self.sim_aligned_vol_lipo`` and the optimal SE(3)
        transformation in ``self.transform_vol_lipo``.

        Parameters
        ----------
        lipo_weight : float, optional
            Weight of the lipophilicity channel in [0, 1]; shape gets ``1 - lipo_weight``.
            Default is 0.5.
        alpha : float, optional
            Gaussian width for the shape and lipophilicity overlaps. Default is 0.81 (volumetric,
            heavy atoms).
        lam : float, optional
            Width of the value-matching kernel in the lipophilicity overlap, used raw as in
            ``align_with_vol_esp``. Default is 0.1.
        num_repeats : int, optional
            Number of SE(3) initializations. Default is 50.
        trans_init : bool, optional
            Translation-seeded initialization from the reference atoms. Default is ``False``.
        lr : float, optional
            Learning rate. Default is 0.1.
        max_num_steps : int, optional
            Maximum optimization steps. Default is 200.
        verbose : bool, optional
            Print progress. Default is ``False``.
        similarity : str or (float, float), optional
            ``'tanimoto'`` (default), ``'tversky'``, ``'tversky_ref'``, ``'tversky_fit'`` or
            Tversky weights ``(a, b)``. A Tversky similarity stores the result in
            ``self.sim_aligned_vol_lipo_tversky`` / ``self.transform_vol_lipo_tversky`` and
            runs the generic PyTorch optimizer.
            Applies to both channels.

        Returns
        -------
        aligned_fit_centers : np.ndarray (M, 3)
            Transformed fit atom (heavy-atom, shape-channel) coordinates.
        """
        mode = vol_lipo(lipo_weight=lipo_weight, alpha=alpha, lam=lam, similarity=similarity)
        slot = _similarity_slot('vol_lipo', similarity)
        return self._align_and_store(slot, mode, num_repeats=num_repeats,
                                     trans_init=trans_init, lr=lr, max_num_steps=max_num_steps,
                                     verbose=verbose)[0]['points']


    def align_with_pharm(self,
                         similarity: _SIM_TYPE = 'tanimoto',
                         extended_points: bool = False,
                         only_extended: bool = False,
                         num_repeats: int = 50,
                         trans_init: bool = False,
                         lr: float = 0.1,
                         max_num_steps: int = 200,
                         use_jax: bool = False,
                         verbose: bool = False,
                         use_vectorized: bool = True,
                         use_analytical: bool = True,
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align fit_molec to ref_molec using pharmacophore similarity.

        Optimally aligned score found in ``self.sim_aligned_pharm`` and the optimal SE(3)
        transformation is at ``self.transform_pharm``.

        Parameters
        ----------
        similarity : str from ('tanimoto', 'tversky', 'tversky_ref', 'tversky_fit')
            Specifies what similarity function to use. Options are:
            'tanimoto' -- symmetric scoring function
            'tversky' -- asymmetric -> Uses OpenEye's formulation 95% normalization by molec 1
            'tversky_ref' -- asymmetric -> Uses Pharao's formulation 100% normalization by molec 1.
            'tversky_fit' -- asymmetric -> Uses Pharao's formulation 100% normalization by molec 2.
        extended_points : bool, optional
            Whether to score HBA/HBD with gaussian overlaps of extended points. Default is ``False``.
        only_extended : bool, optional
            When ``extended_points`` is ``True``, decide whether to only score the extended points
            (ignore anchor overlaps). Default is ``False``.
        num_repeats : int, optional
            Number of different random initializations of SO(3) transformation parameters.
            Default is 50.
        trans_init : bool, optional
            Apply translation initializiation for alignment. ``fit_molec``'s COM is translated to
            each ``ref_molecs``'s pharmacophore, with 10 rotations for each translation. So the
            number of initializations scales as (# translation centers * 10 + 5) where 5 is from
            the identity and 4 PCA with aligned COM's. If ``None``, then ``num_repeats`` rotations
            are done with aligned COM's. Default is ``False``.
        lr : float, optional
            Learning rate or step-size for optimization. Default is 0.1.
        max_num_steps : int, optional
            Maximum number of steps to optimize over.
            Default is 200.
        use_jax : bool, optional
            Whether to use Jax instead of PyTorch. Default is ``False``.
        verbose : bool, optional
            Print initial and final similarity scores with scores every 100 steps.
            Default is ``False``.
        use_vectorized : bool, optional
            Whether to use the vectorized version of the pharmacophore scoring function.
            This is only relevant if ``use_jax=True``.
            Default is ``True``.
        use_analytical : bool, optional
            Whether to use the analytical version of the pharmacophore scoring function.
            Currently only implemented for PyTorch.
            Default is ``True``.
        Returns
        -------
        tuple
            aligned_fit_anchors : np.ndarray
                Aligned coordinates of pharmacophore positions. Shape: (P, 3).
            aligned_fit_vectors : np.ndarray
                Aligned coordinates of pharmacophore vectors. Shape: (P, 3).
        """
        if use_jax:
            backend = 'jax' if use_vectorized else 'jax_unvectorized'
        else:
            backend = _legacy_backend(False, use_analytical)
        mode = Pharm(similarity=similarity, extended_points=extended_points, only_extended=only_extended)
        aligned = self._align_and_store('pharm', mode, num_repeats=num_repeats, trans_init=trans_init,
                                        lr=lr, max_num_steps=max_num_steps, verbose=verbose,
                                        backend=backend)
        return aligned['anchors'], aligned['vectors']


    def align_with_vol_pharm(self,
                             color_weight: float = 0.5,
                             alpha: float = 0.81,
                             similarity: Similarity = 'tanimoto',
                             extended_points: bool = False,
                             only_extended: bool = False,
                             num_repeats: int = 50,
                             trans_init: bool = False,
                             lr: float = 0.1,
                             max_num_steps: int = 200,
                             verbose: bool = False,
                             color_similarity: Optional[Similarity] = None) -> np.ndarray:
        """Align using a combined atom-centered Gaussian *shape* (volume) and *directional
        pharmacophore* overlay. This is equivalent to the ``vol_color`` combo with the
        orientation-vector weighting kept (``directionless=False``).

        Score and transform are stored in ``self.sim_aligned_vol_pharm`` /
        ``self.transform_vol_pharm``.

        Parameters
        ----------
        color_weight : float, optional
            Weight of the color channel. Default is 0.5.
        alpha : float, optional
            Alpha parameter for the Gaussian overlap. Default is 0.81.
        similarity : Similarity, optional
            Similarity function to use for the shape channel. Default is 'tanimoto'.
        extended_points : bool, optional
            Whether to score HBA/HBD with gaussian overlaps of extended points. Default is ``False``.
        only_extended : bool, optional
            When ``extended_points`` is ``True``, decide whether to only score the extended points
            (ignore anchor overlaps). Default is ``False``.
        num_repeats : int, optional
            Number of different random initializations of SO(3) transformation parameters.
            Default is 50.
        trans_init : bool, optional
            Apply translation initializiation for alignment. ``fit_molec``'s COM is translated to
            each ``ref_molecs``'s pharmacophore, with 10 rotations for each translation. So the
            number of initializations scales as (# translation centers * 10 + 5) where 5 is from
            the identity and 4 PCA with aligned COM's. If ``None``, then ``num_repeats`` rotations
            are done with aligned COM's. Default is ``False``.
        lr : float, optional
            Learning rate or step-size for optimization. Default is 0.1.
        max_num_steps : int, optional
            Maximum number of steps to optimize over.
            Default is 200.
        verbose : bool, optional
            Print initial and final similarity scores with scores every 100 steps.
            Default is ``False``.
        color_similarity : Similarity, optional
            Similarity function to use for the color channel. Default is ``similarity``.

        Returns
        -------
        aligned_fit_points : np.ndarray
            Aligned coordinates of pharmacophore positions. Shape: (P, 3).
        """
        mode = vol_color(color_weight=color_weight, alpha=alpha, similarity=similarity,
                         color_similarity=color_similarity, directionless=False,
                         extended_points=extended_points, only_extended=only_extended)
        return self._align_and_store('vol_pharm', mode, num_repeats=num_repeats, trans_init=trans_init,
                                     lr=lr, max_num_steps=max_num_steps, verbose=verbose)[0]['points']


    def align_with_vol_atomtype(self,
                                atomtype_weight: float = 0.5,
                                alpha: float = 0.81,
                                similarity: Similarity = 'tanimoto',
                                num_repeats: int = 50,
                                trans_init: bool = False,
                                lr: float = 0.1,
                                max_num_steps: int = 200,
                                verbose: bool = False,
                                type_similarity: Optional[Similarity] = None) -> np.ndarray:
        """Align using a combined atom-centered Gaussian *shape* (volume) and *atom-identity*
        overlay, ``(1 - atomtype_weight) * shape + atomtype_weight * identity``. The identity
        channel is a Gaussian overlap in which only same-element atoms contribute. Score and
        transform are stored in ``self.sim_aligned_vol_atomtype`` /
        ``self.transform_vol_atomtype``. ``similarity`` applies to both channels;
        ``type_similarity`` (default ``similarity``) overrides it for the identity channel, which
        takes only the named similarities."""
        mode = vol_atomtype(atomtype_weight=atomtype_weight, alpha=alpha, similarity=similarity,
                            type_similarity=type_similarity)
        return self._align_and_store('vol_atomtype', mode, num_repeats=num_repeats,
                                     trans_init=trans_init, lr=lr, max_num_steps=max_num_steps,
                                     verbose=verbose)[0]['points']


    def align_with_vol_mr(self,
                          mr_weight: float = 0.5,
                          alpha: float = 0.81,
                          lam: float = 0.1,
                          num_repeats: int = 50,
                          trans_init: bool = False,
                          lr: float = 0.1,
                          max_num_steps: int = 200,
                          verbose: bool = False) -> np.ndarray:
        """Align using a combined atom-centered Gaussian *shape* (volume) and *molar-refractivity*
        overlay, ``(1 - mr_weight) * shape + mr_weight * mr``: ``vol_lipo`` with the per-atom
        Crippen MR contribution in place of logP. Score and transform are stored in
        ``self.sim_aligned_vol_mr`` / ``self.transform_vol_mr``."""
        return self._align_and_store('vol_mr', vol_mr(mr_weight=mr_weight, alpha=alpha, lam=lam),
                                     num_repeats=num_repeats, trans_init=trans_init, lr=lr,
                                     max_num_steps=max_num_steps, verbose=verbose)[0]['points']


    def align_with_vol_fukui(self,
                             fukui_weight: float = 0.5,
                             alpha: float = 0.81,
                             lam: float = 0.1,
                             num_repeats: int = 50,
                             trans_init: bool = False,
                             lr: float = 0.1,
                             max_num_steps: int = 200,
                             verbose: bool = False) -> np.ndarray:
        """Align using a combined atom-centered Gaussian *shape* (volume) and *Fukui-reactivity*
        overlay, ``(1 - fukui_weight) * shape + fukui_weight * fukui``: ``vol_lipo`` with the
        per-atom Fukui dual descriptor ``f+ - f-`` (``Molecule.fukui``) in place of logP, so
        nucleophilic sites overlap nucleophilic and electrophilic overlap electrophilic. Score and
        transform are stored in ``self.sim_aligned_vol_fukui`` / ``self.transform_vol_fukui``."""
        return self._align_and_store('vol_fukui', vol_fukui(fukui_weight=fukui_weight, alpha=alpha, lam=lam),
                                     num_repeats=num_repeats, trans_init=trans_init, lr=lr,
                                     max_num_steps=max_num_steps, verbose=verbose)[0]['points']


    def score_with_surf(self,
                        alpha: float,
                        use: str = 'np'
                        ) -> np.ndarray:
        """
        Score fit_molec to ref_molec using surface similarity given current alignment.
        By default it uses the numpy implementation.

        Parameters
        ----------
        alpha : float
            Gaussian width parameter for overlap.
        use : str, optional
            Specifies what implementation to use. Options are:
            - 'np' or 'numpy' (numpy implementation)
            - 'jax' or 'jnp' (Jax implementation)
            - 'torch' or 'pytorch' (PyTorch implementation)
            Default is 'np'.

        Returns
        -------
        score : np.ndarray
            Similarity score. Shape: (1,).
        """
        use = use.lower()
        accepted_keys = ('jax', 'jnp', 'torch', 'pytorch', 'np', 'numpy')
        if use not in accepted_keys:
            raise ValueError(f"`use` must be in {accepted_keys}. Instead {use} was passed.")
        if self.num_surf_points is None:
            raise ValueError('The Molecule objects were initialized with no surface points so this method cannot be used.')
        if use == 'jax' or use == 'jnp': # Use Jax optimization implementation
            jnp = _require_jax()
            from shepherd_score.score.gaussian_overlap_jax import get_overlap_jax
            score = get_overlap_jax(
                centers_1=jnp.array(self.ref_molec.surf_pos),
                centers_2=jnp.array(self.fit_molec.surf_pos),
                alpha=alpha,
            )
            return np.array(score)
        elif use == 'torch' or use == 'pytorch':
            # Torch
            score = get_overlap(
                centers_1=self._to_tensor(self.ref_molec.surf_pos),
                centers_2=self._to_tensor(self.fit_molec.surf_pos),
                alpha=alpha,
            )
            return score.cpu().numpy()
        elif use == 'np' or use == 'numpy':
            score = get_overlap_np(
                centers_1=self.ref_molec.surf_pos,
                centers_2=self.fit_molec.surf_pos,
                alpha=alpha,
            )
            return score


    def score_with_esp(self,
                       alpha: float,
                       lam: float = 0.3,
                       use: str = 'np'
                       ) -> np.ndarray:
        """
        Score fit_molec to ref_molec using ESP+surface similarity given current alignment.
        ``lam`` is scaled by ``(1e4/(4*55.263*np.pi))**2`` for correct units.

        Typically ``lam = 0.3`` is used and is scaled internally.
        By default it uses the numpy implementation.

        Parameters
        ----------
        alpha : float
            Gaussian width parameter for overlap.
        lam : float, optional
            Weighting factor for ESP scoring. Default is 0.3.
        use : str, optional
            Specifies what implementation to use. Options are:
            - 'np' or 'numpy' (numpy implementation)
            - 'jax' or 'jnp' (Jax implementation)
            - 'torch' or 'pytorch' (PyTorch implementation)
            Default is 'np'.

        Returns
        -------
        score : np.ndarray
            Similarity score. Shape: (1,).
        """
        lam_scaled = LAM_SCALING * lam
        use = use.lower()
        accepted_keys = ('jax', 'jnp', 'torch', 'pytorch', 'np', 'numpy')
        if use not in accepted_keys:
            raise ValueError(f"`use` must be in {accepted_keys}. Instead {use} was passed.")
        if self.num_surf_points is None:
            raise ValueError('The Molecule objects were initialized with no surface points so this method cannot be used.')
        if use in ('jax', 'jnp'): # Use Jax implementation
            jnp = _require_jax()
            from shepherd_score.score.electrostatic_scoring_jax import get_overlap_esp_jax
            score = get_overlap_esp_jax(
                centers_1=jnp.array(self.ref_molec.surf_pos),
                centers_2=jnp.array(self.fit_molec.surf_pos),
                charges_1=jnp.array(self.ref_molec.surf_esp),
                charges_2=jnp.array(self.fit_molec.surf_esp),
                alpha=alpha,
                lam=lam_scaled,
            )
            return np.array(score)
        elif use in ('torch', 'pytorch'): # Use Torch implementation
            score = get_overlap_esp(
                centers_1=self._to_tensor(self.ref_molec.surf_pos),
                centers_2=self._to_tensor(self.fit_molec.surf_pos),
                charges_1=self._to_tensor(self.ref_molec.surf_esp),
                charges_2=self._to_tensor(self.fit_molec.surf_esp),
                alpha=alpha,
                lam=lam_scaled,
            )
            return score.cpu().numpy()
        elif use in ('np', 'numpy'):
            score = get_overlap_esp_np(
                centers_1=self.ref_molec.surf_pos,
                centers_2=self.fit_molec.surf_pos,
                charges_1=self.ref_molec.surf_esp,
                charges_2=self.fit_molec.surf_esp,
                alpha=alpha,
                lam=lam_scaled,
            )
            return score


    def score_with_pharm(self,
                         similarity: _SIM_TYPE = 'tanimoto',
                         extended_points: bool = False,
                         only_extended: bool = False,
                         use: str = 'np'
                         ) -> np.ndarray:
        """
        Score fit_molec to ref_molec using pharmacophore similarity given current alignment.
        By default it uses the numpy implementation.

        Parameters
        ----------
        similarity : str from ('tanimoto', 'tversky', 'tversky_ref', 'tversky_fit')
            Specifies what similarity function to use. Options are:
            'tanimoto' -- symmetric scoring function
            'tversky' -- asymmetric -> Uses OpenEye's formulation 95% normalization by molec 1
            'tversky_ref' -- asymmetric -> Uses Pharao's formulation 100% normalization by molec 1.
            'tversky_fit' -- asymmetric -> Uses Pharao's formulation 100% normalization by molec 2.
        extended_points : bool, optional
            Whether to score HBA/HBD with gaussian overlaps of extended points.
            Default is ``False``.
        only_extended : bool, optional
            When ``extended_points`` is ``True``, decide whether to only score the extended
            points (ignore anchor overlaps). Default is ``False``.
        use : str, optional
            Specifies what implementation to use. Options are:
            - 'np' or 'numpy' (numpy implementation)
            - 'jax' or 'jnp' (Jax implementation)
            - 'torch' or 'pytorch' (PyTorch implementation)
            Default is 'np'.

        Returns
        -------
        score : np.ndarray
            Similarity score. Shape: (1,).
        """
        use = use.lower()
        accepted_keys = ('jax', 'jnp', 'torch', 'pytorch', 'np', 'numpy')
        if use not in accepted_keys:
            raise ValueError(f"`use` must be in {accepted_keys}. Instead {use} was passed.")
        elif use in ('torch', 'pytorch'):
            # PyTorch
            score = get_overlap_pharm(
                ptype_1=self._to_tensor(self.ref_molec.pharm_types),
                ptype_2=self._to_tensor(self.fit_molec.pharm_types),
                anchors_1=self._to_tensor(self.ref_molec.pharm_ancs),
                anchors_2=self._to_tensor(self.fit_molec.pharm_ancs),
                vectors_1=self._to_tensor(self.ref_molec.pharm_vecs),
                vectors_2=self._to_tensor(self.fit_molec.pharm_vecs),
                similarity=similarity,
                extended_points=extended_points,
                only_extended=only_extended
            )
            return score.cpu().numpy()
        elif use in ('np', 'numpy'):
            score = get_overlap_pharm_np(
                ptype_1=self.ref_molec.pharm_types,
                ptype_2=self.fit_molec.pharm_types,
                anchors_1=self.ref_molec.pharm_ancs,
                anchors_2=self.fit_molec.pharm_ancs,
                vectors_1=self.ref_molec.pharm_vecs,
                vectors_2=self.fit_molec.pharm_vecs,
                similarity=similarity,
                extended_points=extended_points,
                only_extended=only_extended
            )
            return score
        elif use in ('jax', 'jnp'):
            jnp = _require_jax()
            from shepherd_score.score.pharmacophore_scoring_jax import get_overlap_pharm_jax

            score = get_overlap_pharm_jax(
                ptype_1=jnp.array(self.ref_molec.pharm_types),
                ptype_2=jnp.array(self.fit_molec.pharm_types),
                anchors_1=jnp.array(self.ref_molec.pharm_ancs),
                anchors_2=jnp.array(self.fit_molec.pharm_ancs),
                vectors_1=jnp.array(self.ref_molec.pharm_vecs),
                vectors_2=jnp.array(self.fit_molec.pharm_vecs),
                similarity=similarity,
                extended_points=extended_points,
                only_extended=only_extended
            )
            return np.array(score)


    def get_transformed_mol_and_feats(self,
                                      se3_transform: np.ndarray
                                      ) -> Tuple:
        """
        Get an RDKit mol object and applicable features with a transformation applied.

        Parameters
        ----------
        se3_transform : np.ndarray
            SE(3) transformation matrix. Shape: (4,4).

        Returns
        -------
        tuple
            transformed_mol : rdkit.Chem.Mol
                Molecule with transformed coordinates.
            transformed_surf_pos : np.ndarray
                Transformed surface points. Shape: (N, 3).
            transformed_pharm_ancs : np.ndarray
                Transformed pharmacophore anchor positions. Shape: (P, 3).
            transformed_pharm_vecs : np.ndarray
                Transformed pharmacophore vector positions. Shape: (P, 3).
        """
        # Transform mol
        transformed_mol = update_mol_coordinates(
            mol=self.fit_molec.mol,
            coordinates=apply_SE3_transform_np(
                points=self.fit_molec.mol.GetConformer().GetPositions(),
                SE3_transform=se3_transform
            )
        )

        # Transform surface points
        transformed_surf_pos = None
        if self.fit_molec.surf_pos is not None:
            transformed_surf_pos = apply_SE3_transform_np(
                points=self.fit_molec.surf_pos,
                SE3_transform=se3_transform
            )

        # Transform pharmacophore features
        transformed_pharm_ancs = None
        transformed_pharm_vecs = None
        if self.fit_molec.pharm_ancs is not None and self.fit_molec.pharm_vecs is not None:
            transformed_pharm_ancs = apply_SE3_transform_np(
                points=self.fit_molec.pharm_ancs,
                SE3_transform=se3_transform
            )
            transformed_pharm_vecs = apply_SO3_transform_np(
                points=self.fit_molec.pharm_vecs,
                SE3_transform=se3_transform
            )
        return transformed_mol, transformed_surf_pos, transformed_pharm_ancs, transformed_pharm_vecs


    def get_transformed_molecule(self,
                                 se3_transform: np.ndarray
                                 ) -> Molecule:
        """
        Get Molecule object transformation applied to all applicable features for the fit molecule.

        Parameters
        ----------
        se3_transform : np.ndarray
            SE(3) transformation matrix. Shape: (4,4).

        Returns
        -------
        Molecule
            Molecule with transformed features.
        """
        (transformed_mol,
        transformed_surf_pos,
        transformed_pharm_ancs,
        transformed_pharm_vecs) = self.get_transformed_mol_and_feats(se3_transform=se3_transform)

        transformed_fit_molec = Molecule(mol=transformed_mol,
                                         probe_radius=self.fit_molec.probe_radius,
                                         surface_points=transformed_surf_pos,
                                         partial_charges=self.fit_molec.partial_charges,
                                         electrostatics=self.fit_molec.surf_esp,
                                         pharm_multi_vector=self.fit_molec.pharm_multi_vector,
                                         pharm_types=self.fit_molec.pharm_types,
                                         pharm_ancs=transformed_pharm_ancs,
                                         pharm_vecs=transformed_pharm_vecs
                                         )
        return transformed_fit_molec
