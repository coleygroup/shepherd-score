"""The per-molecule data channels the alignment modes read, as one table.

A channel is one named per-molecule array (heavy-atom cloud, surface points, partial
charges, pharmacophore anchors, ...) together with how to read it off a ``Molecule`` or
``MoleculeProfile``, which cached tensor attribute carries it on a ``MoleculePair``
(``_ref_<attr>_t`` / ``_fit_<attr>_t``), how a ``ProfileStore`` persists it, how it
behaves under the store's centring and canonical rotation, and what a padded slot holds.
Every mode-shaped consumer reads this table through a :class:`ModeSpec`'s channel names;
a mode that needs new data adds one row here. Pure Python + numpy, no torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

# Bases: which atom / point set a channel is indexed by. Channels sharing a basis share their
# real counts and (on the object path) their padded width.

#: basis -> (offset-table key in a store shard, dense?)
BASES = {
    "atoms": "atom_off",       # Chem.RemoveHs coordinate set (retains isotope-labelled H)
    "heavy": "heavy_off",      # strict heavy atoms (Z != 1), 1:1 with the heavy charges
    "surf": None,              # fixed-width surface (dense (K, S, ...) block in the store)
    "pharm": "pharm_off",      # pharmacophore features
    "withH": "all_off",        # every atom of the with-H conformer
    "lipo": "lipo_off",        # strict heavy, own table (Chem.RemoveHs may retain an H)
    "fukui": "fukui_off",
    "mr": "mr_off",
    "atomtype": "atomtype_off",
    "pair": None,              # a per-PAIR input that is not molecule data (vol_avoid's cloud)
}


@dataclass(frozen=True)
class Channel:
    """One per-molecule array.

    name : the channel id a :class:`ModeSpec` term refers to.
    kind : ``"points"`` (N,3) coordinates that rotate and translate; ``"vectors"`` (N,3)
        directions that rotate only; ``"scalar"`` (N,) per-point values; ``"labels"`` (N,) int
        type indices.
    basis : see :data:`BASES`; ``"pair"`` for a pair-level input read off the MoleculePair.
    attr : the pair tensor stem: ``_ref_<attr>_t`` / ``_fit_<attr>_t`` (``_<attr>_t`` for a
        pair-level channel).
    read : ``molecule -> np.ndarray`` (host); raises ``ValueError`` when the molecule lacks it.
    key : the store array key; ``None`` for a channel the store never holds.
    flag : the schema flag that says a store carries it; ``None`` = always stored.
    pad : the value a padded slot holds (0 for coordinates; the Dummy type for pharmacophore
        labels so an unmasked slot could never match; 0 for element labels).
    dtype : ``"float32"`` or ``"int64"`` (the device tensor dtype).
    profile : the attribute name on ``MoleculeProfile`` (defaults to ``key``).
    """
    name: str
    kind: str
    basis: str
    attr: str
    read: Callable
    key: Optional[str]
    flag: Optional[str] = None
    pad: float = 0
    dtype: str = "float32"
    profile: Optional[str] = None

    @property
    def off(self) -> Optional[str]:
        return BASES[self.basis]

    @property
    def dense(self) -> bool:
        return self.basis == "surf"

    @property
    def is_pair(self) -> bool:
        return self.basis == "pair"

    @property
    def ref_attr(self) -> str:
        return f"_{self.attr}_t" if self.is_pair else f"_ref_{self.attr}_t"

    @property
    def fit_attr(self) -> str:
        return f"_fit_{self.attr}_t"

    @property
    def prof(self) -> str:
        return self.profile or self.key

# Readers (Molecule or MoleculeProfile -> numpy, host side)

def _need(m, attr, what):
    v = getattr(m, attr, None)
    if v is None:
        raise ValueError(f"{what} is None; cannot align this mode")
    return v


def _heavy_positions(m):
    """Strict-heavy (Z != 1) atom coordinates ordered to match ``partial_charges[_nonH_atoms_idx]``.
    Read from the with-H conformer when present; a heavy-only profile falls back to its stored
    strict-heavy set, or to ``atom_pos`` when Chem.RemoveHs kept no H."""
    mol = getattr(m, "mol", None)
    idx = getattr(m, "_nonH_atoms_idx", None)
    if mol is not None and idx is not None:
        return np.asarray(mol.GetConformer().GetPositions())[np.asarray(idx)]
    noH = getattr(m, "atom_pos_noH", None)
    return np.asarray(m.atom_pos if noH is None else noH)


def _heavy_charges(m):
    pc = _need(m, "partial_charges", "Partial charges")
    return np.asarray(pc)[np.asarray(m._nonH_atoms_idx)]


def _read_accessor(getter, what):
    def _r(m):
        fn = getattr(m, getter, None)
        if fn is None:
            raise ValueError(f"{what}: molecule cannot provide it (no {getter}())")
        v = fn()
        if v is None:
            raise ValueError(f"{what} is None; cannot align this mode")
        return np.asarray(v)
    return _r


def _read_accessor_noH(getter, what):
    def _r(m):
        fn = getattr(m, getter, None)
        if fn is None:
            raise ValueError(f"{what}: molecule cannot provide it (no {getter}())")
        v = fn(no_H=True)
        if v is None:
            raise ValueError(f"{what} is None; cannot align this mode")
        return np.asarray(v)
    return _r


def _read_pharm(attr):
    def _r(m):
        if getattr(m, "pharm_types", None) is None:
            raise ValueError("Pharmacophores are None; cannot align this mode")
        return np.asarray(getattr(m, attr))
    return _r


def _read_surf(attr, what):
    def _r(m):
        if getattr(m, "surf_pos", None) is None:
            raise ValueError("Surface points are None; cannot align this mode")
        return np.asarray(_need(m, attr, what))
    return _r


# Padding label for pharmacophore slots: the 'Dummy' family (lookup category 3, skipped by the
# kernel). Derived by name so an upstream reorder of P_TYPES stays correct.
def _pharm_pad_type() -> int:
    from ..score.constants import P_TYPES
    return P_TYPES.index("Dummy")


#: Element-label pad: atomic number 0 (no real element); the element tables give it category 3.
ATOMTYPE_PAD = 0


CHANNELS = {}


def _reg(c: Channel) -> Channel:
    CHANNELS[c.name] = c
    return c


_reg(Channel("atoms", "points", "atoms", "xyz", lambda m: np.asarray(m.atom_pos), "atom_pos"))
_reg(Channel("heavy", "points", "heavy", "xyz_noH", _heavy_positions, "xyz_noH", flag="charges",
             profile="atom_pos_noH"))
_reg(Channel("charges", "scalar", "heavy", "xyz_esp", _heavy_charges, "charges", flag="charges",
             profile="partial_charges"))
_reg(Channel("surf", "points", "surf", "surf", _read_surf("surf_pos", "Surface points"), "surf_pos",
             flag="surf"))
_reg(Channel("surf_esp", "scalar", "surf", "surf_esp", _read_surf("surf_esp", "Surface ESP"),
             "surf_esp", flag="surf_esp"))
_reg(Channel("pharm_types", "labels", "pharm", "pharm_types", _read_pharm("pharm_types"),
             "pharm_types", flag="pharm", pad=_pharm_pad_type(), dtype="int64"))
_reg(Channel("pharm_ancs", "points", "pharm", "pharm_ancs", _read_pharm("pharm_ancs"),
             "pharm_ancs", flag="pharm"))
_reg(Channel("pharm_vecs", "vectors", "pharm", "pharm_vecs", _read_pharm("pharm_vecs"),
             "pharm_vecs", flag="pharm"))
_reg(Channel("lipo_pos", "points", "lipo", "lipo_pos",
             _read_accessor("get_lipo_positions", "Lipophilicity centers"), "lipo_pos",
             flag="lipophilicity"))
_reg(Channel("lipo", "scalar", "lipo", "lipo",
             _read_accessor_noH("get_lipophilicity", "Lipophilicity"), "lipophilicity",
             flag="lipophilicity"))
_reg(Channel("fukui_pos", "points", "fukui", "fukui_pos",
             _read_accessor("get_fukui_positions", "Fukui centers"), "fukui_pos", flag="fukui"))
_reg(Channel("fukui", "scalar", "fukui", "fukui",
             _read_accessor_noH("get_fukui", "Fukui field"), "fukui", flag="fukui"))
_reg(Channel("mr_pos", "points", "mr", "mr_pos",
             _read_accessor("get_mr_positions", "Molar-refractivity centers"), "mr_pos", flag="mr"))
_reg(Channel("mr", "scalar", "mr", "mr",
             _read_accessor_noH("get_molar_refractivity", "Molar refractivity"), "mr", flag="mr",
             profile="molar_refractivity"))
_reg(Channel("type_pos", "points", "atomtype", "type_pos",
             _read_accessor("get_atomtype_positions", "Atom-type centers"), "atomtype_pos",
             flag="atomtype"))
_reg(Channel("atomlabels", "labels", "atomtype", "atomlabels",
             _read_accessor_noH("get_atomic_numbers", "Atomic numbers"), "atomic_numbers",
             flag="atomtype", pad=ATOMTYPE_PAD, dtype="int64"))
# ShaEP combo channels: the with-H conformer, its charges and vdW radii (one basis, one table).
_reg(Channel("cwh", "points", "withH", "centers_w_H",
             lambda m: np.asarray(_need(m, "mol", "With-H conformer").GetConformer().GetPositions()),
             "cwh", flag="centers_w_H"))
# flag="charges" because this channel is the store's ``charges`` array; the with-H basis is what
# additionally implies the ``with_H`` flag (see ``screen._mode_flags``).
_reg(Channel("partial", "scalar", "withH", "partial",
             lambda m: np.asarray(_need(m, "partial_charges", "Partial charges")), "charges",
             flag="charges", profile="partial_charges"))
_reg(Channel("radii", "scalar", "withH", "radii",
             lambda m: np.asarray(_need(m, "radii", "vdW radii")), "radii", flag="radii"))
def _spread_bits(x):
    """Spread the low 10 bits of ``x`` (int64) to every third bit, for a 30-bit Morton code."""
    x = x & 0x3FF
    x = (x | (x << 16)) & 0x30000FF
    x = (x | (x << 8)) & 0x300F00F
    x = (x | (x << 4)) & 0x30C30C3
    return (x | (x << 2)) & 0x9249249


def z_order(points) -> np.ndarray:
    """``points`` (K, 3) as float32, reordered along a Z-order curve. The avoid kernels skip
    whole tiles of avoid points beyond reach of the fit, which only pays when a tile is spatially
    compact; the penalty is a plain sum over points, so the order changes only float rounding."""
    p = np.ascontiguousarray(points, dtype=np.float32)
    if len(p) < 2:
        return p
    lo = p.min(0)
    cell = ((p - lo) / np.maximum(p.max(0) - lo, 1e-6) * 1023).astype(np.int64)
    code = sum(_spread_bits(cell[:, i]) << i for i in range(3))
    return p[np.argsort(code, kind="stable")]


# The one pair-level channel: a fixed avoid cloud in the frame the pair is scored in (used as-is,
# never shifted), attached to the pair (``MoleculePair.avoid_points``) or supplied with the query
# on the screen path. Z-ordered once here, not per call.
_reg(Channel("avoid", "points", "pair", "avoid_pts",
             lambda p: z_order(_need(p, "avoid_points", "avoid_points")), None))


def channel(name: str) -> Channel:
    return CHANNELS[name]


def basis_of(names) -> dict:
    """basis -> [channel names] for the given channels, in first-seen order."""
    out = {}
    for n in names:
        out.setdefault(CHANNELS[n].basis, []).append(n)
    return out

# Store-side helpers (numpy only)

#: schema flags a store can carry, in a stable order. ``atoms`` needs none (always stored).
SCHEMA_FLAGS = ("surf", "surf_esp", "charges", "with_H", "radii", "centers_w_H", "pharm",
                "lipophilicity", "fukui", "mr", "atomtype")


def load_store_channel(arrs: dict, name: str):
    """``(flat_or_dense array, offsets or None)`` for channel ``name`` out of one shard's arrays.

    The heavy basis needs more than a key lookup: ``heavy`` / ``charges`` live on ``heavy_off``
    + ``xyz_noH`` only when some molecule's ``Chem.RemoveHs`` retained an H; otherwise
    ``atom_off`` + ``atom_pos`` already are the heavy set. On a with-H store the heavy charges
    are the with-H array gathered by ``nonH`` plus each molecule's ``all_off`` start.
    """
    c = CHANNELS[name]
    if c.is_pair:
        raise KeyError(f"{name} is a pair-level channel; a store does not hold it")
    if c.dense:
        return arrs[c.key], None
    if c.basis == "heavy":
        hoff = arrs["heavy_off"] if "heavy_off" in arrs else arrs["atom_off"]
        if name == "heavy":
            return (arrs["xyz_noH"] if "xyz_noH" in arrs else arrs["atom_pos"]), hoff
        if "all_off" in arrs:                       # with-H store: heavy = charges[all_off][nonH]
            alloff, nonH = arrs["all_off"], arrs["nonH"]
            heavy = arrs["charges"][nonH + np.repeat(alloff[:-1], np.diff(hoff))]
        else:
            heavy = arrs["charges"]
        return heavy, hoff
    return arrs[c.key], arrs[c.off]
