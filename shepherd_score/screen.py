"""Virtual screening of molecule libraries too large to hold in memory.

Screening takes two steps:

1. **Build database (one-time cost).** :class:`ProfileStore` saves the arrays each library
   molecule needs for alignment (atoms, surface, charges, pharmacophores, ...) to sharded
   files on disk. Only the arrays needed by the requested ``modes`` are kept.
2. **Screen database.** :func:`screen` streams the from disk to align every library molecule
   to the query, and keeps the best ``top_k``. At most two shards are in memory at a time.
   :func:`screen_many` does the same for several queries in a single pass over the library.

Example::

    from shepherd_score.container import Molecule
    from shepherd_score.screen import ProfileStore, screen

    with ProfileStore.create("lib.fss", num_surf_points=200, modes=("surf_esp",)) as store:
        for mol in library_rdkit_mols:
            store.add(Molecule(mol, num_surf_points=200, pharm_multi_vector=False))

    query = Molecule(query_rdkit_mol, num_surf_points=200, pharm_multi_vector=False)
    hits = screen(query, ProfileStore.open("lib.fss"), mode="surf_esp", lam=0.3, top_k=1000)
    for hit in hits[:10]:
        print(hit.id, hit.score)          # hit.transform: 4x4 pose of the library molecule

Results are :data:`Hit` tuples ``(score, id, transform)``, best first. Pass ``ndev=N`` to
use N GPUs; :func:`close_multigpu_pool` stops those worker processes.

The public API is at the top of this file.
"""
from __future__ import annotations

import atexit
import copy
import heapq
import json
import os
import re
from collections import namedtuple
from operator import attrgetter
from typing import Iterator, List, Optional, Sequence

import numpy as np

from shepherd_score.accel._modes import (
    MODE_ATTRS as _MODE_ATTRS, canonical as _canon_mode,
    CONST_SEED_MODES as _CONST_SEED_MODES, SPECS as _SPECS, spec_of as _spec_of,
)
from shepherd_score.accel.channels import CHANNELS as _CHANNELS, SCHEMA_FLAGS as _SCHEMA_FLAGS

__all__ = ["MoleculeProfile", "ProfileStore", "screen", "screen_many", "Hit",
           "close_multigpu_pool"]


#: One screening result, holding the library molecule's ``score``, its store ``id`` and the
#: 4x4 SE(3) ``transform`` that aligns it onto the query.
Hit = namedtuple("Hit", ["score", "id", "transform"])

# Names and valid modes come from the mode registry, so this module and the accel layers agree.
_TRANSFORM_ATTR = {m: a[0] for m, a in _MODE_ATTRS.items()}
_SCORE_ATTR = {m: a[1] for m, a in _MODE_ATTRS.items()}
# Prebuilt score readers, so a shard's score vector is one ``map`` pass rather than K ``getattr``s.
_SCORE_GETTER = {m: attrgetter(a) for m, a in _SCORE_ATTR.items()}
_VALID_MODES = tuple(_SCORE_ATTR)
# Modes whose ``alpha`` auto-defaults to ALPHA(num_surf_points): those whose shape channel is the
# surface cloud. A mode that reads surfaces for another channel only must be given alpha.
_SURF_ALPHA_MODES = {m for m, sp in _SPECS.items()
                     if any(t.kernel in ("shape", "esp") and t.ref and t.ref[0] == "surf"
                            for t in sp.terms)}


def screen(query, store: "ProfileStore", mode: str = "surf_esp", *,
           backend: Optional[str] = None, do_center: Optional[bool] = None,
           top_k: int = 1000, ndev: Optional[int] = None,
           worker_threads: Optional[int] = None,
           scores_out: Optional[np.ndarray] = None, alpha: Optional[float] = None,
           progress: bool = False, **align_kwargs) -> List["Hit"]:
    """Align every molecule in ``store`` to ``query`` and return the best ``top_k`` hits.

    The library is streamed from disk one shard at a time, so memory use does not grow
    with library size. To screen several queries, use :func:`screen_many`, which reads
    the library only once for all of them.

    Parameters
    ----------
    query : Molecule or MoleculeProfile
        The reference molecule. It must carry the arrays ``mode`` needs (e.g. a surface
        with ``num_surf_points`` equal to the store's for the surface modes). It is never
        modified.
    store : ProfileStore
        The library, from :meth:`ProfileStore.open`.
    mode : str
        Similarity objective, e.g. ``"vol"``, ``"vol_esp"``, ``"surf_esp"``, ``"pharm"``
        or ``"vol_color"``. It must be one of the modes the store was built for (see
        ``store.modes`` and :meth:`ProfileStore.create`). Default ``"surf_esp"``.
    backend : str, optional
        Compute backend. Defaults to ``"triton"`` when a GPU is available, else
        ``"numba"``. ``"jax"`` is accepted but uses the slower per-pair path.
    do_center : bool, optional
        Whether to center molecules on their heavy-atom centroid before aligning. By
        default, on a pre-centered store (the :meth:`ProfileStore.create` default) only the
        query is centered; otherwise both the query and the library molecules are.
    top_k : int
        Number of best hits to return. Default 1000.
    ndev : int, optional
        Spread the screen over this many GPUs, one worker process per GPU. The workers
        start on the first such call and are reused; :func:`close_multigpu_pool` stops
        them. Requires a pre-centered store and a GPU backend.
    worker_threads : int, optional
        CPU threads per GPU worker when ``ndev > 1``. Defaults to the available cores
        divided by ``ndev``.
    scores_out : np.ndarray, optional
        A preallocated ``(len(store),)`` array (e.g. an ``np.memmap``) that receives every
        library molecule's score in store order, not just the top ``top_k``. Not
        supported with ``ndev > 1``.
    alpha : float, optional
        Width of the shape Gaussians. Defaults to ``ALPHA(num_surf_points)`` for the
        surface modes and ``0.81`` otherwise. In ``vol_and_surf_esp``, ``0.81`` scores
        volumetric shape and any other value scores surface shape. Ignored by ``pharm``.
    progress : bool
        Print a line after each shard. Default False.
    **align_kwargs
        Mode parameters and optimizer settings passed to the aligner, for example
        ``lam`` (ESP weight), ``color_weight``, ``tversky_alpha``/``tversky_beta``,
        ``similarity`` (``pharm``), ``avoid_points`` (required by ``vol_avoid``: a
        ``(K, 3)`` cloud in the query's frame), ``num_repeats``, ``max_num_steps`` and
        ``lr``. Unset parameters take the mode's defaults. ``trans_init=True`` is
        supported but uses the slower per-pair path.

    Returns
    -------
    list[Hit]
        Up to ``top_k`` ``Hit(score, id, transform)`` tuples, best first. ``id`` is the id
        given to :meth:`ProfileStore.add` (by default the molecule's index in the store).
        ``transform`` is the 4x4 SE(3) matrix that moves the library molecule onto the
        query, acting on centered coordinates when centring is on.
    """
    return screen_many([query], store, mode, backend=backend, do_center=do_center,
                       top_k=top_k, ndev=ndev, worker_threads=worker_threads,
                       scores_out=scores_out, alpha=alpha,
                       progress=progress, **align_kwargs)[0]


def screen_many(queries: Sequence, store: "ProfileStore", mode: str = "surf_esp", *,
                backend: Optional[str] = None, do_center: Optional[bool] = None,
                top_k: int = 1000, ndev: Optional[int] = None,
                worker_threads: Optional[int] = None,
                scores_out=None, alpha: Optional[float] = None,
                progress: bool = False, **align_kwargs) -> List[List["Hit"]]:
    """Screen several queries against ``store`` in one pass over the library.

    Equivalent to calling :func:`screen` once per query, but each shard is read from disk
    (and uploaded to the GPU) only once and aligned against every query in turn.

    Parameters not listed here are the same as for :func:`screen`.

    Parameters
    ----------
    queries : sequence of Molecule or MoleculeProfile
        The reference molecules. None of them is modified.
    scores_out : list of np.ndarray, optional
        One preallocated ``(len(store),)`` array per query (or a single array when there
        is one query); see :func:`screen`. Not supported with ``ndev > 1``.

    Returns
    -------
    list[list[Hit]]
        ``out[j]`` holds the hits for ``queries[j]``, best first.
    """
    import torch
    queries = list(queries)
    mode = _canon_mode(mode)                      # accept legacy esp / esp_combo everywhere below
    align_kwargs = _resolve_screen(store, mode, alpha, align_kwargs)
    backend = backend or _default_backend()

    _spec = _spec_of(mode)
    _chans = [_CHANNELS[c] for c in _spec.all_channels()]
    if any(c.basis == "surf" for c in _chans):
        for q in queries:
            qn = getattr(q, "num_surf_points", None)
            if qn is not None and qn != store.num_surf_points:
                raise ValueError(f"query num_surf_points ({qn}) != store "
                                 f"({store.num_surf_points}); ALPHA is calibrated to it")

    # Fast-path query preconditions: read every channel once so a missing field is reported
    # here, by name, rather than mid-screen.
    for q in queries:
        for c in _chans:
            if c.is_pair:
                continue
            try:
                c.read(q)
            except ValueError as e:
                raise ValueError(f"{mode} query cannot provide {c.name}: {e}") from e

    center = (not store.pre_centered) if do_center is None else bool(do_center)
    if store.pre_centered or center:
        qs = [_centered_copy(q) for q in queries]
        center_profiles = (not store.pre_centered) and center
    else:
        qs = list(queries)
        center_profiles = False

    fast = (mode in _FAST_MODES and store.pre_centered
            and not align_kwargs.get("trans_init") and backend != "jax")
    device = (torch.device("cpu") if backend in ("numba", "cpu")
              else torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    # The fast CPU path runs the batched kernels through numba; without it the aligner falls
    # into a per-pair fallback the ``_FastPair`` stand-ins cannot serve. Fail clearly up front.
    if fast and device.type == "cpu":
        try:
            import numba  # noqa: F401
        except ImportError:
            raise ImportError(
                "fast CPU screen requires numba (the batched CPU kernels run on it) -- "
                "install it (pip install numba) or screen on a GPU backend") from None

    if ndev and ndev > 1:
        if not fast:
            raise ValueError("ndev>1 requires the fast path (a pre-centered store, a "
                             f"{sorted(_FAST_MODES)} mode, trans_init=False, GPU backend)")
        if scores_out is not None:
            # The multi-GPU workers return only per-query top-K heaps, so a preallocated score
            # array could not be filled; fail rather than leave it silently unwritten.
            raise ValueError("scores_out is not supported with ndev>1 (multi-GPU screening "
                             "returns top-K hits only). Run single-process for full score vectors.")
        heaps = _screen_many_multigpu(qs, store.path, mode, ndev,
                                      _fast_batch_kwargs(mode, align_kwargs), top_k, progress,
                                      worker_threads=worker_threads)
        # The multi-GPU workers rebuild their own device tensors, so the avoid cloud crosses as
        # the plain numpy array in ``align_kwargs`` and is uploaded inside each worker.
        return [h.sorted() for h in heaps]

    so = _normalize_scores_out(scores_out, len(queries))
    if fast:
        qs_ref = [_query_ref_arrays(q, mode) for q in qs]
        batch_kw = _fast_batch_kwargs(mode, align_kwargs)
        batch_kw = _with_avoid(batch_kw, align_kwargs, device)
    else:
        qs_ref = qs
        batch_kw = None
    heaps = _run_shards_inproc(store, range(store.num_shards), qs_ref, mode, device, top_k,
                               batch_kw, align_kwargs, backend, fast, center_profiles, so,
                               progress, len(store))
    return [h.sorted() for h in heaps]


def close_multigpu_pool():
    """Stop the GPU worker processes started by ``screen(..., ndev>1)``, if any.

    This runs automatically at interpreter exit; call it to free the GPUs sooner. The next
    multi-GPU screen starts a new pool.
    """
    global _MGPU_POOL
    pool, _MGPU_POOL = _MGPU_POOL, None
    if pool is not None:
        _stop_mgpu_pool(pool)


def _stop_mgpu_pool(pool):
    from shepherd_score.accel._procs import shutdown
    shutdown(pool["job_qs"], pool["procs"], pool["conns"], grace=10.0)


# --------------------------------------------------------------------------- #
# Data storage

class MoleculeProfile:
    """A lightweight, RDKit-free copy of the arrays a :class:`~shepherd_score.container.Molecule`
    needs for alignment.

    This is what a :class:`ProfileStore` holds per library molecule. It can be used as a
    ``screen`` query or inside a ``MoleculePair`` in place of a ``Molecule``, but it is not a
    general replacement: it has no RDKit ``Mol`` (``.mol`` is ``None``, or a small stand-in
    for ``vol_and_surf_esp``) and none of ``Molecule``'s build settings.

    Build one with :meth:`from_molecule`. As in ``Molecule``, ``atom_pos`` holds heavy-atom
    coordinates; ``partial_charges`` holds either heavy-atom charges or all-atom charges with
    ``_nonH_atoms_idx`` selecting the heavy ones, depending on what the store needs.
    """

    __slots__ = ("atom_pos", "atom_pos_noH", "surf_pos", "surf_esp", "partial_charges", "radii",
                 "_nonH_atoms_idx", "pharm_types", "pharm_ancs", "pharm_vecs",
                 "lipo_pos", "lipophilicity",
                 "fukui_pos", "fukui",
                 "mr_pos", "molar_refractivity",
                 "atomtype_pos", "atomic_numbers",
                 "num_surf_points", "mol", "id", "rot")

    def __init__(self, *, atom_pos, surf_pos=None, surf_esp=None,
                 partial_charges=None, radii=None, nonH_atoms_idx=None,
                 pharm_types=None, pharm_ancs=None, pharm_vecs=None,
                 lipo_pos=None, lipophilicity=None,
                 fukui_pos=None, fukui=None,
                 mr_pos=None, molar_refractivity=None,
                 atomtype_pos=None, atomic_numbers=None,
                 centers_w_H=None, atom_pos_noH=None, id=None, rot=None):
        self.atom_pos = _f32(atom_pos)
        #: (3,3) float32 principal-axis rotation applied at build time (canonical stores only),
        #: needed to map an alignment transform back to the molecule's original frame.
        self.rot = None if rot is None else np.asarray(rot, np.float32)
        # Strict-heavy vol_esp centers (1:1 with the heavy charges); None when identical to
        # atom_pos, in which case callers use atom_pos.
        self.atom_pos_noH = _f32(atom_pos_noH)
        self.surf_pos = _f32(surf_pos)
        self.surf_esp = _f32(surf_esp)
        self.partial_charges = _f32(partial_charges)
        self.radii = _f32(radii)
        if nonH_atoms_idx is not None:
            self._nonH_atoms_idx = np.asarray(nonH_atoms_idx, dtype=np.int64)
        elif self.partial_charges is not None:
            # heavy-atom charges stored already-indexed -> identity selection so
            # the aligner's ``partial_charges[_nonH_atoms_idx]`` is a no-op.
            self._nonH_atoms_idx = np.arange(len(self.partial_charges), dtype=np.int64)
        else:
            self._nonH_atoms_idx = None
        self.pharm_types = None if pharm_types is None else np.asarray(pharm_types)
        self.pharm_ancs = _f32(pharm_ancs)
        self.pharm_vecs = _f32(pharm_vecs)
        # Per-channel true-heavy centers and per-atom scalars (already heavy-sliced); their counts
        # can differ from atom_pos when RemoveHs retained an H. ``atomic_numbers`` is a
        # categorical label the color kernel matches on, kept float32 like the other channels.
        self.lipo_pos = _f32(lipo_pos)
        self.lipophilicity = _f32(lipophilicity)
        self.fukui_pos = _f32(fukui_pos)
        self.fukui = _f32(fukui)
        self.mr_pos = _f32(mr_pos)
        self.molar_refractivity = _f32(molar_refractivity)
        self.atomtype_pos = _f32(atomtype_pos)
        self.atomic_numbers = _f32(atomic_numbers)
        self.num_surf_points = None if self.surf_pos is None else len(self.surf_pos)
        self.mol = _MolShim(_f32(centers_w_H)) if centers_w_H is not None else None
        self.id = id

    def center_to(self, xyz_means) -> None:
        """Translate every coordinate array by ``-xyz_means``, like ``Molecule.center_to``."""
        mu = np.asarray(xyz_means, dtype=np.float32)
        self.atom_pos = self.atom_pos - mu
        if self.atom_pos_noH is not None:
            self.atom_pos_noH = self.atom_pos_noH - mu
        if self.surf_pos is not None:
            self.surf_pos = self.surf_pos - mu
        if self.pharm_ancs is not None:
            self.pharm_ancs = self.pharm_ancs - mu
        if self.lipo_pos is not None:
            self.lipo_pos = self.lipo_pos - mu
        if self.fukui_pos is not None:
            self.fukui_pos = self.fukui_pos - mu
        if self.mr_pos is not None:
            self.mr_pos = self.mr_pos - mu
        if self.atomtype_pos is not None:
            self.atomtype_pos = self.atomtype_pos - mu
        if self.mol is not None:
            self.mol = _MolShim(self.mol.GetConformer().GetPositions() - mu)

    def get_lipo_positions(self):
        """True-heavy lipophilicity centers; mirrors ``Molecule.get_lipo_positions()``."""
        return self.lipo_pos

    def get_lipophilicity(self, no_H: bool = True):
        """Per-atom Crippen logP, stored heavy-sliced; ``no_H`` is accepted for signature parity."""
        return self.lipophilicity

    def get_fukui_positions(self):
        """True-heavy Fukui centers; mirrors ``Molecule.get_fukui_positions()``."""
        return self.fukui_pos

    def get_fukui(self, no_H: bool = True):
        """Per-atom Fukui dual descriptor, stored heavy-sliced; ``no_H`` is for signature parity."""
        return self.fukui

    def get_mr_positions(self):
        """True-heavy molar-refractivity centers; mirrors ``Molecule.get_mr_positions()``."""
        return self.mr_pos

    def get_molar_refractivity(self, no_H: bool = True):
        """Per-atom Crippen molar refractivity, stored heavy-sliced."""
        return self.molar_refractivity

    def get_atomtype_positions(self):
        """True-heavy element-label centers; mirrors ``Molecule.get_atomtype_positions()``."""
        return self.atomtype_pos

    def get_atomic_numbers(self, no_H: bool = True):
        """Per-atom atomic numbers, stored heavy-sliced."""
        return self.atomic_numbers

    @classmethod
    def from_molecule(cls, m, *, modes=_VALID_MODES, id=None) -> "MoleculeProfile":
        """Build a profile from a ``Molecule`` (or another ``MoleculeProfile``), keeping only
        the arrays ``modes`` need. Coordinates are copied as-is (no centring)."""
        sch = _schema_from_modes(modes)
        return _profile_from_schema(m, sch, id=id, pre_center=False)


class ProfileStore:
    """An on-disk library of molecules prepared for :func:`screen`.

    A store is written once and then screened any number of times::

        with ProfileStore.create("lib.fss", num_surf_points=200, modes=("surf_esp",)) as store:
            for mol in molecules:          # Molecule objects
                store.add(mol)             # or store.add(mol, id="ZINC000123")
        store = ProfileStore.open("lib.fss")
        len(store), store.modes            # number of molecules, supported modes

    On disk it is a directory holding ``manifest.json`` (settings and the shard list) and
    the shards, each holding up to ``shard_size`` molecules. Shards are memory-mapped
    ``shard_NNNNN__<array>.npy`` files by default, or one ``shard_NNNNN.npz`` per shard
    with ``shard_format="npz"``.
    """

    MANIFEST = "manifest.json"
    VERSION = 1

    def __init__(self, path, manifest, rw):
        self.path = path
        self.manifest = manifest
        self._rw = rw
        # writer state
        self._buf: List[MoleculeProfile] = []
        self._n = int(manifest.get("n_total", 0))
        self._shard_id = len(manifest.get("shards", []))

    # ---- writer ---------------------------------------------------------- #
    @classmethod
    def create(cls, path, *, num_surf_points: int, modes: Sequence[str],
               dtype: str = "float16", shard_size: int = 100_000,
               pre_centered: bool = True, overwrite: bool = False,
               canonical: Optional[bool] = None,
               shard_format: str = "npy") -> "ProfileStore":
        """Create a new, empty store at ``path`` and open it for writing.

        Add molecules with :meth:`add`, then :meth:`close` it (or use it as a context
        manager, as in the class example).

        Parameters
        ----------
        path : str
            Directory to write. It is created if missing.
        num_surf_points : int
            Number of surface points per molecule. Queries must use the same number, and it
            should be between 50 and 400, the range ``ALPHA`` is calibrated for.
        modes : sequence of str
            The modes this store will be screened with, e.g. ``("vol", "surf_esp")``. Only
            the arrays those modes need are saved. ``"vol"`` works on any store.
        dtype : {"float16", "float32"}
            On-disk precision. ``"float16"`` (default) halves disk use and read time at
            about 0.01 Å coordinate precision; arrays are widened to float32 when read.
        shard_size : int
            Molecules per shard. Default 100,000.
        pre_centered : bool
            Center each molecule on its heavy-atom centroid before saving. Default True,
            which is required by the fast and multi-GPU screening paths.
        overwrite : bool
            Replace an existing store at ``path``. Only the store's own manifest and shard
            files are deleted. Default False.
        canonical : bool, optional
            Also rotate each molecule into its principal-axis frame before saving. Screens in
            the volume modes (``accel._modes.CONST_SEED_MODES``) can then use one shared set
            of starting orientations, which is faster. Scores differ slightly from a
            non-canonical store. Returned transforms still refer to the molecule's own
            (centered) frame. Requires ``pre_centered``. By default this is on when
            ``pre_centered`` is set and ``modes`` includes a volume mode.
        shard_format : {"npy", "npz"}
            ``"npy"`` (default) writes one memory-mappable file per array; ``"npz"`` writes
            one archive per shard.

        Notes
        -----
        Only one process may write to a store. To build a library in parallel, give each
        worker its own store (``lib.part0.fss``, ``lib.part1.fss``, ...), screen each part
        and merge the hit lists.
        """
        if dtype not in ("float16", "float32"):
            raise ValueError("dtype must be 'float16' or 'float32'")
        modes = tuple(modes)
        if canonical is None:
            # Canonical when the store serves a constant-seed mode and is pre-centered; a store
            # serving only surface- or anchor-seeded modes keeps raw centered coordinates.
            canonical = bool(pre_centered) and any(_canon_mode(m) in _CONST_SEED_MODES
                                                   for m in modes)
        elif canonical and not pre_centered:
            raise ValueError("canonical stores require pre_centered=True (axes are centroid-relative)")
        schema = _schema_from_modes(modes)
        os.makedirs(path, exist_ok=True)
        manifest_path = os.path.join(path, cls.MANIFEST)
        if os.path.exists(manifest_path) and not overwrite:
            raise FileExistsError(
                f"{manifest_path} already exists; pass overwrite=True to replace it")
        if overwrite:
            # Remove only this store's own files (manifest and shard sequence, either format),
            # never every ``.npz`` in the directory, which may hold unrelated data.
            for f in os.listdir(path):
                if (f == cls.MANIFEST or re.fullmatch(r"shard_\d{5}\.npz", f)
                        or re.fullmatch(r"shard_\d{5}__\w+\.npy", f)):
                    os.remove(os.path.join(path, f))
        manifest = dict(version=cls.VERSION, num_surf_points=int(num_surf_points),
                        modes=list(modes), schema=schema, dtype=dtype,
                        shard_size=int(shard_size), pre_centered=bool(pre_centered),
                        canonical=bool(canonical),
                        shard_format=str(shard_format),
                        n_total=0, shards=[])
        store = cls(path, manifest, "w")
        store._write_manifest()
        return store

    def add(self, molecule, id=None) -> None:
        """Add one ``Molecule`` (or ``MoleculeProfile``) to the store.

        Only the arrays the store's modes need are kept. ``id`` is returned in each
        :data:`Hit`; it defaults to the molecule's index in the store. A shard is written to
        disk every ``shard_size`` molecules.
        """
        if self._rw != "w":
            raise RuntimeError("store is open for reading")
        if id is None:
            id = self._n + len(self._buf)
        prof = _profile_from_schema(molecule, self.manifest["schema"], id=id,
                                    pre_center=self.manifest["pre_centered"],
                                    canonical=bool(self.manifest.get("canonical", False)))
        self._buf.append(prof)
        if len(self._buf) >= self.manifest["shard_size"]:
            self._flush()

    def add_profile(self, profile: "MoleculeProfile", id=None) -> None:
        """Add a pre-built :class:`MoleculeProfile` directly (no extraction)."""
        if not isinstance(profile, MoleculeProfile):
            raise TypeError("add_profile expects a MoleculeProfile")
        self.add(profile, id=id if id is not None else profile.id)

    def close(self) -> None:
        """Write any buffered molecules and the manifest. Does nothing for a read-only store."""
        if self._rw == "w":
            self._flush()
            self._write_manifest()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _write_manifest(self) -> None:
        with open(os.path.join(self.path, self.MANIFEST), "w") as fh:
            json.dump(self.manifest, fh, indent=2)

    def _flush(self) -> None:
        if not self._buf:
            return
        arrs = self._concat(self._buf)
        if self.manifest.get("shard_format") == "npy":
            # One .npy per array rather than one .npz per shard, so the reader can memory-map
            # them: an .npz is a ZIP whose members are CRC-checked on every read. np.save aligns
            # the data start, which is what lets np.load return a real mmap.
            name = f"shard_{self._shard_id:05d}"
            for k, v in arrs.items():
                np.save(os.path.join(self.path, f"{name}__{k}.npy"), v)
            entry = dict(name=name, n=len(self._buf), start=self._n, keys=sorted(arrs))
        else:
            name = f"shard_{self._shard_id:05d}.npz"
            np.savez(os.path.join(self.path, name), **arrs)
            entry = dict(name=name, n=len(self._buf), start=self._n)
        self.manifest["shards"].append(entry)
        self._n += len(self._buf)
        self.manifest["n_total"] = self._n
        self._buf = []
        self._shard_id += 1
        self._write_manifest()

    def _concat(self, recs: List["MoleculeProfile"]) -> dict:
        """Pack one shard: every basis the schema carries as a flat array plus its own CSR offset
        table (a dense ``np.stack`` block for the fixed-width surface). Each per-atom field keeps
        its own table because ``atom_pos`` (the ``RemoveHs`` set) can be longer than the true-heavy set.
        """
        sch = self.manifest["schema"]
        dt = np.float16 if self.manifest["dtype"] == "float16" else np.float32

        def offsets(lengths):
            return np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)

        out = {"ids": np.array([r.id for r in recs])}
        out["atom_off"] = offsets([len(r.atom_pos) for r in recs])
        out["atom_pos"] = np.concatenate([r.atom_pos for r in recs]).astype(dt)
        if recs[0].rot is not None:
            # float32 regardless of the store dtype; a float16 rotation is too coarse to keep a pose.
            out["rot"] = np.stack([r.rot for r in recs]).astype(np.float32)

        if sch["surf"]:
            out["surf_pos"] = np.stack([r.surf_pos for r in recs]).astype(dt)
        if sch["surf_esp"]:
            out["surf_esp"] = np.stack([r.surf_esp for r in recs]).astype(dt)
        if sch["charges"]:
            if sch["with_H"]:
                out["all_off"] = offsets([len(r.partial_charges) for r in recs])
                out["charges"] = np.concatenate([r.partial_charges for r in recs]).astype(dt)
                out["nonH"] = np.concatenate([r._nonH_atoms_idx for r in recs]).astype(np.int64)
                if sch["radii"]:
                    out["radii"] = np.concatenate([r.radii for r in recs]).astype(dt)
                if sch["centers_w_H"]:
                    out["cwh"] = np.concatenate(
                        [r.mol.GetConformer().GetPositions() for r in recs]).astype(dt)
                heavy_lens = [len(r._nonH_atoms_idx) for r in recs]
            else:
                out["charges"] = np.concatenate([r.partial_charges for r in recs]).astype(dt)
                heavy_lens = [len(r.partial_charges) for r in recs]
            # Heavy centers 1:1 with the heavy charges, emitted only when some molecule's RemoveHs
            # retained an H; the reader falls back to atom_off when the keys are absent.
            if any(r.atom_pos_noH is not None for r in recs):
                out["heavy_off"] = offsets(heavy_lens)
                out["xyz_noH"] = np.concatenate(
                    [(r.atom_pos_noH if r.atom_pos_noH is not None else r.atom_pos)
                     for r in recs]).astype(dt)
        for _basis, flag, off_key, attrs in _BASIS_TABLE:
            if not sch.get(flag, False):
                continue
            first = attrs[0][0]
            out[off_key] = offsets([len(getattr(r, first)) for r in recs])
            for attr, key in attrs:
                vals = [getattr(r, attr) for r in recs]
                kdt = np.int32 if attr == "pharm_types" else dt
                out[key] = np.concatenate(vals).astype(kdt)
        return out

    # ---- reader ---------------------------------------------------------- #
    @classmethod
    def open(cls, path) -> "ProfileStore":
        """Open an existing store for screening (read-only)."""
        with open(os.path.join(path, cls.MANIFEST)) as fh:
            manifest = json.load(fh)
        ver = manifest.get("version")
        if ver != cls.VERSION:
            raise ValueError(
                f"ProfileStore at {path!r} has on-disk format version {ver!r}, but this "
                f"shepherd_score reads version {cls.VERSION}. Rebuild the store.")
        return cls(path, manifest, "r")

    def __len__(self) -> int:
        """Number of molecules in the store."""
        return int(self.manifest["n_total"])

    @property
    def num_surf_points(self) -> int:
        """Surface points per molecule; queries must match it."""
        return int(self.manifest["num_surf_points"])

    @property
    def modes(self) -> tuple:
        """The modes the store was created for."""
        return tuple(self.manifest["modes"])

    @property
    def schema(self) -> dict:
        """Which arrays are stored, as ``{flag: bool}``."""
        return self.manifest["schema"]

    @property
    def pre_centered(self) -> bool:
        """Whether molecules were centered on their heavy-atom centroid when added."""
        return bool(self.manifest["pre_centered"])

    @property
    def canonical(self) -> bool:
        """Whether molecules were rotated into their principal-axis frame when added.

        The rotation is saved per molecule and undone in every returned transform. Stores
        written before this option existed report False.
        """
        return bool(self.manifest.get("canonical", False))

    def supports(self, mode: str) -> bool:
        """Whether the store holds every array ``mode`` needs."""
        return _store_supports(self.manifest["schema"], mode)

    @property
    def num_shards(self) -> int:
        """Number of shards on disk."""
        return len(self.manifest["shards"])

    def _load_raw(self, sh) -> dict:
        """Materialise one shard's arrays as ``{name: np.ndarray}`` without the per-molecule
        split. An ``npy``-format store is memory-mapped copy-on-write (nothing here writes, so
        nothing is copied); an ``.npz`` store is read and closed before returning."""
        if sh.get("keys") is not None:
            base = os.path.join(self.path, sh["name"])
            # mmap_mode="c" rather than "r": a read-only mapping gives a non-writable array,
            # and ``torch.from_numpy`` warns on those every call.
            return {k: np.load(f"{base}__{k}.npy", mmap_mode="c") for k in sh["keys"]}
        with np.load(os.path.join(self.path, sh["name"])) as data:
            return {k: data[k] for k in data.files}

    def read_shard(self, idx: int) -> tuple:
        """Return ``(shard_meta, arrays_dict)`` for shard ``idx``. Called on a background thread
        by :func:`_iter_shards_prefetched`, so it must touch no shared mutable state: it only
        indexes the read-only manifest and opens its own file handle."""
        sh = self.manifest["shards"][idx]
        return sh, self._load_raw(sh)

    def iter_shards(self) -> Iterator[List["MoleculeProfile"]]:
        """Yield one shard at a time as ``list[MoleculeProfile]``, via :meth:`_load_raw`, the one
        place that knows how a shard is stored."""
        for sh in self.manifest["shards"]:
            yield self._reconstruct(self._load_raw(sh), sh)

    def read_profiles(self, idx: int) -> List["MoleculeProfile"]:
        """Reconstruct shard ``idx`` as ``list[MoleculeProfile]`` (random access)."""
        sh, arrs = self.read_shard(idx)
        return self._reconstruct(arrs, sh)

    def _reconstruct(self, data, sh) -> List["MoleculeProfile"]:
        """Split one shard's arrays back into ``MoleculeProfile``s. ``rot`` is not restored: a
        profile read from a canonical store is in the canonical frame, and the array path composes
        poses from the shard's ``rot`` instead."""
        sch = self.manifest["schema"]
        n = sh["n"]
        files = set(data.files) if hasattr(data, "files") else set(data.keys())
        atom_off = data["atom_off"]
        atom_pos = data["atom_pos"]
        ids = data["ids"]
        surf_pos = data["surf_pos"] if sch["surf"] else None
        surf_esp = data["surf_esp"] if sch["surf_esp"] else None
        all_off = data["all_off"] if (sch["charges"] and sch["with_H"]) else None
        # Heavy charges, ``nonH`` and ``xyz_noH`` are laid out by strict-heavy count, which differs
        # from ``atom_off`` (the RemoveHs count) when RemoveHs retained an H.
        heavy_off = data["heavy_off"] if "heavy_off" in files else atom_off
        offs = {flag: data[off_key] for _b, flag, off_key, _a in _BASIS_TABLE
                if sch.get(flag, False)}

        out = []
        for i in range(n):
            a0, a1 = int(atom_off[i]), int(atom_off[i + 1])
            h0, h1 = int(heavy_off[i]), int(heavy_off[i + 1])
            kw = dict(atom_pos=atom_pos[a0:a1], id=_id_to_py(ids[i]))
            if sch["surf"]:
                kw["surf_pos"] = surf_pos[i]
            if sch["surf_esp"]:
                kw["surf_esp"] = surf_esp[i]
            if sch["charges"]:
                if sch["with_H"]:
                    c0, c1 = int(all_off[i]), int(all_off[i + 1])
                    kw["partial_charges"] = data["charges"][c0:c1]
                    kw["nonH_atoms_idx"] = data["nonH"][h0:h1]
                    if sch["radii"] and "radii" in files:
                        kw["radii"] = data["radii"][c0:c1]
                    if sch["centers_w_H"] and "cwh" in files:
                        kw["centers_w_H"] = data["cwh"][c0:c1]
                else:
                    kw["partial_charges"] = data["charges"][h0:h1]
                if "xyz_noH" in files:
                    hp = data["xyz_noH"][h0:h1]
                    if hp.shape != kw["atom_pos"].shape or not np.array_equal(hp, kw["atom_pos"]):
                        kw["atom_pos_noH"] = hp
            for _b, flag, _off_key, attrs in _BASIS_TABLE:
                if not sch.get(flag, False):
                    continue
                off = offs[flag]
                p0, p1 = int(off[i]), int(off[i + 1])
                for attr, key in attrs:
                    kw[attr] = data[key][p0:p1]
            out.append(MoleculeProfile(**kw))
        return out


# --------------------------------------------------------------------------- #
# Internal: profile extraction and store schema

class _ConformerShim:
    __slots__ = ("_p",)

    def __init__(self, p):
        self._p = p

    def GetPositions(self):
        return self._p


class _MolShim:
    """Stand-in so ``profile.mol.GetConformer().GetPositions()`` returns the stored with-H atom
    centers, the only RDKit access the batched ``vol_and_surf_esp`` aligner makes."""
    __slots__ = ("_c",)

    def __init__(self, centers_w_H):
        self._c = _ConformerShim(centers_w_H)

    def GetConformer(self, *args, **kwargs):
        return self._c


def _f32(a):
    # Always an independent C-contiguous float32 copy: on the reload path a view into a shard
    # slice would keep the whole shard's buffers alive for as long as any profile survives.
    return None if a is None else np.array(a, dtype=np.float32, order="C")


def _heavy_positions(m):
    """Strict-heavy (Z != 1) atom coordinates ordered to match ``partial_charges[_nonH_atoms_idx]``,
    read from the with-H conformer when present, else ``atom_pos``. ``atom_pos`` is the
    ``RemoveHs`` set, which is longer than the heavy charges when an H was retained."""
    mol = getattr(m, "mol", None)
    idx = getattr(m, "_nonH_atoms_idx", None)
    if mol is not None and idx is not None:
        return np.asarray(mol.GetConformer().GetPositions())[np.asarray(idx)]
    return np.asarray(m.atom_pos)


def _mode_flags(mode: str) -> set:
    """The schema flags ``mode`` needs, read off its channels; ``atom_pos`` is unconditional and
    has none. ``vol_and_surf_esp`` reaches ``with_H`` through its ``partial`` channel, which is
    why a combo store keeps the with-H charges plus the ``nonH`` index rather than the heavy slice."""
    spec = _spec_of(mode)
    flags = set()
    for name in spec.all_channels():
        ch = _CHANNELS[name]
        if ch.is_pair:
            continue
        if ch.flag:
            flags.add(ch.flag)
        if ch.basis == "withH":
            # A with-H channel needs the store laid out on the with-H basis: ``_flush`` writes
            # ``all_off`` / ``nonH`` / ``radii`` / ``cwh`` only under ``schema["with_H"]``.
            flags.add("with_H")
    return flags


def _schema_from_modes(modes) -> dict:
    """The store schema serving ``modes``: one boolean per channel flag, derived from what each
    mode's channels read. Every flag in ``SCHEMA_FLAGS`` is always present (a reader may use
    ``schema[flag]``); only the values vary with the mode set."""
    modes = {_canon_mode(m) for m in modes}            # accept legacy esp / esp_combo
    unknown = modes - set(_VALID_MODES)
    if unknown:
        raise ValueError(f"unknown modes {sorted(unknown)}; valid: {list(_VALID_MODES)}")
    want = set()
    for m in modes:
        want |= _mode_flags(m)
    return {f: (f in want) for f in _SCHEMA_FLAGS}


def _store_supports(schema: dict, mode: str) -> bool:
    """Whether a store with ``schema`` carries every array ``mode`` reads. ``vol_avoid``'s avoid
    cloud belongs to the query (``screen(..., avoid_points=...)``), not to the store, so its
    per-molecule requirement is just ``atom_pos`` like ``vol``'s."""
    mode = _canon_mode(mode)                           # accept legacy esp / esp_combo
    if mode not in _SPECS:
        return False
    return all(schema.get(f, False) for f in _mode_flags(mode))


#: How each extra per-molecule basis is carried through a shard, as ``(basis, schema flag,
#: offset key, [(profile attr, store key), ...])``; drives extraction, centring, rotation,
#: ``_concat`` and ``_reconstruct``. ``surf`` (fixed width) and ``charges`` are handled inline.
_BASIS_TABLE = (
    ("pharm", "pharm", "pharm_off", (("pharm_types", "pharm_types"), ("pharm_ancs", "pharm_ancs"),
                                     ("pharm_vecs", "pharm_vecs"))),
    ("lipo", "lipophilicity", "lipo_off", (("lipo_pos", "lipo_pos"),
                                           ("lipophilicity", "lipophilicity"))),
    ("fukui", "fukui", "fukui_off", (("fukui_pos", "fukui_pos"), ("fukui", "fukui"))),
    ("mr", "mr", "mr_off", (("mr_pos", "mr_pos"), ("molar_refractivity", "mr"))),
    ("atomtype", "atomtype", "atomtype_off", (("atomtype_pos", "atomtype_pos"),
                                              ("atomic_numbers", "atomic_numbers"))),
)


#: The channel reader for each ``_BASIS_TABLE`` profile attribute, so the store and the aligners
#: share one definition of each center set (the accessor, never ``atom_pos``).
_PROF_READER = {c.prof: c.read for c in _CHANNELS.values() if c.key}


def _profile_from_schema(m, sch: dict, *, id, pre_center: bool, canonical: bool = False) -> "MoleculeProfile":
    """Pull the schema's arrays off a ``Molecule``/``MoleculeProfile`` ``m``, optionally centring
    to the heavy-atom COM and rotating into its principal frame. Returns a ``MoleculeProfile``."""
    atom_pos = _f32(m.atom_pos)
    kw = {"atom_pos": atom_pos}
    if sch["surf"]:
        if m.surf_pos is None:
            raise ValueError("store needs surfaces but molecule has none "
                             "(build Molecule with num_surf_points / surface_points)")
        kw["surf_pos"] = _f32(m.surf_pos)
    if sch["surf_esp"]:
        if m.surf_esp is None:
            raise ValueError("store needs surface ESP but molecule has none")
        kw["surf_esp"] = _f32(m.surf_esp)
    if sch["charges"]:
        if m.partial_charges is None:
            raise ValueError("store needs partial charges but molecule has none")
        if sch["with_H"]:
            kw["partial_charges"] = _f32(m.partial_charges)
            kw["nonH_atoms_idx"] = np.asarray(m._nonH_atoms_idx, dtype=np.int64)
        else:
            # Heavy charges. _nonH_atoms_idx is the real heavy index for a Molecule (full
            # charges) and the identity for a heavy MoleculeProfile.
            kw["partial_charges"] = _f32(np.asarray(m.partial_charges)[m._nonH_atoms_idx])
        # Heavy Gaussian centers for vol_esp, 1:1 with the heavy charges. Kept only when they
        # actually differ from atom_pos (i.e. RemoveHs retained an H).
        hp = np.asarray(_heavy_positions(m), dtype=np.float32)
        if hp.shape != atom_pos.shape or not np.array_equal(hp, atom_pos):
            kw["atom_pos_noH"] = hp
    if sch["radii"]:
        if m.radii is None:
            raise ValueError("store needs vdW radii but molecule has none")
        kw["radii"] = _f32(m.radii)
    if sch["centers_w_H"]:
        kw["centers_w_H"] = _f32(m.mol.GetConformer().GetPositions())
    for _basis, flag, _off, attrs in _BASIS_TABLE:
        if not sch.get(flag, False):
            continue
        for attr, _key in attrs:
            try:
                v = _PROF_READER[attr](m)
            except ValueError as e:
                raise ValueError(f"store needs {flag} but molecule cannot provide it: {e}") from e
            kw[attr] = (np.asarray(v, dtype=np.int32) if attr == "pharm_types" else _f32(v))

    # Profile attributes holding positions (translate and rotate) and directions (rotate only).
    pos_attrs = ["atom_pos", "surf_pos", "centers_w_H", "atom_pos_noH"]
    dir_attrs = []
    for _basis, flag, _off, attrs in _BASIS_TABLE:
        for attr, _key in attrs:
            ch = next((c for c in _CHANNELS.values() if c.prof == attr and c.key), None)
            if ch is None:
                continue
            if ch.kind == "points":
                pos_attrs.append(attr)
            elif ch.kind == "vectors":
                dir_attrs.append(attr)

    if pre_center:
        mu = atom_pos.mean(0)
        for a in pos_attrs:
            v = kw.get(a)
            if v is not None and len(v):
                # Shifted by the atom_pos COM, never by their own, matching the in-memory
                # conformer transform, which moves every channel of a molecule together.
                kw[a] = v - mu

    rot = None
    if canonical:
        # Canonical frame: rotate every position channel and every direction channel into the
        # molecule's principal axes, so the screen can use one constant seed set. ``rot`` is
        # kept to map a returned pose back; requires pre_center (axes are centroid-relative).
        if not pre_center:
            raise ValueError("canonical stores require pre_center=True (axes are centroid-relative)")
        _c = kw["atom_pos"] - kw["atom_pos"].mean(0)
        _cov = _c.T @ _c
        _w, _v = np.linalg.eigh(_cov.astype(np.float64))
        _v = _v[:, ::-1]                                  # descending eigenvalue order
        if np.linalg.det(_v) < 0:                         # keep it a proper rotation
            _v[:, 2] = -_v[:, 2]
        rot = np.ascontiguousarray(_v.T, dtype=np.float32)   # maps original -> canonical
        for a in pos_attrs + dir_attrs:
            v = kw.get(a)
            if v is not None and len(v):
                kw[a] = (v @ rot.T).astype(np.float32)

    return MoleculeProfile(id=id, rot=rot, **kw)


def _id_to_py(x):
    """Numpy scalar -> python int/str for clean Hit ids."""
    if isinstance(x, np.generic):
        return x.item()
    return x


# --------------------------------------------------------------------------- #
# Internal: screen driver -- argument resolution and the single-process shard loop

def _resolve_screen(store, mode, alpha, align_kwargs):
    """Shared validation + alpha resolution for screen()/screen_many()."""
    if mode not in _VALID_MODES:
        raise ValueError(f"unknown mode {mode!r}; valid: {list(_VALID_MODES)}")
    if not store.supports(mode):
        raise ValueError(f"store at {store.path!r} was built for modes {store.modes} "
                         f"and does not support {mode!r}")
    if align_kwargs.get("no_H") is False:
        raise ValueError("screen() aligns heavy atoms only; no_H=False is not supported")
    if alpha is None and mode in _SURF_ALPHA_MODES:
        from shepherd_score.score.constants import ALPHA
        alpha = float(ALPHA(store.num_surf_points))
    if alpha is not None:
        align_kwargs["alpha"] = alpha
    # A parameter the registry declares required (spec default None) must be supplied; raise the
    # same error on the fast and the slow path.
    for _name, _default in _spec_of(mode).params.items():
        if _default is None and _name not in align_kwargs:
            raise ValueError(f"{mode} requires an explicit {_name}=...; pass {_name}=...")
    if _spec_of(mode).name == "vol_avoid" and align_kwargs.get("avoid_points") is None:
        raise ValueError("vol_avoid requires an explicit avoid_points=... (an (K,3) cloud in "
                         "the QUERY's frame to keep library molecules out of)")
    return align_kwargs


def _normalize_scores_out(scores_out, n_queries):
    if scores_out is None:
        return [None] * n_queries
    if isinstance(scores_out, np.ndarray) and n_queries == 1:
        return [scores_out]
    if isinstance(scores_out, (list, tuple)) and len(scores_out) == n_queries:
        return list(scores_out)
    raise ValueError("scores_out must be None, a single array (1 query), or a list "
                     "of one array per query")


def _default_backend() -> str:
    try:
        import torch
        return "triton" if torch.cuda.is_available() else "numba"
    except Exception:
        return "numba"


def _centered_copy(query):
    q = copy.deepcopy(query)
    q.center_to(q.atom_pos.mean(0))
    return q


def _iter_shards_prefetched(store, shard_idxs):
    """Yield ``(shard_meta, arrays)`` for ``shard_idxs`` in order, reading the next shard on one
    background thread so the disk read overlaps the current shard's alignment.

    I/O only: the thread must not touch CUDA, because a stream synchronise from a second thread
    raises while the main thread is capturing a fine-loop graph (see :func:`_to_device`). A
    failed read is re-raised in the caller's thread, in order, and the executor is shut down on
    every exit path. Costs one extra resident shard.
    """
    idxs = list(shard_idxs)
    if len(idxs) < 2:
        for i in idxs:
            yield store.read_shard(i)
        return
    from concurrent.futures import ThreadPoolExecutor
    ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fss-screen-prefetch")
    try:
        fut = ex.submit(store.read_shard, idxs[0])
        for nxt in idxs[1:]:
            cur = fut.result()                  # re-raises a failed read here, in order
            fut = ex.submit(store.read_shard, nxt)
            yield cur
            del cur                             # drop before waiting on the next read
        yield fut.result()
    finally:
        ex.shutdown(wait=True)                  # never leave a reader thread behind


def _canonical_batch_kw(store, qs_ref, mode, device, batch_kw, fast=True):
    """``batch_kw`` with ``const_seeds`` added when the screen can use one shared seed set.

    That needs a canonical store, the array path (only its aligner takes ``const_seeds``), a
    mode in ``_CONST_SEED_MODES`` and a single query, since the seeds depend on the query's
    frame. Otherwise ``batch_kw`` is returned unchanged.
    """
    if not (fast and _use_arrays(mode) and getattr(store, "canonical", False)
            and mode in _CONST_SEED_MODES and len(qs_ref) == 1):
        return batch_kw
    _spec = _spec_of(mode)
    # Resolve the seed channel under this call's keywords: the combo modes seed from the atom
    # clouds only at alpha == 0.81, and the store canonicalises the atom frame alone.
    _seed_ch = _spec.resolve_channel(_spec.seed_channel, batch_kw)
    if _seed_ch not in ("atoms", "heavy"):
        return batch_kw
    _rx = qs_ref[0].get(_seed_ch)
    if _rx is None:
        return batch_kw
    from shepherd_score.accel.drivers._common import canonical_seed_quats
    from .accel._modes import MODE_SEEDS
    batch_kw = dict(batch_kw)
    batch_kw["const_seeds"] = canonical_seed_quats(_rx, len(_rx), int(MODE_SEEDS.get(mode, 10)),
                                                   device)
    return batch_kw


def _canonical_batch_kws(store, qs_ref, mode, device, batch_kw, fast=True):
    """:func:`_canonical_batch_kw` for each query of a panel."""
    return [_canonical_batch_kw(store, [ra], mode, device, batch_kw, fast) for ra in qs_ref]


def _run_shards_inproc(store, shard_idxs, qs_ref, mode, device, top_k, batch_kw,
                       align_kwargs, backend, fast, center_profiles, scores_out, progress,
                       n_total):
    """Read each shard in ``shard_idxs`` once and align every query against it.

    Returns one ``_TopK`` per query. Each shard takes one of three paths:

    * fast, array-native (the usual case): the shard's arrays go straight to the device and
      :func:`_align_fast_arrays`;
    * fast, object path: the same tensors wrapped in per-molecule ``_FastPair`` stand-ins;
    * slow (``fast=False``): full ``MoleculeProfile``/``MoleculePair`` objects through
      ``MoleculePairBatch``.
    """
    heaps = [_TopK(top_k) for _ in qs_ref]
    tf_attr = _TRANSFORM_ATTR[mode]
    # On a canonical store, one constant seed set per query for the whole screen.
    batch_kws = _canonical_batch_kws(store, qs_ref, mode, device, batch_kw, fast)
    done = 0
    if not fast:
        from shepherd_score.container import MoleculePair, MoleculePairBatch
    for sh, arrs in _iter_shards_prefetched(store, shard_idxs):
        # Per-molecule rotation to undo on canonical stores; None otherwise.
        rot = _canonical_rot(store, arrs)
        if fast:
            start = sh["start"]
            if _use_arrays(mode):
                build, align = _array_dispatch(mode)
                ids, fit = build(arrs, device)
                for qi, ra in enumerate(qs_ref):
                    ref = _ref_tensors_from_arrays(ra, mode, device)
                    scores, se3 = align(ref, fit, batch_kws[qi])
                    _accumulate_arrays(heaps[qi], ids, scores, se3,
                                       scores_out, qi, start, rot)
            else:
                ids, pairs = _build_fit_fast_pairs(arrs, mode, device,
                                                   avoid=batch_kw.get("avoid_points"))
                for qi, ra in enumerate(qs_ref):
                    ref = _ref_tensors_from_arrays(ra, mode, device)
                    scores = _align_fast(pairs, ref, mode, batch_kw)
                    _accumulate(heaps[qi], ids, scores, pairs, tf_attr, scores_out, qi,
                                start, rot)
        else:
            profiles = store._reconstruct(arrs, sh)   # == store.read_profiles(idx)
            if center_profiles:
                for p in profiles:
                    p.center_to(p.atom_pos.mean(0))
            ids = [_id_to_py(p.id) for p in profiles]
            start = sh["start"]
            for qi, q in enumerate(qs_ref):
                pairs = [MoleculePair(q, p, do_center=False) for p in profiles]
                result = MoleculePairBatch(pairs)._align_mode(
                    mode, backend=backend, **{"return_aligned": False, **align_kwargs})
                scores = np.asarray(result[0], dtype=float)
                _accumulate(heaps[qi], ids, scores, pairs, tf_attr, scores_out, qi,
                            start, rot)
        done += sh["n"]
        if progress:
            print(f"[screen] {done}/{n_total} library molecules aligned "
                  f"x {len(qs_ref)} queries", flush=True)
    return heaps


# --------------------------------------------------------------------------- #
# Internal: fast engine -- store arrays straight to device tensors
# On a pre-centered store with trans_init=False and a non-jax backend, a shard's arrays are
# uploaded once and aligned against every query without building per-molecule objects.
# --------------------------------------------------------------------------- #
#: Modes the fast engine serves. Currently every registry mode; the fast path additionally
#: needs a pre-centered store, ``trans_init=False`` and a non-jax backend.
_FAST_MODES = tuple(_SPECS)


#: Modes taking the array-native fast path rather than the ``_FastPair`` object path.
#: Currently every registry mode.
_ARRAY_MODES = tuple(_SPECS)


def _use_arrays(mode: str) -> bool:
    """Whether to take the array-native screen path (``_arrays.ENABLED`` is the switch)."""
    from shepherd_score.accel.batch import _arrays
    return mode in _ARRAY_MODES and _arrays.ENABLED


#: Persistent pinned staging buffers for the shard upload, keyed by (dtype, device index) and
#: grown to the largest shard seen; each carries the event that marks its last copy-out done.
_PIN_STAGE: dict = {}


def _to_device(a, device, *, dtype=None):
    """Upload one store array asynchronously, staged through pinned host memory, and retype on
    the device (a float16 store widens to float32 on the GPU, which is exact and halves the bytes
    pushed). Runs on the calling thread: a stream synchronise from a second thread raises while
    the main thread is capturing a CUDA graph, so the read-ahead thread stays I/O-only."""
    import torch
    if device.type != "cuda":
        t = torch.as_tensor(a, device=device)
        return t if dtype is None or t.dtype == dtype else t.to(dtype)
    src = torch.from_numpy(np.asarray(a))
    # Key on the RESOLVED device index: torch.device("cuda") carries index None, and two
    # devices must not share one event (pinned host memory is shareable, a CUDA event is not).
    _idx = getattr(device, "index", None)
    key = (src.dtype, torch.cuda.current_device() if _idx is None else int(_idx))
    ent = _PIN_STAGE.get(key)
    if ent is None or ent[0].numel() < src.numel():
        ent = (torch.empty(src.numel(), dtype=src.dtype, pin_memory=True), torch.cuda.Event())
        _PIN_STAGE[key] = ent
    else:
        ent[1].synchronize()          # the previous copy out of this buffer has landed
    buf, ev = ent
    flat = buf[:src.numel()]
    flat.copy_(src.reshape(-1))
    out = torch.empty(tuple(src.shape), dtype=src.dtype, device=device)
    out.view(-1).copy_(flat, non_blocking=True)
    ev.record()
    return out if dtype is None or out.dtype == dtype else out.to(dtype)


def _query_ref_arrays(q, mode: str) -> dict:
    """The (centered) query's numpy arrays for ``mode``, keyed by channel; plain numpy so they are
    cheap to ship to the multi-GPU workers. The channel readers treat a ``Molecule`` and a
    ``MoleculeProfile`` identically and raise a ValueError naming any missing field."""
    spec = _spec_of(mode)
    out = {}
    for name in spec.all_channels():
        ch = _CHANNELS[name]
        if ch.is_pair:
            continue                                  # supplied by the caller (avoid_points)
        out[name] = np.ascontiguousarray(
            np.asarray(ch.read(q), dtype=np.int64 if ch.dtype == "int64" else np.float32))
    return out


def _ref_tensors_from_arrays(ra: dict, mode: str, device) -> dict:
    """The query's device tensors, keyed by the pair attribute the aligners read
    (``_ref_<attr>_t``). Uploaded through the same pinned staging as the fit side."""
    import torch
    out = {}
    for name, arr in ra.items():
        ch = _CHANNELS[name]
        dt = torch.int64 if ch.dtype == "int64" else torch.float32
        out[ch.ref_attr] = _to_device(arr, device, dtype=dt)
    return out


def _ref_channel_tensors(ref_tensors: dict, mode: str) -> dict:
    """Re-key ``_ref_<attr>_t`` tensors by channel, which is what the array aligner takes."""
    spec = _spec_of(mode)
    out = {}
    for name in spec.all_channels():
        ch = _CHANNELS[name]
        if ch.is_pair or ch.ref_attr not in ref_tensors:
            continue
        out[name] = ref_tensors[ch.ref_attr]
    return out


def _fast_batch_kwargs(mode: str, ak: dict) -> dict:
    """Translate ``screen()``'s align_kwargs into the ``_align_batch_<mode>`` keywords. The mode's
    parameters and defaults come from the registry, the fine-step count from ``_steps_for`` and
    ``lr`` from the spec's ``screen_lr``."""
    from shepherd_score.accel.batch.aligners import _steps_for, _seeds_for
    spec = _spec_of(mode)
    kw = {"steps_fine": ak.get("max_num_steps", _steps_for(mode)), "trans_init": False}
    for name, default in spec.params.items():
        if name == "lr":
            kw["lr"] = ak.get("lr", spec.screen_lr)
        elif default is None:
            if name not in ak:
                raise ValueError(f"{mode} requires an explicit {name}=...; pass {name}=...")
            kw[name] = ak[name]
        else:
            kw[name] = ak.get(name, default)
    kw["num_repeats"] = ak.get("num_repeats", _seeds_for(mode))
    return kw


def _with_avoid(batch_kw: dict, ak: dict, device) -> dict:
    """Attach the query-side avoid cloud for ``vol_avoid`` (shape Tanimoto minus an excluded-volume
    penalty against a cloud that belongs to the query, e.g. a pocket wall). It arrives as
    ``screen(..., avoid_points=...)`` and is uploaded once here, like the query's own channels."""
    cloud = ak.get("avoid_points")
    if cloud is None:
        return batch_kw
    import torch
    batch_kw = dict(batch_kw)
    batch_kw["avoid_points"] = np.ascontiguousarray(np.asarray(cloud, dtype=np.float32))
    batch_kw["avoid_points_t"] = _to_device(batch_kw["avoid_points"], device,
                                            dtype=torch.float32)
    return batch_kw


def _build_fit_arrays(arrs: dict, mode: str, device):
    """Array-native twin of :func:`_build_fit_fast_pairs`: ``(ids, {channel: (flat, off)})``.
    Each contiguous store buffer is uploaded once with its CSR offsets (``None`` for the dense
    fixed-width surface). Which store keys feed a channel is ``channels.load_store_channel``'s
    business, since the heavy basis and the heavy charges depend on the store layout."""
    import torch
    from shepherd_score.accel.channels import load_store_channel
    spec = _spec_of(mode)
    fit = {}
    for name in spec.all_channels():
        ch = _CHANNELS[name]
        if ch.is_pair:
            continue
        flat, off = load_store_channel(arrs, name)
        dt = torch.int64 if ch.dtype == "int64" else torch.float32
        fit[name] = (_to_device(flat, device, dtype=dt),
                     None if off is None else _to_device(off, device, dtype=torch.long))
    return arrs["ids"], fit


def _align_fast_arrays(ref: dict, fit: dict, mode: str, batch_kw: dict):
    """Array-native twin of :func:`_align_fast`: ``(scores, SE3)`` for one shard, one query. The
    mode parameters resolve through the same ``resolve_params`` as the pairwise aligner, which is
    where the surface modes' ``lam`` is scaled by ``LAM_SCALING`` exactly once."""
    from shepherd_score.accel.batch._arrays import align_arrays
    from shepherd_score.accel.batch.aligners import resolve_params
    spec = _spec_of(mode)
    params = resolve_params(spec, batch_kw, f"screen({mode})")
    return align_arrays(mode, _ref_channel_tensors(ref, mode), fit, params=params,
                        steps_fine=int(batch_kw["steps_fine"]),
                        num_seeds=batch_kw.get("num_repeats"),
                        const_seeds=batch_kw.get("const_seeds"),
                        avoid=batch_kw.get("avoid_points_t"))


def _make_array_builder(mode: str):
    def build(arrs, device, _m=mode):
        return _build_fit_arrays(arrs, _m, device)
    build.__name__ = build.__qualname__ = f"_build_fit_arrays_{mode}"
    build.__doc__ = (f"Array-native fit builder for ``{mode}``: one name over "
                     ":func:`_build_fit_arrays`, so a parity test can wrap this mode's builder "
                     "alone and assert WHICH one ran.")
    return build


def _make_array_aligner(mode: str):
    def align(ref, fit, batch_kw, _m=mode):
        return _align_fast_arrays(ref, fit, _m, batch_kw)
    align.__name__ = align.__qualname__ = f"_align_fast_arrays_{mode}"
    align.__doc__ = f"Array-native aligner for ``{mode}``; see :func:`_align_fast_arrays`."
    return align


#: mode -> (fit-array builder, array-native aligner): one generic body per side but a distinct
#: named function per mode, so a test can wrap one mode's builder and assert which one ran.
_ARRAY_BUILDERS = {m: _make_array_builder(m) for m in _ARRAY_MODES}


_ARRAY_ALIGNERS = {m: _make_array_aligner(m) for m in _ARRAY_MODES}


for _m, _fn in _ARRAY_BUILDERS.items():
    globals()[_fn.__name__] = _fn


for _m, _fn in _ARRAY_ALIGNERS.items():
    globals()[_fn.__name__] = _fn


del _m, _fn


def _array_dispatch(mode: str):
    """The ``(builder, aligner)`` pair for ``mode``: the single selector shared by
    :func:`_run_shards_inproc` and :func:`_screen_worker`. It reads the tables at call time so
    a test can patch ``_ARRAY_BUILDERS``."""
    return _ARRAY_BUILDERS[mode], _ARRAY_ALIGNERS[mode]


# --------------------------------------------------------------------------- #
# Internal: object path -- per-molecule _FastPair stand-ins
# Kept as the reference the array path is tested against.
# --------------------------------------------------------------------------- #
class _FastPair:
    """Cached-tensor-only stand-in a batched aligner can consume. ``__slots__`` is derived from
    the channel table (``_ref_<attr>_t`` / ``_fit_<attr>_t``) plus every mode's result pair."""

    __slots__ = (("device", "ref_molec", "fit_molec")
                 + tuple(dict.fromkeys(
                     [a for c in _CHANNELS.values() for a in (c.ref_attr, c.fit_attr)]))
                 + tuple(a for pair in _MODE_ATTRS.values() for a in pair))

    def __init__(self, device):
        self.device = device


def _build_fit_fast_pairs(arrs: dict, mode: str, device, avoid=None):
    """Load one shard's fit arrays as device tensors once; return ``(ids, [_FastPair])`` whose
    fit tensors are per-molecule views (one ``torch.split`` or ``unbind`` per channel). This is
    the object path, kept wired for every mode as the reference the array path is compared to."""
    import torch
    from shepherd_score.accel.channels import load_store_channel
    ids = arrs["ids"]
    K = len(ids)
    pairs = [_FastPair(device) for _ in range(K)]
    spec = _spec_of(mode)
    for name in spec.all_channels():
        ch = _CHANNELS[name]
        if ch.is_pair:
            if avoid is not None:
                t = _to_device(np.ascontiguousarray(np.asarray(avoid, np.float32)), device,
                               dtype=torch.float32)
                for p in pairs:
                    setattr(p, ch.ref_attr, t)
            continue
        flat, off = load_store_channel(arrs, name)
        dt = torch.int64 if ch.dtype == "int64" else torch.float32
        big = _to_device(flat, device, dtype=dt)
        views = (torch.unbind(big) if off is None
                 else torch.split(big, np.diff(off).tolist()))
        for p, v in zip(pairs, views):
            setattr(p, ch.fit_attr, v)
    return ids, pairs


def _align_fast(pairs, ref_tensors: dict, mode: str, batch_kw: dict):
    """Set the shared query ref tensors on the resident fit pairs, run the batched aligner and
    return the per-pair scores. Transforms are materialised lazily for top-K survivors only
    (``_TopK.offer_pair``), since a screen keeps only ~top_k of K."""
    from shepherd_score.accel.batch import aligners
    items = tuple(ref_tensors.items())          # materialise the view once, not per pair
    for p in pairs:
        for k, v in items:
            setattr(p, k, v)
    getattr(aligners, "_align_batch_" + mode)(pairs, **batch_kw)
    # The aligners write plain Python floats, so ``fromiter`` over ``attrgetter`` is the same
    # float64 vector as a list comprehension without K Python-level calls.
    return np.fromiter(map(_SCORE_GETTER[mode], pairs), dtype=float, count=len(pairs))


# --------------------------------------------------------------------------- #
# Internal: top-K reduction
# --------------------------------------------------------------------------- #
class _TopK:
    """Bounded max-list keyed on score (min-heap of size k). Tie-break by a counter
    so transforms are never compared."""
    __slots__ = ("k", "heap", "_c")

    def __init__(self, k):
        self.k = k
        self.heap = []
        self._c = 0

    def _push(self, score, id_, transform):
        self._c += 1
        item = (score, self._c, id_, transform)
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, item)
        elif score > self.heap[0][0]:
            heapq.heapreplace(self.heap, item)

    def offer_pair(self, score, id_, pair, tf_attr, rot=None):
        """Offer a candidate, materialising its transform from ``pair`` only if the score makes
        the top-K. Must be called while ``pair`` still holds this query's pose. The canonical-frame
        composition is deferred to :meth:`_materialize`, since most accepted candidates are evicted."""
        if len(self.heap) < self.k or score > self.heap[0][0]:
            self._push(score, id_, (_transform_of(pair, tf_attr), rot))

    def offer_row(self, score, id_, transforms, i, rot=None):
        """Array-native twin of :meth:`offer_pair`: the transform is row ``i`` of a ``(K,4,4)``
        array. Same acceptance test and tie-break, so the heap matches the object path's down to
        ties. Composition with ``rot`` is deferred to :meth:`_materialize`; the rows are views."""
        if len(self.heap) < self.k or score > self.heap[0][0]:
            self._push(score, id_, (transforms[i], None if rot is None else rot[i]))

    @staticmethod
    def _materialize(t):
        """Compose a deferred ``(transform, rot)`` pair; pass anything else through.

        ``offer_pair`` (the object path) pushes a real array, and ``merge_raw`` receives
        already-composed transforms from another worker, so both stay untouched."""
        return _compose_rot(t[0], t[1]) if type(t) is tuple else t

    def threshold(self):
        """Score a candidate must strictly exceed to change this heap, or ``-inf`` until it fills.

        The pre-filter in :func:`_accumulate` is exact because this is monotone non-decreasing
        once the heap is full (``_push`` only replaces the minimum with a strictly larger score)
        and a rejected offer mutates nothing.
        """
        # The ``self.k`` guard keeps a degenerate k=0 heap failing inside offer_pair, not here.
        return self.heap[0][0] if (self.k and len(self.heap) >= self.k) else float("-inf")

    def merge_raw(self, raw):
        for (s, i, t) in raw:
            self._push(s, i, t)

    def raw(self):
        # Composed on the way out, so what crosses a process boundary is a plain (4,4) array,
        # never a view that would drag its whole shard through pickle.
        return [(s, i, self._materialize(t)) for (s, _, i, t) in self.heap]

    def sorted(self):
        return [Hit(score=s, id=i, transform=self._materialize(t))
                for (s, _, i, t) in sorted(self.heap, key=lambda x: x[0], reverse=True)]


# Block size for the top-K pre-filter; the threshold is refreshed between blocks. It affects
# speed only, never the result.
_ACCUM_BLOCK = 4096


def _accumulate(heap, ids, scores, pairs, tf_attr, scores_out, qi, start, rot=None):
    """Fold one shard's scores for one query into ``heap`` (and ``scores_out``, if given).

    Scores are filtered in numpy against ``heap.threshold()`` first, so only candidates that
    can enter the heap reach Python. The result is identical to offering every candidate,
    including tie order (see :meth:`_TopK.threshold`).
    """
    n = len(ids)
    lo = 0
    while lo < n:
        hi = lo + _ACCUM_BLOCK
        if hi > n:
            hi = n
        thr = heap.threshold()
        if thr == float("-inf"):
            sel = np.arange(lo, hi)                  # heap not full: every offer is taken
        else:
            sel = np.flatnonzero(scores[lo:hi] > thr) + lo
        # One vectorised conversion per block: ``ndarray.tolist()`` is elementwise ``.item()``,
        # which is what ``_id_to_py`` does for a numpy scalar, so the heap sees the same values.
        cs = scores[sel].tolist()
        cid = np.asarray(ids)[sel].tolist()
        for j, i in enumerate(sel.tolist()):
            heap.offer_pair(cs[j], cid[j], pairs[i], tf_attr,
                            None if rot is None else rot[i])
        lo = hi
    if scores_out is not None and scores_out[qi] is not None:
        scores_out[qi][start:start + n] = scores


def _accumulate_arrays(heap, ids, scores, transforms, scores_out, qi, start, rot=None):
    """Array-native twin of :func:`_accumulate`: the same block reduce, except the survivor's
    transform is row ``i`` of the ``(K,4,4)`` array rather than a pair attribute."""
    n = len(ids)
    lo = 0
    while lo < n:
        hi = lo + _ACCUM_BLOCK
        if hi > n:
            hi = n
        thr = heap.threshold()
        if thr == float("-inf"):
            sel = np.arange(lo, hi)
        else:
            sel = np.flatnonzero(scores[lo:hi] > thr) + lo
        # One vectorised conversion per block: ``ndarray.tolist()`` is elementwise ``.item()``,
        # which is what ``_id_to_py`` does for a numpy scalar, so the heap sees the same values.
        cs = scores[sel].tolist()
        cid = ids[sel].tolist()
        for j, i in enumerate(sel.tolist()):
            heap.offer_row(cs[j], cid[j], transforms, i, rot)
        lo = hi
    if scores_out is not None and scores_out[qi] is not None:
        scores_out[qi][start:start + n] = scores


def _transform_of(pair, tf_attr):
    t = getattr(pair, tf_attr, None)
    if t is None:
        return None
    try:
        import torch
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(t)


def _canonical_rot(store, arrs):
    """The per-molecule rotation a canonical store applied at build time, or ``None`` for a legacy
    or ``canonical=False`` store (which makes every composition below a no-op)."""
    if not getattr(store, "canonical", False):
        return None
    return arrs.get("rot") if hasattr(arrs, "get") else None


def _compose_rot(T, R):
    """Map a pose solved in the canonical frame back into the molecule's own frame.

    A canonical store holds ``x_canon = (x_orig - mu) @ R.T``, so a transform ``T`` solved
    against it gives ``x_aligned = (x_orig - mu) @ (T_R @ R).T + T_t``: the rotation composes
    as ``T_R @ R`` and the translation is unchanged (convention ``points @ R.T + t``, as in
    ``alignment/utils/se3.py``). Called per survivor rather than per molecule.
    """
    if R is None or T is None:
        return T
    T = np.array(T, copy=True)
    T[:3, :3] = T[:3, :3] @ np.asarray(R, dtype=T.dtype)
    return T


# --------------------------------------------------------------------------- #
# Internal: multi-GPU worker pool
# One persistent worker process per GPU, spawned on first use and kept for the life of the
# calling process. Each worker streams a static, interleaved share of the shards
# (r, r+ndev, r+2*ndev, ...) with the same read-ahead as the single-process screen.
# --------------------------------------------------------------------------- #
_MGPU_POOL = None                        # {"key": (ndev, threads), "procs", "job_qs", "conns"}


def _screen_worker(rank, threads, store_path, ref_arrays_list, mode, batch_kw, top_k,
                   shard_q, out_conn):
    """One device's share of one screen, inside its worker process. ``shard_q`` is either a
    static list of shard indices (streamed with read-ahead) or a queue yielding indices then
    ``None`` (serial reads). The reply goes to ``out_conn`` (this worker's result pipe) as
    ``(rank, per_query_raw_heaps)`` or ``(rank, "__ERR__", traceback)``."""
    try:
        import torch
        from shepherd_score.accel.multi_gpu import _cap_threads
        _cap_threads(threads)
        torch.cuda.set_device(rank)
        dev = torch.device("cuda", rank)
        store = ProfileStore.open(store_path)
        ref_tensors = [_ref_tensors_from_arrays(ra, mode, dev) for ra in ref_arrays_list]
        heaps = [_TopK(top_k) for _ in ref_arrays_list]
        tf_attr = _TRANSFORM_ATTR[mode]
        # The canonical store's constant seeds, one set per query, as the in-process loop sets them.
        batch_kws = _canonical_batch_kws(store, ref_arrays_list, mode, dev, batch_kw)

        def _drain(q):                                     # the queue form: serial reads
            while True:
                idx = q.get()
                if idx is None:
                    return
                yield store.read_shard(idx)

        if isinstance(shard_q, (list, tuple, range)):
            share = list(shard_q)
            shards = _iter_shards_prefetched(store, share) if share else iter(())
        else:
            shards = _drain(shard_q)
        for _sh, arrs in shards:
            rot = _canonical_rot(store, arrs)      # canonical-frame stores; None otherwise
            if _use_arrays(mode):
                # The same selector the in-process driver uses; see _array_dispatch.
                build, align = _array_dispatch(mode)
                ids, fit = build(arrs, dev)
                for qi, ref in enumerate(ref_tensors):
                    scores, se3 = align(ref, fit, batch_kws[qi])
                    _accumulate_arrays(heaps[qi], ids, scores, se3, None, qi, 0, rot)
                torch.cuda.synchronize()
                continue
            ids, pairs = _build_fit_fast_pairs(arrs, mode, dev,
                                               avoid=batch_kw.get("avoid_points"))
            for qi, ref in enumerate(ref_tensors):
                scores = _align_fast(pairs, ref, mode, batch_kw)
                # Same pre-filtered reduce as the in-process driver; scores_out is unsupported here.
                _accumulate(heaps[qi], ids, scores, pairs, tf_attr, None, qi, 0, rot)
            torch.cuda.synchronize()
        out_conn.send((rank, [h.raw() for h in heaps]))
    except Exception:                            # noqa: BLE001 - relayed to parent
        import traceback
        out_conn.send((rank, "__ERR__", traceback.format_exc()))


def _mgpu_pool_worker(rank, threads, job_q, out_conn):
    """The persistent per-device process: one CUDA device, many screens. A job is
    ``(store_path, ref_arrays_list, mode, batch_kw, top_k, shard_list)``; ``None`` ends it."""
    while True:
        job = job_q.get()
        if job is None:
            return
        store_path, ref_arrays_list, mode, batch_kw, top_k, shards = job
        _screen_worker(rank, threads, store_path, ref_arrays_list, mode, batch_kw, top_k,
                       shards, out_conn)


def _mgpu_pool(ndev, threads):
    """The pool for ``(ndev, threads)``: reused while alive, (re)spawned otherwise, so the CUDA
    process spawn is paid once per process lifetime."""
    global _MGPU_POOL
    pool = _MGPU_POOL
    if pool is not None and pool["key"] == (ndev, threads) and all(p.is_alive() for p in pool["procs"]):
        return pool
    close_multigpu_pool()
    import torch.multiprocessing as mp
    from shepherd_score.accel._procs import start_worker
    ctx = mp.get_context("spawn")
    pool = {"key": (ndev, threads), "procs": [], "job_qs": [], "conns": []}
    try:
        for r in range(ndev):
            q, conn, p = start_worker(ctx, _mgpu_pool_worker, (r, threads), daemon=True)
            pool["job_qs"].append(q)
            pool["conns"].append(conn)
            pool["procs"].append(p)
    except BaseException:
        _stop_mgpu_pool(pool)                        # the workers that did start
        raise
    _MGPU_POOL = pool
    return pool


def _screen_many_multigpu(qs, store_path, mode, ndev, batch_kw, top_k, progress,
                         worker_threads=None):
    import os as _os
    import torch
    from shepherd_score.accel._procs import collect

    ndev = max(1, min(ndev, torch.cuda.device_count() if torch.cuda.is_available() else 1))
    try:
        cores = len(_os.sched_getaffinity(0))
    except AttributeError:
        cores = _os.cpu_count() or ndev
    if worker_threads is not None and (not isinstance(worker_threads, int) or worker_threads < 1):
        raise ValueError('worker_threads must be a positive integer')
    threads = worker_threads if worker_threads is not None else max(1, cores // ndev)
    ref_arrays_list = [_query_ref_arrays(q, mode) for q in qs]

    store = ProfileStore.open(store_path)
    n_shards = store.num_shards

    # The thread caps are read by the workers at spawn; set them around the spawn only.
    _saved = {k: _os.environ.get(k) for k in
              ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    for k in _saved:
        _os.environ[k] = str(threads)
    try:
        pool = _mgpu_pool(ndev, threads)
    finally:
        for k, v in _saved.items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v

    # Static, interleaved shares: shards are equal-sized except the last, so this is balanced
    # without a shared queue, and a static share is what lets a worker read ahead.
    def submit():
        for r, q in enumerate(pool["job_qs"]):
            q.put((store_path, ref_arrays_list, mode, batch_kw, top_k,
                   list(range(r, n_shards, ndev))))

    # Any failure closes the pool at once: the other ranks' work is wasted anyway, and a CUDA
    # error can leave a worker alive with a broken context.
    replies = collect(dict(enumerate(zip(pool["conns"], pool["procs"]))), close_multigpu_pool,
                      submit=submit, what="multi-GPU screen worker", fail_fast=True,
                      error_of=lambda m: m[2] if len(m) == 3 and m[1] == "__ERR__" else None)
    heaps = [_TopK(top_k) for _ in qs]
    for rank in sorted(replies):                     # rank order, so equal scores merge repeatably
        per_query = replies[rank][1]
        for qi, raw in enumerate(per_query):
            heaps[qi].merge_raw(raw)
    return heaps


atexit.register(close_multigpu_pool)
