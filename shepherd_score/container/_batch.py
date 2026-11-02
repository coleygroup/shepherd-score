"""Batched molecular alignment with Numba, Triton, and optional JAX backends."""
import warnings
from contextlib import contextmanager
from importlib.metadata import version as _pkg_version
from typing import List, Optional, Tuple

import numpy as np

from shepherd_score.container._core import MoleculePair
from shepherd_score.modes import ALIASES as MODE_ALIASES, get_mode
from shepherd_score.score.similarity import Similarity, tversky_weights
from shepherd_score.container._batch_utils import (
    _pad_arrays,
    _dispatch_parallel,
    _align_vol_shmap,
    _align_vol_esp_shmap,
    _align_surf_shmap,
    _align_esp_shmap,
    _align_pharm_shmap,
    _align_vol_worker,
    _align_vol_esp_worker,
    _align_surf_worker,
    _align_esp_worker,
    _align_pharm_worker,
)

# Backend names that run the batched accel engine: Triton on CUDA, numba on CPU.
_ACCEL_BACKENDS = ("triton", "cuda", "gpu", "numba", "cpu")
_NUMBA_BACKENDS = ("numba", "cpu")

# Optimizer defaults of the JAX and per-pair paths, as in ``MoleculePair``. The accel backends
# use the per-mode registry defaults in ``accel/_modes.py`` instead.
_REFERENCE_DEFAULTS = {"num_repeats": 50, "lr": 0.1, "max_num_steps": 200}


def _resolve_backend(backend):
    """Resolve ``backend=None`` to ``screen._default_backend()`` (``triton`` when CUDA is
    available, else ``numba``); explicit values pass through. Imported lazily to avoid a
    circular import."""
    if backend is not None:
        return backend
    from shepherd_score.screen import _default_backend
    return _default_backend()


def _reference_opt(num_repeats, lr, max_num_steps):
    """``num_repeats``, ``lr`` and ``max_num_steps`` with ``None`` replaced by the reference
    defaults (50 / 0.1 / 200)."""
    d = _REFERENCE_DEFAULTS
    return (d["num_repeats"] if num_repeats is None else num_repeats,
            d["lr"] if lr is None else lr,
            d["max_num_steps"] if max_num_steps is None else max_num_steps)


def _compute_bucket_splits(sizes_a, sizes_b, num_buckets):
    """Sort pairs by (max(a,b), min(a,b)) and split into buckets.

    Parameters
    ----------
    sizes_a, sizes_b : array-like of int
        Per-pair sizes (e.g. atom counts) for the two molecules.
    num_buckets : int
        Number of buckets.  ``<= 1`` returns a single bucket with all
        pairs in their original order (no sorting).

    Returns
    -------
    list of list of int
        Each inner list is a bucket of global pair indices.
    """
    n = len(sizes_a)
    if num_buckets <= 1:
        return [list(range(n))]
    sizes_a = np.asarray(sizes_a)
    sizes_b = np.asarray(sizes_b)
    sort_keys = np.array([np.minimum(sizes_a, sizes_b),
                           np.maximum(sizes_a, sizes_b)])
    sorted_order = np.lexsort(sort_keys)
    num_buckets_actual = min(num_buckets, n)
    return [
        arr.tolist()
        for arr in np.array_split(sorted_order, num_buckets_actual)
        if len(arr) > 0
    ]


@contextmanager
def _pinned_torch_threads(pairs):
    """Pin torch to one intra-op thread for the duration of a CPU batch alignment.

    The numba kernels own the cores on this path (``parallel=True`` over ``NUMBA_NUM_THREADS``)
    and torch only runs the small tails around them; left unpinned the two pools oversubscribe
    and the tails spin-wait against the kernels. Scoped rather than set once at import because
    ``torch.set_num_threads`` is process-global and would silently reconfigure the caller's
    torch. Restored on every exit path. A no-op on CUDA.
    """
    import torch
    if not pairs or pairs[0].device.type != "cpu":
        yield
        return
    prev = torch.get_num_threads()
    if prev == 1:
        yield                                            # already pinned; do not touch it
        return
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(prev)


class MoleculePairBatch:
    """Align molecular pairs in batches and store scores and transforms on each pair.

    Every ``align_with_<mode>`` method writes its results to the same ``transform_*`` /
    ``sim_aligned_*`` slots on each pair as ``MoleculePair.align_with_<mode>``, and they share
    these keywords:

    backend : str, optional
        ``None`` (default) resolves to ``"triton"`` on CUDA, else ``"numba"``.
        ``"triton"`` (aliases ``"cuda"``/``"gpu"``) and ``"numba"`` (alias ``"cpu"``) run the
        batched accel engine (heavy atoms only). ``"jax"`` runs the padded JAX path for ``vol``,
        ``vol_esp``, ``surf``, ``surf_esp`` and ``pharm``, and each pair's ``MoleculePair.align``
        for the other modes (with a warning when the mode has no JAX optimizer). ``"torch"``
        runs each pair's ``MoleculePair.align`` with PyTorch.
    num_repeats, lr, max_num_steps : optional
        SE(3) seeds, Adam learning rate and maximum steps. ``None`` (default) takes the per-mode
        registry defaults in ``accel/_modes.py`` on the accel backends (``lr`` from the mode's
        ``screen_lr``, as ``screen`` does), and 50 / 0.1 / 200 on ``"jax"`` and ``"torch"``.
    trans_init : bool
        Also seed by translating the fit to each reference atom (or pharmacophore).
    num_workers : int
        On the numba backend, shard the pairs across this many single-threaded worker
        processes (``accel.cpu_pool``). Sharding changes early stopping (a shard runs until all
        of its pairs converge), so results may differ slightly from a single process.
    similarity : str or (float, float)
        ``'tanimoto'`` (default), ``'tversky'``, ``'tversky_ref'``, ``'tversky_fit'`` or explicit
        Tversky weights ``(a, b)``. A Tversky similarity runs the registry's
        ``<mode>_tversky`` mode and stores results in the ``<mode>_tversky`` slots; its scores
        are not clipped to [0, 1].
    return_aligned : bool
        Also return the aligned fit coordinates (default False); ``None`` entries when False.
    verbose : bool
        Print per-pair progress on the JAX and per-pair paths; ignored by triton/numba.

    Each method returns ``(scores, aligned)``, or ``(scores, anchors, vectors)`` for ``pharm``.
    """

    def __init__(self, pairs: List[MoleculePair]):
        self.pairs = pairs

    def align(self, mode: str, similarity: Similarity = 'tanimoto', **kwargs):
        """Align all pairs under the named ``mode``, as ``MoleculePair.align`` names them.

        Calls ``align_with_<mode>``, so results land in that method's slots and ``kwargs`` are
        that method's keywords::

            batch.align('vol', similarity='tversky', backend='triton')

        A registry ``<base>_tversky`` name (e.g. ``'vol_tversky'``, ``'pharm_tversky'``) runs
        that registry mode directly and takes its keywords, ``tversky_alpha`` / ``tversky_beta``::

            batch.align('vol_tversky', tversky_alpha=0.7, tversky_beta=0.3)
        """
        name = MODE_ALIASES.get(mode, mode)

        # Registry Tversky names bypass align_with_* and go straight to the registry _align_mode
        if name.endswith('_tversky'):
            if similarity != 'tanimoto':
                raise ValueError(f'{mode!r} is already a Tversky mode; pass similarity= to '
                                 f"align({name[:-len('_tversky')]!r}) instead.")
            return self._align_mode(name, **kwargs)

        method = getattr(self, f'align_with_{name}', None)
        if method is None:
            raise ValueError(f'No batched implementation of {mode!r}.')

        if similarity != 'tanimoto':
            kwargs['similarity'] = similarity
        return method(**kwargs)

    def _align_mode(self, mode: str, *, backend: Optional[str] = None,
                    num_repeats: Optional[int] = None, lr: Optional[float] = None,
                    max_num_steps: Optional[int] = None, trans_init: bool = False,
                    num_workers: int = 1, verbose: bool = False, return_aligned: bool = False,
                    **params):
        """Align all pairs under the accel registry mode ``mode`` (an ``accel/_modes.py`` name,
        e.g. ``'vol'`` or ``'vol_tversky'``) and store the results in its slots.

        ``params`` are the registry mode's parameters (e.g. ``alpha``, ``lam``,
        ``tversky_alpha``), plus ``no_H`` and, for ``vol_avoid``, ``avoid_points``. The other
        keywords are as in the class docstring. ``backend="jax"`` here always runs each pair's
        ``MoleculePair.align``; the padded JAX paths live in the ``align_with_*`` methods.
        """
        from shepherd_score.accel._modes import SPECS, canonical

        name = canonical(mode)
        if name not in SPECS:
            raise ValueError(f'No batched implementation of {mode!r}.')
        spec = SPECS[name]
        backend = _resolve_backend(backend)

        if name == 'vol_avoid':
            self._attach_avoid_points(params.pop('avoid_points', None))

        # Batched Triton / numba alignment
        if backend in _ACCEL_BACKENDS:
            if not params.pop('no_H', True):
                raise NotImplementedError(
                    f"the Triton/numba {name} backend aligns heavy atoms only (no_H=True)")
            if backend in _NUMBA_BACKENDS:
                self._prepare_numba()

            # None: the registry seeds and steps, and the mode's screen_lr (the lr main used)
            kw = dict(params, trans_init=trans_init, lr=spec.screen_lr if lr is None else lr)
            for key, value in (('num_repeats', num_repeats), ('steps_fine', max_num_steps)):
                if value is not None:
                    kw[key] = value

            self._run_accel(name, kw, num_workers)
            aligned = self._aligned_fits(spec) if return_aligned else None

        # Per-pair MoleculePair.align otherwise
        elif backend in ('jax', 'torch'):
            num_repeats, lr, max_num_steps = _reference_opt(num_repeats, lr, max_num_steps)
            aligned = self._align_per_pair(
                spec, backend, num_repeats=num_repeats, lr=lr, max_num_steps=max_num_steps,
                trans_init=trans_init, verbose=verbose, **params)
            if not return_aligned:
                aligned = None

        else:
            raise ValueError(
                f"unknown backend {backend!r}; use 'triton', 'numba', 'jax' or 'torch'")

        # Scores from MoleculePair attributes
        score_attr = spec.score_attr
        scores = np.array([float(getattr(p, score_attr)) for p in self.pairs])
        n = len(self.pairs)

        if spec.pharm_style:
            anchors, vectors = aligned if aligned is not None else ([None] * n, [None] * n)
            return scores, anchors, vectors
        return scores, aligned[0] if aligned is not None else [None] * n

    def _align_with(self, mode: str, similarity: Similarity, **kwargs):
        """Run ``mode``, or its registry ``<mode>_tversky`` mode for a Tversky ``similarity``."""
        weights = tversky_weights(similarity)

        if weights is not None:
            mode = f'{mode}_tversky'
            kwargs.update(tversky_alpha=weights[0], tversky_beta=weights[1])

        return self._align_mode(mode, **kwargs)

    def _prepare_numba(self):
        """Move every pair onto CPU for a ``backend="numba"`` call. Kernel selection is
        device-driven (:mod:`shepherd_score.accel.kernels.dispatch`), so CPU tensors run the
        numba kernels even in a process that has the Triton kernels loaded.
        """
        import torch
        try:
            import numba  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "backend='numba' needs the 'numba' package, a core dependency "
                "of shepherd-score; reinstall it (pip install numba).") from exc
        cpu = torch.device("cpu")
        for p in self.pairs:
            p.device = cpu

    def _attach_avoid_points(self, avoid_points):
        """Attach ``vol_avoid``'s avoid cloud to every pair as ``p.avoid_points``: one ``(K,3)``
        array for all pairs, or a list of per-pair ``(K_i,3)`` arrays."""
        if avoid_points is None:
            raise TypeError("vol_avoid requires avoid_points")

        # One point cloud per pair, or one shared point cloud broadcast to every pair
        if isinstance(avoid_points, (list, tuple)):
            if len(avoid_points) != len(self.pairs):
                raise ValueError(f"avoid_points list length {len(avoid_points)} != "
                                 f"number of pairs {len(self.pairs)}")
            clouds = [np.ascontiguousarray(np.asarray(a, dtype=np.float32)) for a in avoid_points]
        else:
            cloud = np.ascontiguousarray(np.asarray(avoid_points, dtype=np.float32))
            clouds = [cloud] * len(self.pairs)

        for p, c in zip(self.pairs, clouds):
            p.avoid_points = c

    def _run_accel(self, name: str, kw: dict, num_workers: int):
        """Run the batched accel aligner for ``name``, which stores results on each pair.
        ``num_workers > 1`` on CPU shards the pairs across ``accel.cpu_pool``."""
        from shepherd_score.accel.batch import aligners

        pairs = self.pairs
        if num_workers > 1 and pairs and pairs[0].device.type == "cpu":
            from shepherd_score.accel import cpu_pool
            if name in cpu_pool.POOL_MODES:
                cpu_pool.align_pairs(name, pairs, num_workers, kw)
                return

        # Single process: the accelerated aligner bound in accel.batch.aligners
        with _pinned_torch_threads(pairs):
            getattr(aligners, f"_align_batch_{name}")(pairs, **kw)

    def _aligned_fits(self, spec):
        """Each pair's cached fit tensors moved by its stored transform, as lists of arrays:
        ``[points]``, or ``[anchors, vectors]`` for the pharmacophore modes (vectors are rotated
        only). Batched per device, so multi-GPU shards are handled."""
        import torch
        from shepherd_score.accel.channels import CHANNELS

        # The fit point cloud(s) that are rotated/translated
        if spec.pharm_style:
            channels = ('pharm_ancs', 'pharm_vecs')
        else:
            channels = ('surf',) if 'surf' in spec.channels else ('atoms',)
        attrs = [CHANNELS[c].fit_attr for c in channels]

        pairs = self.pairs
        out = [[None] * len(pairs) for _ in attrs]

        # Group by device to do one batched matmul
        by_dev: dict = {}
        for i, p in enumerate(pairs):
            by_dev.setdefault(getattr(p, attrs[0]).device, []).append(i)

        for dev, idxs in by_dev.items():
            Ss = torch.stack([torch.as_tensor(getattr(pairs[i], spec.transform_attr),
                                              dtype=torch.float32) for i in idxs]).to(dev)
            Rt = Ss[:, :3, :3].transpose(1, 2)

            for k, attr in enumerate(attrs):
                fits = [getattr(pairs[i], attr) for i in idxs]
                fit_pad = torch.nn.utils.rnn.pad_sequence(fits, batch_first=True)

                # row-vector SE(3) convention: points -> x @ R.T + t; vectors -> v @ R.T
                moved = (torch.baddbmm(Ss[:, :3, 3][:, None, :], fit_pad, Rt) if k == 0
                         else torch.bmm(fit_pad, Rt)).cpu().numpy()

                # Unpad
                for j, i in enumerate(idxs):
                    out[k][i] = moved[j, :fits[j].shape[0]]

        return out

    def _align_per_pair(self, spec, backend: str, **kwargs):
        """Run each pair's ``MoleculePair.align`` for the registry mode ``spec`` and store the
        results in its attributes. Returns the aligned fit inputs in ``_aligned_fits`` form."""
        name = spec.name
        params = dict(kwargs)

        # <base>_tversky -> MoleculePair.align(<base>, similarity=...)
        if name.endswith('_tversky'):
            name = name[:-len('_tversky')]
            if name == 'pharm':
                params.setdefault('similarity', 'tversky')
            else:
                params['similarity'] = (params.pop('tversky_alpha', 0.95),
                                        params.pop('tversky_beta', 0.05))

        # Result attributes
        tf_attr, sc_attr = spec.attrs
        if name in ('vol', 'vol_esp') and params.get('no_H') is False and name == spec.name:
            tf_attr, sc_attr = f'transform_{name}', f'sim_aligned_{name}'   # with-H attributes

        opt = {k: params.pop(k) for k in ('num_repeats', 'lr', 'max_num_steps', 'trans_init',
                                          'verbose')}

        # 'auto' is MoleculePair's PyTorch path; jax falls back to it without a JAX optimizer
        if backend == 'jax':
            mode = get_mode(name, **({'avoid_points': None} if name == 'vol_avoid' else {}),
                            **params)
            if 'jax' not in mode.backends():
                warnings.warn(f"{spec.name} has no JAX implementation; backend='jax' falls back "
                              "to per-pair PyTorch.", RuntimeWarning, stacklevel=4)
                backend = 'auto'
        else:
            backend = 'auto'

        # Align each pair and write its attributes
        out = [[None] * len(self.pairs) for _ in range(2 if spec.pharm_style else 1)]
        for i, pair in enumerate(self.pairs):
            if name == 'vol_avoid':
                params['avoid_points'] = pair.avoid_points
            res = pair.align(name, backend=backend, **opt, **params)
            setattr(pair, tf_attr, res.transform)
            setattr(pair, sc_attr, res.score)

            # Combined modes return a tuple of per-term inputs
            aligned = res.aligned[0] if isinstance(res.aligned, tuple) else res.aligned
            if spec.pharm_style:
                out[0][i], out[1][i] = aligned['anchors'], aligned['vectors']
            else:
                out[0][i] = aligned['points']

        return out

    def _delegate_alignment(self, method_name: str, score_attr: str, **kwargs):
        """Delegate alignment to each MoleculePair's method and collect results.

        Parameters
        ----------
        method_name : str
            Name of the MoleculePair method to call (e.g. 'align_with_surf').
        score_attr : str
            Name of the attribute on MoleculePair where the score is stored after alignment.
        **kwargs
            Forwarded to each pair's method.

        Returns
        -------
        scores : np.ndarray
            Shape: (N,).
        aligned_list : list of np.ndarray
            Aligned fit coordinates for each pair.
        """
        aligned_list = []
        scores = np.zeros(len(self.pairs))
        for i, pair in enumerate(self.pairs):
            aligned_pts = getattr(pair, method_name)(**kwargs)
            scores[i] = float(getattr(pair, score_attr))
            aligned_list.append(aligned_pts)
        return scores, aligned_list

    def align_with_vol(self,
                       no_H: bool = True,
                       num_repeats: int = None,
                       trans_init: bool = False,
                       lr: Optional[float] = None,
                       max_num_steps: int = None,
                       num_workers: int = 1,
                       use_shmap: bool = True,
                       num_buckets: int = 1,
                       verbose: bool = False,
                       backend: Optional[str] = None,
                       alpha: float = 0.81,
                       return_aligned: bool = False,
                       similarity: Similarity = 'tanimoto',
                       ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Align all pairs by volumetric similarity.

        Results are stored in-place on each MoleculePair:
        - ``pair.transform_vol_noH`` / ``pair.sim_aligned_vol_noH`` (when ``no_H=True``)
        - ``pair.transform_vol``     / ``pair.sim_aligned_vol``      (when ``no_H=False``)

        Parameters
        ----------
        no_H : bool
            Whether to exclude hydrogens. Default is True; triton/numba support only True.
        alpha : float
            Gaussian width. Default is 0.81.
        num_workers : int
            On ``"jax"``, ``> 1`` runs the pairs in parallel: with ``use_shmap=True`` (default)
            through ``jax.shard_map`` over ``len(jax.devices())`` virtual CPU devices (set
            ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` before JAX is first imported;
            fixed steps, no early stopping), otherwise with ``multiprocessing`` spawn workers.
            On ``"numba"``, see the class docstring.
        use_shmap : bool
            See ``num_workers``. Default is True.
        num_buckets : int
            JAX only. ``1`` (default) pads all pairs to the global atom-count maximum. Values
            > 1 sort pairs by ``(max(ref,fit), min(ref,fit))`` and pad each bucket separately.
        num_repeats, trans_init, lr, max_num_steps, verbose, backend, return_aligned, similarity
            See the class docstring.

        Returns
        -------
        scores : np.ndarray
            Scores for each pair. Shape: (N,).
        aligned_list : list of np.ndarray
            Aligned fit atom coordinates (unpadded) for each pair.
        """
        backend = _resolve_backend(backend)
        if backend != 'jax' or tversky_weights(similarity) is not None:
            return self._align_with(
                'vol', similarity, alpha=alpha, no_H=no_H, num_repeats=num_repeats, lr=lr,
                max_num_steps=max_num_steps, trans_init=trans_init, num_workers=num_workers,
                verbose=verbose, backend=backend, return_aligned=return_aligned)
        num_repeats, lr, max_num_steps = _reference_opt(num_repeats, lr, max_num_steps)

        # build raw (unpadded) position arrays for every pair
        raw_refs, raw_fits, trans_centers_list = [], [], []
        for pair in self.pairs:
            if no_H:
                ref_pos = pair.ref_molec.atom_pos
                fit_pos = pair.fit_molec.atom_pos
            else:
                ref_pos = pair.ref_molec.mol.GetConformer().GetPositions().astype(np.float32)
                fit_pos = pair.fit_molec.mol.GetConformer().GetPositions().astype(np.float32)
            raw_refs.append(ref_pos)
            raw_fits.append(fit_pos)

            tc = None
            if trans_init:
                tc = ref_pos  # already numpy; worker copies implicitly
            trans_centers_list.append(tc)

        n_pairs = len(self.pairs)
        scores = np.zeros(n_pairs)
        aligned_list = [None] * n_pairs

        if use_shmap and num_workers > 1:  # shard_map path (single process, multi-device)
            _jax_ver = _pkg_version("jax")
            _jax_ver_tuple = tuple(int(x) for x in _jax_ver.split(".")[:2])
            if _jax_ver_tuple < (0, 9):
                raise RuntimeError(
                    f"use_shmap=True requires JAX >= 0.9.0, but found JAX {_jax_ver}. "
                    "Either upgrade JAX (which requires Python >= 3.11) or set use_shmap=False."
                )

            pair_data = list(zip(raw_refs, raw_fits, trans_centers_list))
            results = _align_vol_shmap(
                pair_data, num_workers, num_repeats, lr, max_num_steps, verbose,
                num_buckets=num_buckets, alpha=alpha,
            )
            for i, (score, se3_transform, aligned_pts) in enumerate(results):
                scores[i] = score
                aligned_list[i] = aligned_pts
                pair = self.pairs[i]
                if no_H:
                    pair.transform_vol_noH = se3_transform
                    pair.sim_aligned_vol_noH = score
                else:
                    pair.transform_vol = se3_transform
                    pair.sim_aligned_vol = score

        elif num_workers > 1:  # multiprocessing path
            pair_data = list(zip(raw_refs, raw_fits, trans_centers_list))
            ref_sizes = np.array([len(r) for r in raw_refs])
            fit_sizes = np.array([len(f) for f in raw_fits])
            # Primary key: max(ref, fit) — dominates padding; secondary: min.
            sort_keys = np.array([np.minimum(ref_sizes, fit_sizes),
                                   np.maximum(ref_sizes, fit_sizes)])
            index_splits, chunk_results = _dispatch_parallel(
                pair_data, sort_keys, _align_vol_worker, num_workers,
                (num_repeats, lr, max_num_steps, verbose, alpha),
            )

            for idx_list, chunk_result in zip(index_splits, chunk_results):
                for global_i, (score, se3_transform, aligned_pts) in zip(idx_list, chunk_result):
                    scores[global_i] = score
                    aligned_list[global_i] = aligned_pts
                    pair = self.pairs[global_i]
                    if no_H:
                        pair.transform_vol_noH = se3_transform
                        pair.sim_aligned_vol_noH = score
                    else:
                        pair.transform_vol = se3_transform
                        pair.sim_aligned_vol = score

        else: # sequential
            try:
                import jax.numpy as jnp
            except ImportError as exc:
                raise ImportError(
                    'JAX is required for MoleculePairBatch.align_with_vol. '
                    'Install it with: pip install "shepherd-score[jax]"'
                ) from exc

            from shepherd_score.alignment_jax import optimize_ROCS_overlay_jax_mask

            ref_sizes = np.array([len(r) for r in raw_refs])
            fit_sizes = np.array([len(f) for f in raw_fits])
            bucket_splits = _compute_bucket_splits(ref_sizes, fit_sizes, num_buckets)

            for bucket_idx_list in bucket_splits:
                bucket_refs = [raw_refs[i] for i in bucket_idx_list]
                bucket_fits = [raw_fits[i] for i in bucket_idx_list]
                ref_padded_b, masks_ref_b, _orig_refs_b, _ = _pad_arrays(bucket_refs)
                fit_padded_b, masks_fit_b, orig_fits_b, _ = _pad_arrays(bucket_fits)

                for local_j, global_i in enumerate(bucket_idx_list):
                    pair = self.pairs[global_i]
                    aligned_pts, se3_transform, score = optimize_ROCS_overlay_jax_mask(
                        ref_points=jnp.array(ref_padded_b[local_j]),
                        fit_points=jnp.array(fit_padded_b[local_j]),
                        mask_ref=jnp.array(masks_ref_b[local_j]),
                        mask_fit=jnp.array(masks_fit_b[local_j]),
                        alpha=alpha,
                        num_repeats=num_repeats,
                        trans_centers=trans_centers_list[global_i],
                        lr=lr,
                        max_num_steps=max_num_steps,
                        verbose=verbose,
                    )

                    se3_transform = np.array(se3_transform)
                    score = float(np.array(score))
                    aligned_pts = np.array(aligned_pts)[:orig_fits_b[local_j]]
                    scores[global_i] = score

                    if no_H:
                        pair.transform_vol_noH = se3_transform
                        pair.sim_aligned_vol_noH = score
                    else:
                        pair.transform_vol = se3_transform
                        pair.sim_aligned_vol = score

                    aligned_list[global_i] = aligned_pts

        return scores, aligned_list

    def align_with_vol_esp(self,
                           lam: float = 0.1,
                           no_H: bool = True,
                           num_repeats: int = None,
                           trans_init: bool = False,
                           lr: Optional[float] = None,
                           max_num_steps: int = None,
                           num_workers: int = 1,
                           use_shmap: bool = True,
                           num_buckets: int = 1,
                           verbose: bool = False,
                           backend: Optional[str] = None,
                           alpha: float = 0.81,
                           return_aligned: bool = False,
                           similarity: Similarity = 'tanimoto',
                           ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Align all pairs by volumetric ESP similarity.

        Results are stored in-place on each MoleculePair:
        - ``pair.transform_vol_esp_noH`` / ``pair.sim_aligned_vol_esp_noH`` (when ``no_H=True``)
        - ``pair.transform_vol_esp``     / ``pair.sim_aligned_vol_esp``      (when ``no_H=False``)

        Parameters
        ----------
        lam : float
            Partial charge weighting parameter, used raw. Default is 0.1.
        alpha : float
            Gaussian width. Default is 0.81.
        no_H, num_workers, use_shmap, num_buckets
            As in :meth:`align_with_vol`.
        num_repeats, trans_init, lr, max_num_steps, verbose, backend, return_aligned, similarity
            See the class docstring.

        Returns
        -------
        scores : np.ndarray
            Scores for each pair. Shape: (N,).
        aligned_list : list of np.ndarray
            Aligned fit atom coordinates (unpadded) for each pair.
        """
        backend = _resolve_backend(backend)
        if backend != 'jax' or tversky_weights(similarity) is not None:
            return self._align_with(
                'vol_esp', similarity, alpha=alpha, lam=lam, no_H=no_H, num_repeats=num_repeats,
                lr=lr, max_num_steps=max_num_steps, trans_init=trans_init,
                num_workers=num_workers, verbose=verbose, backend=backend,
                return_aligned=return_aligned)
        num_repeats, lr, max_num_steps = _reference_opt(num_repeats, lr, max_num_steps)

        raw_refs, raw_fits, raw_ref_ch, raw_fit_ch, trans_centers_list = [], [], [], [], []
        for pair in self.pairs:
            if no_H:
                ref_pos = pair.ref_molec.atom_pos
                fit_pos = pair.fit_molec.atom_pos
                ref_ch = pair.ref_molec.partial_charges[pair.ref_molec._nonH_atoms_idx]
                fit_ch = pair.fit_molec.partial_charges[pair.fit_molec._nonH_atoms_idx]
            else:
                ref_pos = pair.ref_molec.mol.GetConformer().GetPositions().astype(np.float32)
                fit_pos = pair.fit_molec.mol.GetConformer().GetPositions().astype(np.float32)
                ref_ch = pair.ref_molec.partial_charges
                fit_ch = pair.fit_molec.partial_charges
            raw_refs.append(ref_pos)
            raw_fits.append(fit_pos)
            raw_ref_ch.append(ref_ch)
            raw_fit_ch.append(fit_ch)
            tc = ref_pos if trans_init else None
            trans_centers_list.append(tc)

        n_pairs = len(self.pairs)
        scores = np.zeros(n_pairs)
        aligned_list = [None] * n_pairs

        if use_shmap and num_workers > 1:  # shard_map path
            _jax_ver = _pkg_version("jax")
            _jax_ver_tuple = tuple(int(x) for x in _jax_ver.split(".")[:2])
            if _jax_ver_tuple < (0, 9):
                raise RuntimeError(
                    f"use_shmap=True requires JAX >= 0.9.0, but found JAX {_jax_ver}. "
                    "Either upgrade JAX (which requires Python >= 3.11) or set use_shmap=False."
                )

            pair_data = list(zip(raw_refs, raw_fits, raw_ref_ch, raw_fit_ch, trans_centers_list))
            results = _align_vol_esp_shmap(
                pair_data, num_workers, lam, num_repeats, lr, max_num_steps, verbose,
                num_buckets=num_buckets, alpha=alpha,
            )
            for i, (score, se3_transform, aligned_pts) in enumerate(results):
                scores[i] = score
                aligned_list[i] = aligned_pts
                pair = self.pairs[i]
                if no_H:
                    pair.transform_vol_esp_noH = se3_transform
                    pair.sim_aligned_vol_esp_noH = score
                else:
                    pair.transform_vol_esp = se3_transform
                    pair.sim_aligned_vol_esp = score

        elif num_workers > 1: # parallel
            pair_data = list(zip(raw_refs, raw_fits, raw_ref_ch, raw_fit_ch, trans_centers_list))
            ref_sizes = np.array([len(r) for r in raw_refs])
            fit_sizes = np.array([len(f) for f in raw_fits])
            sort_keys = np.array([np.minimum(ref_sizes, fit_sizes),
                                   np.maximum(ref_sizes, fit_sizes)])
            index_splits, chunk_results = _dispatch_parallel(
                pair_data, sort_keys, _align_vol_esp_worker, num_workers,
                (lam, num_repeats, lr, max_num_steps, verbose, alpha),
            )

            for idx_list, chunk_result in zip(index_splits, chunk_results):
                for global_i, (score, se3_transform, aligned_pts) in zip(idx_list, chunk_result):
                    scores[global_i] = score
                    aligned_list[global_i] = aligned_pts
                    pair = self.pairs[global_i]
                    if no_H:
                        pair.transform_vol_esp_noH = se3_transform
                        pair.sim_aligned_vol_esp_noH = score
                    else:
                        pair.transform_vol_esp = se3_transform
                        pair.sim_aligned_vol_esp = score

        else: # sequential
            try:
                import jax.numpy as jnp
            except ImportError as exc:
                raise ImportError(
                    'JAX is required for MoleculePairBatch.align_with_vol_esp. '
                    'Install it with: pip install "shepherd-score[jax]"'
                ) from exc

            from shepherd_score.alignment_jax import optimize_ROCS_esp_overlay_jax_mask

            ref_sizes = np.array([len(r) for r in raw_refs])
            fit_sizes = np.array([len(f) for f in raw_fits])
            bucket_splits = _compute_bucket_splits(ref_sizes, fit_sizes, num_buckets)

            for bucket_idx_list in bucket_splits:
                bucket_refs = [raw_refs[i] for i in bucket_idx_list]
                bucket_fits = [raw_fits[i] for i in bucket_idx_list]
                bucket_ref_ch = [raw_ref_ch[i] for i in bucket_idx_list]
                bucket_fit_ch = [raw_fit_ch[i] for i in bucket_idx_list]
                ref_padded_b, masks_ref_b, _orig_refs_b, _ = _pad_arrays(bucket_refs)
                fit_padded_b, masks_fit_b, orig_fits_b, _ = _pad_arrays(bucket_fits)
                ref_ch_padded_b, _, _, _ = _pad_arrays(bucket_ref_ch)
                fit_ch_padded_b, _, _, _ = _pad_arrays(bucket_fit_ch)

                for local_j, global_i in enumerate(bucket_idx_list):
                    pair = self.pairs[global_i]
                    aligned_pts, se3_transform, score = optimize_ROCS_esp_overlay_jax_mask(
                        ref_points=jnp.array(ref_padded_b[local_j]),
                        fit_points=jnp.array(fit_padded_b[local_j]),
                        ref_charges=jnp.array(ref_ch_padded_b[local_j]),
                        fit_charges=jnp.array(fit_ch_padded_b[local_j]),
                        mask_ref=jnp.array(masks_ref_b[local_j]),
                        mask_fit=jnp.array(masks_fit_b[local_j]),
                        alpha=alpha,
                        lam=lam,
                        num_repeats=num_repeats,
                        trans_centers=trans_centers_list[global_i],
                        lr=lr,
                        max_num_steps=max_num_steps,
                        verbose=verbose,
                    )

                    se3_transform = np.array(se3_transform)
                    score = float(np.array(score))
                    aligned_pts = np.array(aligned_pts)[:orig_fits_b[local_j]]
                    scores[global_i] = score

                    if no_H:
                        pair.transform_vol_esp_noH = se3_transform
                        pair.sim_aligned_vol_esp_noH = score
                    else:
                        pair.transform_vol_esp = se3_transform
                        pair.sim_aligned_vol_esp = score

                    aligned_list[global_i] = aligned_pts

        return scores, aligned_list

    def align_with_surf(self,
                        alpha: float,
                        num_repeats: int = None,
                        trans_init: bool = False,
                        lr: Optional[float] = None,
                        max_num_steps: int = None,
                        use_jax: bool = True,
                        use_analytical: bool = True,
                        num_workers: int = 1,
                        use_shmap: bool = False,
                        verbose: bool = False,
                        backend: Optional[str] = None,
                        return_aligned: bool = False,
                        similarity: Similarity = 'tanimoto',
                        ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Align all pairs by surface similarity.

        Results are stored in-place on each MoleculePair:
        - ``pair.transform_surf`` and ``pair.sim_aligned_surf``

        Parameters
        ----------
        alpha : float
            Gaussian width parameter for overlap.
        use_jax, use_analytical : bool
            ``backend="jax"`` only: the per-pair optimizer used when ``num_workers == 1``
            (JAX by default; ``use_jax=False`` runs PyTorch, analytical gradients unless
            ``use_analytical=False``).
        num_workers : int
            On ``"jax"``, ``> 1`` runs the pairs in parallel, with ``jax.shard_map`` when
            ``use_shmap=True`` (default False, faster on CPU) or ``multiprocessing``. Surface
            arrays are the same size for every pair, so no padding is needed. On ``"numba"``,
            see the class docstring.
        num_repeats, trans_init, lr, max_num_steps, verbose, backend, return_aligned, similarity
            See the class docstring.

        Returns
        -------
        scores : np.ndarray
            Scores for each pair. Shape: (N,).
        aligned_list : list of np.ndarray
            Aligned fit surface coordinates for each pair.
        """
        backend = _resolve_backend(backend)
        if backend != 'jax' or tversky_weights(similarity) is not None:
            return self._align_with(
                'surf', similarity, alpha=alpha, num_repeats=num_repeats, lr=lr,
                max_num_steps=max_num_steps, trans_init=trans_init, num_workers=num_workers,
                verbose=verbose, backend=backend, return_aligned=return_aligned)
        num_repeats, lr, max_num_steps = _reference_opt(num_repeats, lr, max_num_steps)

        n_pairs = len(self.pairs)
        pair_data = [
            (pair.ref_molec.surf_pos,
             pair.fit_molec.surf_pos,
             pair.ref_molec.atom_pos if trans_init else None)
            for pair in self.pairs
        ]

        if use_shmap and num_workers > 1:  # shard_map path
            _jax_ver = _pkg_version("jax")
            _jax_ver_tuple = tuple(int(x) for x in _jax_ver.split(".")[:2])
            if _jax_ver_tuple < (0, 9):
                raise RuntimeError(
                    f"use_shmap=True requires JAX >= 0.9.0, but found JAX {_jax_ver}. "
                    "Either upgrade JAX (which requires Python >= 3.11) or set use_shmap=False."
                )

            results = _align_surf_shmap(
                pair_data, num_workers, alpha, num_repeats, lr, max_num_steps, verbose,
            )
            scores = np.zeros(n_pairs)
            aligned_list = [None] * n_pairs
            for i, (score, se3_transform, aligned_pts) in enumerate(results):
                scores[i] = score
                aligned_list[i] = aligned_pts
                pair = self.pairs[i]
                pair.transform_surf = se3_transform
                pair.sim_aligned_surf = score
            return scores, aligned_list

        elif num_workers > 1: # parallel
            index_splits, chunk_results = _dispatch_parallel(
                pair_data, None, _align_surf_worker, num_workers,
                (alpha, num_repeats, lr, max_num_steps, use_jax, use_analytical, verbose),
            )

            scores = np.zeros(n_pairs)
            aligned_list = [None] * n_pairs
            for idx_list, chunk_result in zip(index_splits, chunk_results):
                for global_i, (score, se3_transform, aligned_pts) in zip(idx_list, chunk_result):
                    scores[global_i] = score
                    aligned_list[global_i] = aligned_pts
                    pair = self.pairs[global_i]
                    pair.transform_surf = se3_transform
                    pair.sim_aligned_surf = score
            return scores, aligned_list

        # sequential
        return self._delegate_alignment(
            'align_with_surf', 'sim_aligned_surf',
            alpha=alpha,
            num_repeats=num_repeats,
            trans_init=trans_init,
            lr=lr,
            max_num_steps=max_num_steps,
            use_jax=use_jax,
            use_analytical=use_analytical,
            verbose=verbose,
        )

    def align_with_surf_esp(self,
                       alpha: float,
                       lam: float = 0.3,
                       num_repeats: int = None,
                       trans_init: bool = False,
                       lr: Optional[float] = None,
                       max_num_steps: int = None,
                       use_jax: bool = True,
                       use_analytical: bool = True,
                       num_workers: int = 1,
                       use_shmap: bool = False,
                       verbose: bool = False,
                       backend: Optional[str] = None,
                       return_aligned: bool = False,
                       similarity: Similarity = 'tanimoto',
                       ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Align all pairs by ESP+surface similarity (formerly ``align_with_esp``, which is kept
        as an alias).

        Results are stored in-place on each MoleculePair:
        - ``pair.transform_surf_esp`` and ``pair.sim_aligned_surf_esp``

        Parameters
        ----------
        alpha : float
            Gaussian width parameter for overlap.
        lam : float
            Weighting factor for ESP scoring, scaled internally by ``LAM_SCALING``. Default 0.3.
        use_jax, use_analytical, num_workers, use_shmap
            As in :meth:`align_with_surf`.
        num_repeats, trans_init, lr, max_num_steps, verbose, backend, return_aligned, similarity
            See the class docstring.

        Returns
        -------
        scores : np.ndarray
            Scores for each pair. Shape: (N,).
        aligned_list : list of np.ndarray
            Aligned fit surface coordinates for each pair.
        """
        backend = _resolve_backend(backend)
        if backend != 'jax' or tversky_weights(similarity) is not None:
            return self._align_with(
                'surf_esp', similarity, alpha=alpha, lam=lam, num_repeats=num_repeats, lr=lr,
                max_num_steps=max_num_steps, trans_init=trans_init, num_workers=num_workers,
                verbose=verbose, backend=backend, return_aligned=return_aligned)
        num_repeats, lr, max_num_steps = _reference_opt(num_repeats, lr, max_num_steps)

        from shepherd_score.score.constants import LAM_SCALING
        lam_scaled = float(LAM_SCALING * lam)

        n_pairs = len(self.pairs)
        pair_data = [
            (pair.ref_molec.surf_pos,
             pair.fit_molec.surf_pos,
             pair.ref_molec.surf_esp,
             pair.fit_molec.surf_esp,
             pair.ref_molec.atom_pos if trans_init else None)
            for pair in self.pairs
        ]

        if use_shmap and num_workers > 1:  # shard_map path
            _jax_ver = _pkg_version("jax")
            _jax_ver_tuple = tuple(int(x) for x in _jax_ver.split(".")[:2])
            if _jax_ver_tuple < (0, 9):
                raise RuntimeError(
                    f"use_shmap=True requires JAX >= 0.9.0, but found JAX {_jax_ver}. "
                    "Either upgrade JAX (which requires Python >= 3.11) or set use_shmap=False."
                )

            results = _align_esp_shmap(
                pair_data, num_workers, alpha, lam_scaled, num_repeats, lr, max_num_steps, verbose,
            )
            scores = np.zeros(n_pairs)
            aligned_list = [None] * n_pairs
            for i, (score, se3_transform, aligned_pts) in enumerate(results):
                scores[i] = score
                aligned_list[i] = aligned_pts
                pair = self.pairs[i]
                pair.transform_surf_esp = se3_transform
                pair.sim_aligned_surf_esp = score
            return scores, aligned_list

        elif num_workers > 1: # parallel
            index_splits, chunk_results = _dispatch_parallel(
                pair_data, None, _align_esp_worker, num_workers,
                (alpha, lam_scaled, num_repeats, lr, max_num_steps,
                 use_jax, use_analytical, verbose),
            )

            scores = np.zeros(n_pairs)
            aligned_list = [None] * n_pairs
            for idx_list, chunk_result in zip(index_splits, chunk_results):
                for global_i, (score, se3_transform, aligned_pts) in zip(idx_list, chunk_result):
                    scores[global_i] = score
                    aligned_list[global_i] = aligned_pts
                    pair = self.pairs[global_i]
                    pair.transform_surf_esp = se3_transform
                    pair.sim_aligned_surf_esp = score
            return scores, aligned_list

        # sequential
        return self._delegate_alignment(
            'align_with_surf_esp', 'sim_aligned_surf_esp',
            alpha=alpha,
            lam=lam,
            num_repeats=num_repeats,
            trans_init=trans_init,
            lr=lr,
            max_num_steps=max_num_steps,
            use_jax=use_jax,
            use_analytical=use_analytical,
            verbose=verbose,
        )

    def align_with_pharm(self,
                         similarity: str = 'tanimoto',
                         extended_points: bool = False,
                         only_extended: bool = False,
                         num_repeats: int = None,
                         trans_init: bool = False,
                         lr: Optional[float] = None,
                         max_num_steps: int = None,
                         num_workers: int = 1,
                         use_shmap: bool = True,
                         num_buckets: int = 1,
                         verbose: bool = False,
                         backend: Optional[str] = None,
                         return_aligned: bool = False,
                         ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Align all pairs by pharmacophore similarity.

        Results are stored in-place on each MoleculePair:
        - ``pair.transform_pharm`` and ``pair.sim_aligned_pharm``

        Parameters
        ----------
        similarity : str
            One of ``'tanimoto'``, ``'tversky'``, ``'tversky_ref'``, ``'tversky_fit'``. Unlike
            the other modes, Tversky results stay in the ``pharm`` slots.
        extended_points : bool
            Score HBA/HBD with extended-point Gaussians.
        only_extended : bool
            When ``extended_points`` is True, ignore anchor overlaps.
        num_workers, use_shmap, num_buckets
            As in :meth:`align_with_vol`, over pharmacophore counts.
        num_repeats, trans_init, lr, max_num_steps, verbose, backend, return_aligned
            See the class docstring.

        Returns
        -------
        scores : np.ndarray
            Shape: (N,).
        aligned_anchors_list : list of np.ndarray
            Aligned fit pharmacophore anchors (unpadded) for each pair.
        aligned_vectors_list : list of np.ndarray
            Aligned fit pharmacophore vectors (unpadded) for each pair.
        """
        backend = _resolve_backend(backend)
        if backend != 'jax':
            return self._align_mode(
                'pharm', similarity=similarity, extended_points=extended_points,
                only_extended=only_extended, num_repeats=num_repeats, lr=lr,
                max_num_steps=max_num_steps, trans_init=trans_init, num_workers=num_workers,
                verbose=verbose, backend=backend, return_aligned=return_aligned)
        num_repeats, lr, max_num_steps = _reference_opt(num_repeats, lr, max_num_steps)

        # Validate pharmacophore data and collect raw arrays for all pairs.
        for idx, pair in enumerate(self.pairs):
            if (pair.ref_molec.pharm_types is None or
                    pair.fit_molec.pharm_types is None):
                raise ValueError(
                    f'Pair {idx} is missing pharmacophore data. '
                    'Create Molecule objects with pharm_multi_vector set to True or False.'
                )

        n_pairs = len(self.pairs)
        scores = np.zeros(n_pairs)
        aligned_anchors_list = [None] * n_pairs
        aligned_vectors_list = [None] * n_pairs

        # Build raw (unpadded) per-pair data tuples (plain numpy — picklable).
        pair_data = []
        for pair in self.pairs:
            tc = pair.ref_molec.pharm_ancs if trans_init else None
            pair_data.append((
                pair.ref_molec.pharm_types,
                pair.fit_molec.pharm_types,
                pair.ref_molec.pharm_ancs,
                pair.fit_molec.pharm_ancs,
                pair.ref_molec.pharm_vecs,
                pair.fit_molec.pharm_vecs,
                tc,
                pair.ref_molec.pharm_ancs,
                pair.fit_molec.pharm_ancs,
            ))

        if use_shmap and num_workers > 1:  # shard_map path
            _jax_ver = _pkg_version("jax")
            _jax_ver_tuple = tuple(int(x) for x in _jax_ver.split(".")[:2])
            if _jax_ver_tuple < (0, 9):
                raise RuntimeError(
                    f"use_shmap=True requires JAX >= 0.9.0, but found JAX {_jax_ver}. "
                    "Either upgrade JAX (which requires Python >= 3.11) or set use_shmap=False."
                )

            results = _align_pharm_shmap(
                pair_data, num_workers, similarity, extended_points, only_extended,
                num_repeats, lr, max_num_steps, verbose, num_buckets=num_buckets,
            )
            for i, (score, se3_transform, aligned_ancs, aligned_vecs) in enumerate(results):
                scores[i] = score
                aligned_anchors_list[i] = aligned_ancs
                aligned_vectors_list[i] = aligned_vecs
                pair = self.pairs[i]
                pair.transform_pharm = se3_transform
                pair.sim_aligned_pharm = score

        elif num_workers > 1: # parallel
            ref_sizes = np.array([len(d[2]) for d in pair_data])  # ref_ancs
            fit_sizes = np.array([len(d[3]) for d in pair_data])  # fit_ancs
            # Primary key: max(ref, fit) — dominates padding; secondary: min.
            sort_keys = np.array([np.minimum(ref_sizes, fit_sizes),
                                   np.maximum(ref_sizes, fit_sizes)])
            index_splits, chunk_results = _dispatch_parallel(
                pair_data, sort_keys, _align_pharm_worker, num_workers,
                (similarity, extended_points, only_extended,
                 num_repeats, lr, max_num_steps, verbose),
            )

            for idx_list, chunk_result in zip(index_splits, chunk_results):
                for global_i, (score, se3_transform, aligned_ancs, aligned_vecs) in zip(
                    idx_list, chunk_result
                ):
                    scores[global_i] = score
                    aligned_anchors_list[global_i] = aligned_ancs
                    aligned_vectors_list[global_i] = aligned_vecs
                    pair = self.pairs[global_i]
                    pair.transform_pharm = se3_transform
                    pair.sim_aligned_pharm = score

        else: # sequential
            try:
                import jax.numpy as jnp
            except ImportError as exc:
                raise ImportError(
                    'JAX is required for MoleculePairBatch.align_with_pharm. '
                    'Install it with: pip install "shepherd-score[jax]"'
                ) from exc

            from shepherd_score.alignment_jax import optimize_pharm_overlay_jax_vectorized_mask

            DUMMY_TYPE = 8  # index of 'Dummy' in P_TYPES
            ref_types_list = [p.ref_molec.pharm_types for p in self.pairs]
            fit_types_list = [p.fit_molec.pharm_types for p in self.pairs]
            ref_ancs_list = [p.ref_molec.pharm_ancs for p in self.pairs]
            fit_ancs_list = [p.fit_molec.pharm_ancs for p in self.pairs]
            ref_vecs_list = [p.ref_molec.pharm_vecs for p in self.pairs]
            fit_vecs_list = [p.fit_molec.pharm_vecs for p in self.pairs]

            ref_sizes = np.array([len(a) for a in ref_ancs_list])
            fit_sizes = np.array([len(a) for a in fit_ancs_list])
            bucket_splits = _compute_bucket_splits(ref_sizes, fit_sizes, num_buckets)

            for bucket_idx_list in bucket_splits:
                bucket_ref_ancs = [ref_ancs_list[i] for i in bucket_idx_list]
                bucket_fit_ancs = [fit_ancs_list[i] for i in bucket_idx_list]
                bucket_ref_vecs = [ref_vecs_list[i] for i in bucket_idx_list]
                bucket_fit_vecs = [fit_vecs_list[i] for i in bucket_idx_list]

                ref_ancs_padded, masks_ref, orig_refs_b, max_ref_b = _pad_arrays(bucket_ref_ancs)
                fit_ancs_padded, masks_fit, orig_fits_b, max_fit_b = _pad_arrays(bucket_fit_ancs)
                ref_vecs_padded, _, _, _ = _pad_arrays(bucket_ref_vecs)
                fit_vecs_padded, _, _, _ = _pad_arrays(bucket_fit_vecs)

                for local_j, global_i in enumerate(bucket_idx_list):
                    pair = self.pairs[global_i]
                    orig_ref = orig_refs_b[local_j]
                    orig_fit = orig_fits_b[local_j]

                    ref_types_pad = np.full(max_ref_b, DUMMY_TYPE, dtype=np.int32)
                    ref_types_pad[:orig_ref] = ref_types_list[global_i]
                    fit_types_pad = np.full(max_fit_b, DUMMY_TYPE, dtype=np.int32)
                    fit_types_pad[:orig_fit] = fit_types_list[global_i]

                    trans_centers = pair.ref_molec.pharm_ancs if trans_init else None

                    aligned_ancs, aligned_vecs, se3_transform, score = \
                        optimize_pharm_overlay_jax_vectorized_mask(
                            ref_pharms=jnp.array(ref_types_pad),
                            fit_pharms=jnp.array(fit_types_pad),
                            ref_anchors=jnp.array(ref_ancs_padded[local_j]),
                            fit_anchors=jnp.array(fit_ancs_padded[local_j]),
                            ref_vectors=jnp.array(ref_vecs_padded[local_j]),
                            fit_vectors=jnp.array(fit_vecs_padded[local_j]),
                            mask_ref=jnp.array(masks_ref[local_j]),
                            mask_fit=jnp.array(masks_fit[local_j]),
                            similarity=similarity,
                            extended_points=extended_points,
                            only_extended=only_extended,
                            num_repeats=num_repeats,
                            trans_centers=trans_centers,
                            init_ref_anchors=pair.ref_molec.pharm_ancs,
                            init_fit_anchors=pair.fit_molec.pharm_ancs,
                            lr=lr,
                            max_num_steps=max_num_steps,
                            verbose=verbose,
                        )

                    se3_transform = np.array(se3_transform)
                    score = float(np.array(score))
                    aligned_ancs = np.array(aligned_ancs)[:orig_fit]
                    aligned_vecs = np.array(aligned_vecs)[:orig_fit]

                    scores[global_i] = score
                    pair.transform_pharm = se3_transform
                    pair.sim_aligned_pharm = score
                    aligned_anchors_list[global_i] = aligned_ancs
                    aligned_vectors_list[global_i] = aligned_vecs

        return scores, aligned_anchors_list, aligned_vectors_list

    def align_with_vol_and_surf_esp(self,
                                    alpha: float = 0.81,
                                    lam: float = 0.001,
                                    probe_radius: float = 1.0,
                                    esp_weight: float = 0.5,
                                    num_repeats: int = None,
                                    trans_init: bool = False,
                                    lr: Optional[float] = None,
                                    max_num_steps: int = None,
                                    verbose: bool = False,
                                    backend: Optional[str] = None,
                                    return_aligned: bool = False,
                                    similarity: Similarity = 'tanimoto',
                                    num_workers: int = 1,
                                    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Align all pairs by ShaEP-style volume + surface-ESP similarity (formerly
        ``align_with_esp_combo``, which is kept as an alias). ``alpha``, ``lam``,
        ``probe_radius`` and ``esp_weight`` are as in ``MoleculePair.align_with_vol_and_surf_esp``;
        a Tversky ``similarity`` applies to the shape channel only. Writes
        ``transform_vol_and_surf_esp`` / ``sim_aligned_vol_and_surf_esp`` and returns the aligned
        fit surface points. Other keywords: see the class docstring."""
        return self._align_with(
            'vol_and_surf_esp', similarity, alpha=alpha, lam=lam, probe_radius=probe_radius,
            esp_weight=esp_weight, num_repeats=num_repeats, lr=lr, max_num_steps=max_num_steps,
            trans_init=trans_init, num_workers=num_workers, verbose=verbose, backend=backend,
            return_aligned=return_aligned)

    # legacy method aliases (esp -> surf_esp, esp_combo -> vol_and_surf_esp)
    align_with_esp = align_with_surf_esp
    align_with_esp_combo = align_with_vol_and_surf_esp

    def align_with_vol_color(self,
                             color_weight: float = 0.5,
                             alpha: float = 0.81,
                             num_repeats: int = None,
                             trans_init: bool = False,
                             lr: Optional[float] = None,
                             max_num_steps: int = None,
                             verbose: bool = False,
                             backend: Optional[str] = None,
                             return_aligned: bool = False,
                             similarity: Similarity = 'tanimoto',
                             num_workers: int = 1,
                             ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Align all pairs by ROCS-style shape + directionless pharmacophore color,
        ``(1 - color_weight) * shape + color_weight * color``; a Tversky ``similarity`` applies to
        both channels. Writes ``transform_vol_color`` / ``sim_aligned_vol_color`` and returns the
        aligned fit atoms. Other keywords: see the class docstring."""
        return self._align_with(
            'vol_color', similarity, color_weight=color_weight, alpha=alpha,
            num_repeats=num_repeats, lr=lr, max_num_steps=max_num_steps, trans_init=trans_init,
            num_workers=num_workers, verbose=verbose, backend=backend,
            return_aligned=return_aligned)

    def align_with_vol_lipo(self,
                            lipo_weight: float = 0.5,
                            alpha: float = 0.81,
                            lam: float = 0.1,
                            num_repeats: int = None,
                            lr: Optional[float] = None,
                            max_num_steps: int = None,
                            verbose: bool = False,
                            backend: Optional[str] = None,
                            return_aligned: bool = False,
                            similarity: Similarity = 'tanimoto',
                            trans_init: bool = False,
                            num_workers: int = 1,
                            ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Align all pairs by shape + Crippen logP field,
        ``(1 - lipo_weight) * shape + lipo_weight * lipo``, with ``lam`` the raw width of the
        value-matching kernel; a Tversky ``similarity`` applies to both channels. Writes
        ``transform_vol_lipo`` / ``sim_aligned_vol_lipo`` and returns the aligned fit atoms.
        Other keywords: see the class docstring."""
        return self._align_with(
            'vol_lipo', similarity, lipo_weight=lipo_weight, alpha=alpha, lam=lam,
            num_repeats=num_repeats, lr=lr, max_num_steps=max_num_steps, trans_init=trans_init,
            num_workers=num_workers, verbose=verbose, backend=backend,
            return_aligned=return_aligned)

    def align_with_vol_mr(self, mr_weight: float = 0.5, alpha: float = 0.81, lam: float = 0.1,
                          num_repeats: int = None, lr: Optional[float] = None,
                          max_num_steps: int = None, verbose: bool = False,
                          backend: Optional[str] = None, return_aligned: bool = False,
                          trans_init: bool = False, num_workers: int = 1):
        """Shape + Crippen molar-refractivity field (``vol_mr`` slots)."""
        return self._align_mode(
            'vol_mr', mr_weight=mr_weight, alpha=alpha, lam=lam, num_repeats=num_repeats, lr=lr,
            max_num_steps=max_num_steps, trans_init=trans_init, num_workers=num_workers,
            verbose=verbose, backend=backend, return_aligned=return_aligned)

    def align_with_vol_fukui(self, fukui_weight: float = 0.5, alpha: float = 0.81,
                             lam: float = 0.1, num_repeats: int = None, lr: Optional[float] = None,
                             max_num_steps: int = None, verbose: bool = False,
                             backend: Optional[str] = None, return_aligned: bool = False,
                             trans_init: bool = False, num_workers: int = 1):
        """Shape + condensed-Fukui dual-descriptor field (``vol_fukui`` slots)."""
        return self._align_mode(
            'vol_fukui', fukui_weight=fukui_weight, alpha=alpha, lam=lam,
            num_repeats=num_repeats, lr=lr, max_num_steps=max_num_steps, trans_init=trans_init,
            num_workers=num_workers, verbose=verbose, backend=backend,
            return_aligned=return_aligned)

    def align_with_vol_avoid(self, avoid_points, avoid_weight: float = 1.0,
                             avoid_min_dist: float = 2.0, alpha: float = 0.81,
                             num_repeats: int = None, lr: Optional[float] = None,
                             max_num_steps: int = None, verbose: bool = False,
                             backend: Optional[str] = None, return_aligned: bool = False,
                             trans_init: bool = False, num_workers: int = 1):
        """Shape Tanimoto minus a linear hard-sphere excluded-volume penalty (``vol_avoid``
        slots).

        ``avoid_points`` is the fixed reference-frame cloud to keep the fit out of: a single
        ``(K,3)`` array applied to every pair, or a list of per-pair ``(K_i,3)`` arrays. It is
        attached to each pair as ``p.avoid_points``; the penalty is evaluated on the fit shape
        atoms."""
        return self._align_mode(
            'vol_avoid', avoid_points=avoid_points, avoid_weight=avoid_weight,
            avoid_min_dist=avoid_min_dist, alpha=alpha, num_repeats=num_repeats, lr=lr,
            max_num_steps=max_num_steps, trans_init=trans_init, num_workers=num_workers,
            verbose=verbose, backend=backend, return_aligned=return_aligned)

    def align_with_vol_atomtype(self, atomtype_weight: float = 0.5, alpha: float = 0.81,
                                num_repeats: int = None, lr: Optional[float] = None,
                                max_num_steps: int = None, verbose: bool = False,
                                backend: Optional[str] = None, return_aligned: bool = False,
                                trans_init: bool = False, num_workers: int = 1):
        """Shape + element-identity overlap (``vol_atomtype`` slots)."""
        return self._align_mode(
            'vol_atomtype', atomtype_weight=atomtype_weight, alpha=alpha,
            num_repeats=num_repeats, lr=lr, max_num_steps=max_num_steps, trans_init=trans_init,
            num_workers=num_workers, verbose=verbose, backend=backend,
            return_aligned=return_aligned)

    def align_with_vol_pharm(self, color_weight: float = 0.5, alpha: float = 0.81,
                             num_repeats: int = None, lr: Optional[float] = None,
                             max_num_steps: int = None, verbose: bool = False,
                             backend: Optional[str] = None, return_aligned: bool = False,
                             trans_init: bool = False, num_workers: int = 1):
        """Shape + directional pharmacophore, i.e. ``vol_color`` with vector weighting
        (``vol_pharm`` slots)."""
        return self._align_mode(
            'vol_pharm', color_weight=color_weight, alpha=alpha, num_repeats=num_repeats, lr=lr,
            max_num_steps=max_num_steps, trans_init=trans_init, num_workers=num_workers,
            verbose=verbose, backend=backend, return_aligned=return_aligned)
