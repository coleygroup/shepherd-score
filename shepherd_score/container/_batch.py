"""MoleculePairBatch: batch of MoleculePair objects for fast sequential JAX alignment."""
from importlib.metadata import version as _pkg_version
from typing import List, Tuple

import numpy as np

from shepherd_score.container._core import MoleculePair
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


class MoleculePairBatch:
    """Batch of MoleculePair objects for fast sequential JAX alignment.

    Pads all atom coordinate arrays to common max shapes so JAX's XLA compiler
    reuses the same compiled function for every pair, avoiding recompilation.
    This modifies each MoleculePair in-place (stores results on the pair).

    This is currently optimized for CPU. A GPU-optimized version would
    benefit from optimizing batches of pairs and using a GPU-optimized alignment.
    """

    def __init__(self, pairs: List[MoleculePair]):
        self.pairs = pairs

    def _pad_and_mask_vol(self, no_H: bool = True, include_charges: bool = False):
        """Extract, pad, and create masks for volumetric (and optionally ESP) alignment.

        Does NOT modify the pair objects. Returns padded arrays and masks.

        Parameters
        ----------
        no_H : bool
            If True, use heavy-atom positions (atom_pos). If False, use all atoms.
        include_charges : bool
            If True, also extract and pad partial charge arrays. The returned tuple
            per entry gains two extra elements: ``(ref_pos_pad, fit_pos_pad,
            ref_ch_pad, fit_ch_pad, mask_ref, mask_fit, orig_ref, orig_fit)``.
            If False, each entry is ``(ref_padded, fit_padded, mask_ref, mask_fit,
            orig_ref, orig_fit)``.

        Returns
        -------
        entries : list of tuples
        max_ref_len : int
        max_fit_len : int
        """
        if no_H:
            ref_pos_arrays = [p.ref_molec.atom_pos for p in self.pairs]
            fit_pos_arrays = [p.fit_molec.atom_pos for p in self.pairs]
            if include_charges:
                ref_ch_arrays = [p.ref_molec.partial_charges[p.ref_molec._nonH_atoms_idx]
                                 for p in self.pairs]
                fit_ch_arrays = [p.fit_molec.partial_charges[p.fit_molec._nonH_atoms_idx]
                                 for p in self.pairs]
        else:
            ref_pos_arrays = [p.ref_molec.mol.GetConformer().GetPositions().astype(np.float32)
                              for p in self.pairs]
            fit_pos_arrays = [p.fit_molec.mol.GetConformer().GetPositions().astype(np.float32)
                              for p in self.pairs]
            if include_charges:
                ref_ch_arrays = [p.ref_molec.partial_charges for p in self.pairs]
                fit_ch_arrays = [p.fit_molec.partial_charges for p in self.pairs]

        ref_padded, masks_ref, orig_refs, max_ref_len = _pad_arrays(ref_pos_arrays)
        fit_padded, masks_fit, orig_fits, max_fit_len = _pad_arrays(fit_pos_arrays)

        if include_charges:
            ref_ch_padded, _, _, _ = _pad_arrays(ref_ch_arrays)
            fit_ch_padded, _, _, _ = _pad_arrays(fit_ch_arrays)
            entries = [
                (rp, fp, rc, fc, mr, mf, ori_r, ori_f)
                for rp, fp, rc, fc, mr, mf, ori_r, ori_f in zip(
                    ref_padded, fit_padded, ref_ch_padded, fit_ch_padded,
                    masks_ref, masks_fit, orig_refs, orig_fits
                )
            ]
        else:
            entries = [
                (rp, fp, mr, mf, ori_r, ori_f)
                for rp, fp, mr, mf, ori_r, ori_f in zip(
                    ref_padded, fit_padded, masks_ref, masks_fit, orig_refs, orig_fits
                )
            ]

        return entries, max_ref_len, max_fit_len

    def align_with_vol(self,
                       no_H: bool = True,
                       num_repeats: int = 50,
                       trans_init: bool = False,
                       lr: float = 0.1,
                       max_num_steps: int = 200,
                       num_workers: int = 1,
                       use_shmap: bool = True,
                       num_buckets: int = 1,
                       verbose: bool = False,
                       ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Align all pairs using padded masked volumetric similarity via JAX.

        Because all padded arrays have the same shape, JAX's XLA compiler
        reuses one compiled kernel for every pair — no recompilation overhead.

        When ``num_workers > 1`` the pairs are split into size-sorted chunks
        and processed in parallel. It is recommended to use ``use_shmap=True``
        instead of ``multiprocessing`` for this setting.

        Results are stored in-place on each MoleculePair:
        - ``pair.transform_vol_noH`` / ``pair.sim_aligned_vol_noH`` (when ``no_H=True``)
        - ``pair.transform_vol``     / ``pair.sim_aligned_vol``      (when ``no_H=False``)

        Parameters
        ----------
        no_H : bool
            Whether to exclude hydrogens. Default is True.
        num_repeats : int
            Number of SE(3) initializations per pair. Default is 50.
        trans_init : bool
            If True, initialize translations to each ref atom position. Default is False.
        lr : float
            Optimizer learning rate. Default is 0.1.
        max_num_steps : int
            Maximum optimization steps. Default is 200.
        num_workers : int
            Number of parallel workers.  ``1`` (default) runs sequentially
            in-process. When ``use_shmap=True`` (the default), this value is informational;
            actual parallelism equals ``len(jax.devices())``, which is set by
            ``XLA_FLAGS`` **before** JAX is first imported. When ``use_shmap=False``
            use ``multiprocessing`` with a ``'spawn'`` start method.
        use_shmap : bool
            If ``True`` and ``num_workers > 1``, use ``jax.shard_map`` + ``vmap``
            to parallelise across virtual CPU devices in a single process.
            Requires ``XLA_FLAGS=--xla_force_host_platform_device_count=N``
            to be set before any JAX import.  Uses ``lax.scan`` (fixed steps,
            no early stopping) instead of the ``while_loop``-based sequential
            path.  Required on Linux HPC if num_workers > 1 where ``multiprocessing``
            spawn can be unreliable with JAX.  Default is ``True``.
        num_buckets : int
            ``1`` (default) pads all pairs to the global atom-count maximum —
            lowest overhead for typical use.  Values > 1 sort pairs by
            ``(max(ref,fit), min(ref,fit))`` and process each bucket
            separately with reduced per-bucket padding, which can be
            beneficial for large heterogeneous molecule sets.
        verbose : bool
            Print scores per pair. Default is False.

        Returns
        -------
        scores : np.ndarray
            Scores for each pair. Shape: (N,).
        aligned_list : list of np.ndarray
            Aligned fit atom coordinates (unpadded) for each pair.
        """
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
                num_buckets=num_buckets,
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
                (num_repeats, lr, max_num_steps, verbose),
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
                        alpha=0.81,
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
                           lam: float,
                           no_H: bool = True,
                           num_repeats: int = 50,
                           trans_init: bool = False,
                           lr: float = 0.1,
                           max_num_steps: int = 200,
                           num_workers: int = 1,
                           use_shmap: bool = True,
                           num_buckets: int = 1,
                           verbose: bool = False,
                           ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Align all pairs using padded masked volumetric ESP similarity via JAX.

        Because all padded arrays have the same shape, JAX's XLA compiler
        reuses one compiled kernel for every pair — no recompilation overhead.

        When ``num_workers > 1`` the pairs are split into size-sorted chunks
        and processed in parallel. It is recommended to use ``use_shmap=True``
        instead of ``multiprocessing`` for this setting.

        Results are stored in-place on each MoleculePair:
        - ``pair.transform_vol_esp_noH`` / ``pair.sim_aligned_vol_esp_noH`` (when ``no_H=True``)
        - ``pair.transform_vol_esp``     / ``pair.sim_aligned_vol_esp``      (when ``no_H=False``)

        Parameters
        ----------
        lam : float
            Partial charge weighting parameter. Typically 0.1 for volumetric.
        no_H : bool
            Whether to exclude hydrogens. Default is True.
        num_repeats : int
            Number of SE(3) initializations per pair. Default is 50.
        trans_init : bool
            If True, initialize translations to each ref atom position. Default is False.
        lr : float
            Optimizer learning rate. Default is 0.1.
        max_num_steps : int
            Maximum optimization steps. Default is 200.
        num_workers : int
            Number of parallel worker processes. ``1`` (default) runs
            sequentially in-process. Values greater than ``len(self.pairs)``
            are clamped to ``len(self.pairs)``.
        use_shmap : bool
            If ``True`` and ``num_workers > 1``, use ``jax.shard_map`` + ``vmap``
            to parallelise across virtual CPU devices in a single process.
            Requires ``XLA_FLAGS=--xla_force_host_platform_device_count=N``
            to be set before any JAX import.  Uses ``lax.scan`` (fixed steps,
            no early stopping) instead of the ``while_loop``-based sequential
            path.  Required on Linux HPC if num_workers > 1 where ``multiprocessing``
            spawn can be unreliable with JAX.  Default is ``True``.
        num_buckets : int
            ``1`` (default) pads all pairs to the global atom-count maximum —
            lowest overhead for typical use.  Values > 1 sort pairs by
            ``(max(ref,fit), min(ref,fit))`` and process each bucket
            separately with reduced per-bucket padding, which can be
            beneficial for large heterogeneous molecule sets.
        verbose : bool
            Print scores per pair. Default is False.

        Returns
        -------
        scores : np.ndarray
            Scores for each pair. Shape: (N,).
        aligned_list : list of np.ndarray
            Aligned fit atom coordinates (unpadded) for each pair.
        """
        # Build raw (unpadded) per-pair data tuples (plain numpy — picklable).
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
                num_buckets=num_buckets,
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
                (lam, num_repeats, lr, max_num_steps, verbose),
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
                        alpha=0.81,
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

    def align_with_surf(self,
                        alpha: float,
                        num_repeats: int = 50,
                        trans_init: bool = False,
                        lr: float = 0.1,
                        max_num_steps: int = 200,
                        use_jax: bool = True,
                        use_analytical: bool = True,
                        num_workers: int = 1,
                        use_shmap: bool = False,
                        verbose: bool = False,
                        ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Align all pairs using surface similarity.

        Surface arrays are the same size across all pairs so no padding or
        size-sorting is needed.  It is not recommended to use multiprocessing
        due to this reason.

        Results are stored in-place on each MoleculePair:
        - ``pair.transform_surf`` and ``pair.sim_aligned_surf``

        Parameters
        ----------
        alpha : float
            Gaussian width parameter for overlap.
        num_repeats : int
            Number of SE(3) initializations per pair. Default is 50.
        trans_init : bool
            Apply translation initialization for alignment. Default is False.
        lr : float
            Optimizer learning rate. Default is 0.1.
        max_num_steps : int
            Maximum optimization steps. Default is 200.
        use_jax : bool
            Whether to use JAX backend. Default is True.
        use_analytical : bool
            Whether to use analytical gradients (PyTorch only). Default is True.
        num_workers : int
            Number of parallel worker processes. ``1`` (default) runs
            sequentially in-process. Values greater than ``len(self.pairs)``
            are clamped to ``len(self.pairs)``.
        use_shmap : bool
            Whether to use JAX shard_map for parallel alignment. Default is False.
            Performance is better when use_shmap is False on cpu.
        verbose : bool
            Print scores per pair. Default is False.

        Returns
        -------
        scores : np.ndarray
            Scores for each pair. Shape: (N,).
        aligned_list : list of np.ndarray
            Aligned fit surface coordinates for each pair.
        """
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

    def align_with_esp(self,
                       alpha: float,
                       lam: float = 0.3,
                       num_repeats: int = 50,
                       trans_init: bool = False,
                       lr: float = 0.1,
                       max_num_steps: int = 200,
                       use_jax: bool = True,
                       use_analytical: bool = True,
                       num_workers: int = 1,
                       use_shmap: bool = False,
                       verbose: bool = False,
                       ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Align all pairs using ESP+surface similarity.

        Surface arrays are the same size across all pairs so no padding or
        size-sorting is needed.  It is not recommended to use multiprocessing
        due to this reason.

        Results are stored in-place on each MoleculePair:
        - ``pair.transform_esp`` and ``pair.sim_aligned_esp``

        Parameters
        ----------
        alpha : float
            Gaussian width parameter for overlap.
        lam : float
            Weighting factor for ESP scoring. Scaled internally. Default is 0.3.
        num_repeats : int
            Number of SE(3) initializations per pair. Default is 50.
        trans_init : bool
            Apply translation initialization for alignment. Default is False.
        lr : float
            Optimizer learning rate. Default is 0.1.
        max_num_steps : int
            Maximum optimization steps. Default is 200.
        use_jax : bool
            Whether to use JAX backend. Default is True.
        use_analytical : bool
            Whether to use analytical gradients (PyTorch only). Default is True.
        num_workers : int
            Number of parallel worker processes. ``1`` (default) runs
            sequentially in-process. Values greater than ``len(self.pairs)``
            are clamped to ``len(self.pairs)``.
        use_shmap : bool
            Whether to use JAX shard_map for parallel alignment. Default is False.
            Performance is better when use_shmap is False on cpu.
        verbose : bool
            Print scores per pair. Default is False.

        Returns
        -------
        scores : np.ndarray
            Scores for each pair. Shape: (N,).
        aligned_list : list of np.ndarray
            Aligned fit surface coordinates for each pair.
        """
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
                pair.transform_esp = se3_transform
                pair.sim_aligned_esp = score
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
                    pair.transform_esp = se3_transform
                    pair.sim_aligned_esp = score
            return scores, aligned_list

        # sequential
        return self._delegate_alignment(
            'align_with_esp', 'sim_aligned_esp',
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

    def _pad_and_mask_pharm(self):
        """Extract, pad, and create masks for pharmacophore alignment.

        Validates that all pairs have pharmacophore data. Does NOT modify the
        pair objects. Returns padded arrays and masks.

        Returns
        -------
        entries : list of tuples
            Each tuple is (ref_ptypes, fit_ptypes,
                           ref_ancs_pad, fit_ancs_pad,
                           ref_vecs_pad, fit_vecs_pad,
                           mask_ref, mask_fit,
                           orig_ref_len, orig_fit_len).
        max_ref_len : int
        max_fit_len : int
        """
        for i, pair in enumerate(self.pairs):
            if (pair.ref_molec.pharm_types is None or
                    pair.fit_molec.pharm_types is None):
                raise ValueError(
                    f'Pair {i} is missing pharmacophore data. '
                    'Create Molecule objects with pharm_multi_vector set to True or False.'
                )

        DUMMY_TYPE = 8  # index of 'Dummy' in P_TYPES

        ref_types_list = [p.ref_molec.pharm_types for p in self.pairs]
        fit_types_list = [p.fit_molec.pharm_types for p in self.pairs]

        max_ref_len = max(t.shape[0] for t in ref_types_list)
        max_fit_len = max(t.shape[0] for t in fit_types_list)

        ref_ancs_padded, masks_ref, orig_refs, _ = _pad_arrays([p.ref_molec.pharm_ancs for p in self.pairs])
        fit_ancs_padded, masks_fit, orig_fits, _ = _pad_arrays([p.fit_molec.pharm_ancs for p in self.pairs])
        ref_vecs_padded, _, _, _ = _pad_arrays([p.ref_molec.pharm_vecs for p in self.pairs])
        fit_vecs_padded, _, _, _ = _pad_arrays([p.fit_molec.pharm_vecs for p in self.pairs])

        entries = []
        for (ref_types, fit_types,
             ref_ancs_pad, fit_ancs_pad,
             ref_vecs_pad, fit_vecs_pad,
             mask_ref, mask_fit,
             orig_ref, orig_fit) in zip(
                ref_types_list, fit_types_list,
                ref_ancs_padded, fit_ancs_padded,
                ref_vecs_padded, fit_vecs_padded,
                masks_ref, masks_fit,
                orig_refs, orig_fits
        ):
            ref_types_pad = np.full(max_ref_len, DUMMY_TYPE, dtype=np.int32)
            ref_types_pad[:orig_ref] = ref_types
            fit_types_pad = np.full(max_fit_len, DUMMY_TYPE, dtype=np.int32)
            fit_types_pad[:orig_fit] = fit_types

            entries.append((ref_types_pad, fit_types_pad,
                            ref_ancs_pad, fit_ancs_pad,
                            ref_vecs_pad, fit_vecs_pad,
                            mask_ref, mask_fit,
                            orig_ref, orig_fit))

        return entries, max_ref_len, max_fit_len

    def align_with_pharm(self,
                         similarity: str = 'tanimoto',
                         extended_points: bool = False,
                         only_extended: bool = False,
                         num_repeats: int = 50,
                         trans_init: bool = False,
                         lr: float = 0.1,
                         max_num_steps: int = 200,
                         num_workers: int = 1,
                         use_shmap: bool = True,
                         num_buckets: int = 1,
                         verbose: bool = False,
                         ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Align all pairs using padded masked pharmacophore similarity via JAX.

        Because all padded arrays have the same shape, JAX's XLA compiler
        reuses one compiled kernel for every pair — no recompilation overhead.

        When ``num_workers > 1`` the pairs are split into size-sorted chunks and
        processed in parallel. It is recommended to use ``use_shmap=True``
        instead of ``multiprocessing`` for this setting.

        Results are stored in-place on each MoleculePair:
        - ``pair.transform_pharm`` and ``pair.sim_aligned_pharm``

        Parameters
        ----------
        similarity : str
            One of ``'tanimoto'``, ``'tversky'``, ``'tversky_ref'``, ``'tversky_fit'``.
        extended_points : bool
            Score HBA/HBD with extended-point Gaussians.
        only_extended : bool
            When ``extended_points`` is True, ignore anchor overlaps.
        num_repeats : int
            Number of SE(3) initializations per pair.
        trans_init : bool
            If True, initialize translations to each ref pharmacophore anchor.
        lr : float
            Optimizer learning rate.
        max_num_steps : int
            Maximum optimization steps.
        num_workers : int
            Number of parallel worker processes. ``1`` (default) runs
            sequentially in-process. Values greater than ``len(self.pairs)``
            are clamped to ``len(self.pairs)``.
        use_shmap : bool
            If ``True`` and ``num_workers > 1``, use ``jax.shard_map`` + ``vmap``
            to parallelise across virtual CPU devices in a single process.
            Requires ``XLA_FLAGS=--xla_force_host_platform_device_count=N``
            to be set before any JAX import.  Uses ``lax.scan`` (fixed steps,
            no early stopping) instead of the ``while_loop``-based sequential
            path.  Required on Linux HPC if num_workers > 1 where ``multiprocessing``
            spawn can be unreliable with JAX.  Default is ``True``.
        num_buckets : int
            ``1`` (default) pads all pairs to the global pharmacophore-count
            maximum — lowest overhead for typical use.  Values > 1 sort pairs
            by ``(max(ref,fit), min(ref,fit))`` and process each bucket
            separately with reduced per-bucket padding, which can be
            beneficial for large heterogeneous molecule sets.
        verbose : bool
            Print scores per pair.

        Returns
        -------
        scores : np.ndarray
            Shape: (N,).
        aligned_anchors_list : list of np.ndarray
            Aligned fit pharmacophore anchors (unpadded) for each pair.
        aligned_vectors_list : list of np.ndarray
            Aligned fit pharmacophore vectors (unpadded) for each pair.
        """
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
