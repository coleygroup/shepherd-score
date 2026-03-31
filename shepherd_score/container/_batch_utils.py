"""Utility functions for MoleculePairBatch."""

import multiprocessing as mp

import numpy as np

def _jax_worker_init():
    """Initialize worker process to use CPU for JAX."""
    import os
    os.environ['JAX_PLATFORMS'] = 'cpu'

    import jax
    jax.config.update('jax_platform_name', 'cpu')


def _pad_arrays(arrays, pad_value=0.0, dtype=np.float32):
    """Pad a list of arrays to the same max length along axis 0.

    Parameters
    ----------
    arrays : list of np.ndarray
        Arrays to pad; each shape (N,) or (N, k).
    pad_value : float
        Fill value for padding regions.
    dtype : numpy dtype
        Output dtype.

    Returns
    -------
    padded : list of np.ndarray
        Padded arrays, all shape (max_len,) or (max_len, k).
    masks : list of np.ndarray
        Binary float32 masks, each shape (max_len,).
    orig_lens : list of int
        Original lengths before padding.
    max_len : int
    """
    max_len = max(a.shape[0] for a in arrays)
    is_2d = arrays[0].ndim == 2
    cols = arrays[0].shape[1] if is_2d else None

    padded, masks, orig_lens = [], [], []
    for a in arrays:
        orig_len = a.shape[0]
        orig_lens.append(orig_len)
        if is_2d:
            p = np.full((max_len, cols), pad_value, dtype=dtype)
        else:
            p = np.full(max_len, pad_value, dtype=dtype)
        p[:orig_len] = a.astype(dtype)
        padded.append(p)

        mask = np.zeros(max_len, dtype=np.float32)
        mask[:orig_len] = 1.0
        masks.append(mask)

    return padded, masks, orig_lens, max_len


def _dispatch_parallel(pair_data, sort_keys, worker_fn, num_workers, shared_args):
    """Sort pairs by size, split into chunks, spawn workers, and collect results.

    Shared by all parallel alignment methods.  Pairs are sorted before chunking
    so each worker receives similarly-sized molecules, minimising per-worker
    padding waste and allowing JAX to reuse the same compiled kernel shape
    within a chunk.

    Must be called from a ``'__main__'``-guarded entry point (or an
    already-spawned process) because it uses ``mp.get_context('spawn')``.

    Parameters
    ----------
    pair_data : list
        Per-pair serialised data tuples (plain numpy arrays, no RDKit objects).
    sort_keys : array-like of shape (K, N), or None
        Rows of sort keys passed directly to ``np.lexsort``.  The **last** row
        is the primary sort key (highest priority), matching ``np.lexsort``
        convention.  Pass a 2-row array ``[minor_key, major_key]`` for a
        2-dimensional sort.  Pass ``None`` to skip sorting and chunk pairs in
        their original order (use when all arrays already have the same shape).
    worker_fn : callable
        Module-level worker function (must be picklable with 'spawn').
        Called as ``worker_fn((chunk, *shared_args))``.
    num_workers : int
        Number of worker processes.  Clamped to ``len(pair_data)``.
    shared_args : tuple
        Extra positional arguments appended to each work item after the chunk.

    Returns
    -------
    index_splits : list of list of int
        Original pair indices assigned to each worker chunk, in chunk order.
    chunk_results : list
        ``pool.map`` return values, one list per chunk.
    """
    n_pairs = len(pair_data)
    n_actual = min(num_workers, n_pairs)
    if sort_keys is None:
        sorted_order = list(range(n_pairs))
    else:
        sorted_order = np.lexsort(sort_keys).tolist()
    index_splits = [arr.tolist() for arr in np.array_split(sorted_order, n_actual)
                    if len(arr) > 0]
    work_items = [
        ([pair_data[i] for i in idx_list], *shared_args)
        for idx_list in index_splits
    ]
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=len(work_items), initializer=_jax_worker_init) as pool:
        chunk_results = pool.map(worker_fn, work_items)
    return index_splits, chunk_results


def _init_se3_batch(raw_refs, raw_fits, trans_centers_list, num_repeats):
    """Pre-compute SE(3) initialisations for all pairs using torch/numpy.

    Returns all initialisations as a single float32 array padded to the
    maximum number of repeats across pairs (repeats vary when
    ``trans_centers`` is provided because the count is ``10 * n_ref_atoms``).

    Parameters
    ----------
    raw_refs, raw_fits : list of np.ndarray
        Unpadded atom coordinate arrays for each pair.
    trans_centers_list : list of np.ndarray or None
        Per-pair translation initialisation centres, or ``None``.
    num_repeats : int
        Number of random SE(3) repeats when ``trans_centers`` is ``None``.

    Returns
    -------
    se3_all : np.ndarray of shape (n_pairs, max_repeats, 7), float32
    max_repeats : int
    """
    import torch
    from shepherd_score.alignment import (
        _initialize_se3_params,
        _initialize_se3_params_with_translations,
    )

    all_se3 = []
    for ref, fit, tc in zip(raw_refs, raw_fits, trans_centers_list):
        ref_t = torch.tensor(ref, dtype=torch.float32)
        fit_t = torch.tensor(fit, dtype=torch.float32)
        if tc is None:
            p = _initialize_se3_params(ref_t, fit_t, num_repeats).detach().numpy()
        else:
            p = _initialize_se3_params_with_translations(
                ref_t, fit_t, torch.tensor(tc, dtype=torch.float32),
                num_repeats_per_trans=10
            ).detach().numpy()
        all_se3.append(p)

    max_repeats = max(s.shape[0] for s in all_se3)
    padded = np.zeros((len(all_se3), max_repeats, 7), dtype=np.float32)
    for i, s in enumerate(all_se3):
        padded[i, :s.shape[0]] = s
    return padded, max_repeats


def _align_vol_shmap(
    pair_data_list, num_workers, num_repeats, lr, max_num_steps, verbose,
    n_buckets=None,
):
    """Parallel volumetric alignment via ``jax.shard_map`` across virtual CPU devices.

    ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` must be set
    **before JAX is first imported** so that ``len(jax.devices()) >= 1``.
    The number of devices actually used equals ``len(jax.devices())``.

    Pairs are sorted by ``max(len(ref), len(fit))`` and grouped into
    ``n_buckets`` equal-sized buckets before alignment.  Each bucket pads
    only to its own local maximum atom count, reducing wasted computation for
    molecule sets with varied sizes.  Because ``jit(shard_map(...))``
    automatically caches one compiled XLA kernel per unique input shape, each
    bucket reuses a cached kernel after the first call at that shape.

    Parameters
    ----------
    pair_data_list : list of (ref_pos, fit_pos, trans_centers) tuples
        Plain numpy arrays; ``trans_centers`` may be ``None``.
    num_workers : int
        Requested worker count (informational; actual parallelism is
        determined by the number of JAX devices in the process).
    num_repeats : int
        SE(3) initialisations per pair when ``trans_centers`` is ``None``.
    lr : float
    max_num_steps : int
    verbose : bool
    n_buckets : int or None
        Number of size buckets to split pairs into before padding.  Larger
        values reduce average padding waste at the cost of more JIT
        compilations on the first call.  ``None`` (default) sets
        ``n_buckets = len(jax.devices())``, matching the actual degree of
        parallelism.

    Returns
    -------
    list of (score, se3_transform, aligned_pts) tuples, one per pair.
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            'JAX is required for shard_map alignment. '
            'Install with: pip install "shepherd-score[jax]"'
        ) from exc

    from shepherd_score.alignment_jax import optimize_ROCS_overlay_jax_vol_shmap
    from shepherd_score.score.gaussian_overlap_jax import VAB_2nd_order_jax_mask

    n_pairs = len(pair_data_list)
    raw_refs = [d[0] for d in pair_data_list]
    raw_fits = [d[1] for d in pair_data_list]
    trans_centers_list = [d[2] for d in pair_data_list]

    # Pre-compute SE(3) initialisations: (n_pairs, max_repeats, 7)
    se3_init_all, _actual_repeats = _init_se3_batch(
        raw_refs, raw_fits, trans_centers_list, num_repeats
    )

    alpha = 0.81
    n_devices = len(jax.devices())
    if n_buckets is None:
        n_buckets = n_devices

    # Sort pairs by dominant padding dimension (max of ref/fit size) and group
    # into buckets so each bucket pads only to its local maximum.
    pair_sizes = np.array([max(len(raw_refs[i]), len(raw_fits[i])) for i in range(n_pairs)])
    sorted_order = np.argsort(pair_sizes, kind='stable')
    n_buckets_actual = min(n_buckets, n_pairs)
    bucket_splits = [
        arr.tolist()
        for arr in np.array_split(sorted_order, n_buckets_actual)
        if len(arr) > 0
    ]

    results = [None] * n_pairs

    for bucket_idx_list in bucket_splits:
        bucket_refs = [raw_refs[i] for i in bucket_idx_list]
        bucket_fits = [raw_fits[i] for i in bucket_idx_list]

        # Pad to bucket-local maximum atom count
        ref_padded, masks_ref, orig_refs_b, _ = _pad_arrays(bucket_refs)
        fit_padded, masks_fit, orig_fits_b, _ = _pad_arrays(bucket_fits)

        # Pre-compute self-overlaps for this bucket (invariant; once per loop)
        VAA_bucket = np.array([
            float(VAB_2nd_order_jax_mask(
                jnp.array(ref_padded[j]), jnp.array(ref_padded[j]),
                jnp.array(masks_ref[j]), jnp.array(masks_ref[j]), alpha,
            )) for j in range(len(bucket_idx_list))
        ], dtype=np.float32)
        VBB_bucket = np.array([
            float(VAB_2nd_order_jax_mask(
                jnp.array(fit_padded[j]), jnp.array(fit_padded[j]),
                jnp.array(masks_fit[j]), jnp.array(masks_fit[j]), alpha,
            )) for j in range(len(bucket_idx_list))
        ], dtype=np.float32)

        # Pad pair count to a multiple of n_devices (dummy pairs carry zero masks)
        n_bucket = len(bucket_idx_list)
        pad_to = int(np.ceil(n_bucket / n_devices)) * n_devices

        def _pad_to_devices(arr, pad_val=0.0, _pad_to=pad_to):
            """Pad leading pair axis to `_pad_to` (multiple of n_devices)."""
            if _pad_to == len(arr):
                return arr
            extra = np.full(
                (_pad_to - len(arr),) + arr.shape[1:], pad_val, dtype=arr.dtype
            )
            return np.concatenate([arr, extra], axis=0)

        # Pass flat (pad_to, ...) arrays — shard_map distributes across devices
        # automatically via P('i') on axis 0. Do NOT pre-reshape to (D, B, ...).
        ref_all = _pad_to_devices(np.stack(ref_padded))
        fit_all = _pad_to_devices(np.stack(fit_padded))
        mr_all  = _pad_to_devices(np.stack(masks_ref))
        mf_all  = _pad_to_devices(np.stack(masks_fit))
        se3_all = _pad_to_devices(se3_init_all[bucket_idx_list])
        VAA_arr = _pad_to_devices(VAA_bucket)
        VBB_arr = _pad_to_devices(VBB_bucket)

        aligned_b, se3_b, scores_b = optimize_ROCS_overlay_jax_vol_shmap(
            jnp.array(ref_all),
            jnp.array(fit_all),
            jnp.array(mr_all),
            jnp.array(mf_all),
            jnp.array(VAA_arr),
            jnp.array(VBB_arr),
            jnp.array(se3_all),
            alpha, lr, max_num_steps,
        )

        # Outputs are already (pad_to, ...) — no reshape needed
        aligned_flat = np.array(aligned_b)   # (pad_to, M, 3)
        se3_flat     = np.array(se3_b)       # (pad_to, 4, 4)
        scores_flat  = np.array(scores_b)    # (pad_to,)

        for local_j, global_i in enumerate(bucket_idx_list):
            score = float(scores_flat[local_j])
            se3t  = se3_flat[local_j]
            apts  = aligned_flat[local_j][:orig_fits_b[local_j]]
            if verbose:
                print(f'Pair {global_i}: score={score:.4f}')
            results[global_i] = (score, se3t, apts)

    return results


def _align_vol_worker(args):
    """Worker for parallel JAX volumetric alignment.

    Parameters
    ----------
    args : tuple
        ``(pair_data_list, num_repeats, lr, max_num_steps, verbose)`` where
        ``pair_data_list`` is a list of ``(ref_pos, fit_pos, trans_centers)``
        tuples (all plain numpy arrays; ``trans_centers`` may be ``None``).

    Returns
    -------
    list of (score, se3_transform, aligned_pts) tuples
    """
    pair_data_list, num_repeats, lr, max_num_steps, verbose = args

    try:
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            'JAX is required for multiprocessing align_with_vol. '
            'Install it with: pip install "shepherd-score[jax]"'
        ) from exc

    from shepherd_score.alignment_jax import optimize_ROCS_overlay_jax_mask

    ref_padded, masks_ref, _orig_refs, _ = _pad_arrays([d[0] for d in pair_data_list])
    fit_padded, masks_fit, orig_fits, _ = _pad_arrays([d[1] for d in pair_data_list])
    trans_centers_list = [d[2] for d in pair_data_list]

    results = []
    for ref_pad, fit_pad, mask_ref, mask_fit, orig_fit, trans_centers in zip(
        ref_padded, fit_padded, masks_ref, masks_fit, orig_fits, trans_centers_list
    ):
        aligned_pts, se3_transform, score = optimize_ROCS_overlay_jax_mask(
            ref_points=jnp.array(ref_pad),
            fit_points=jnp.array(fit_pad),
            mask_ref=jnp.array(mask_ref),
            mask_fit=jnp.array(mask_fit),
            alpha=0.81,
            num_repeats=num_repeats,
            trans_centers=trans_centers,
            lr=lr,
            max_num_steps=max_num_steps,
            verbose=verbose,
        )
        results.append((
            float(np.array(score)),
            np.array(se3_transform),
            np.array(aligned_pts)[:orig_fit],
        ))

    return results


def _align_vol_esp_worker(args):
    """Worker for parallel JAX volumetric ESP alignment.

    Parameters
    ----------
    args : tuple
        ``(pair_data_list, lam, num_repeats, lr, max_num_steps, verbose)``
        where each element of ``pair_data_list`` is
        ``(ref_pos, fit_pos, ref_charges, fit_charges, trans_centers)``
        (all plain numpy arrays; ``trans_centers`` may be ``None``).

    Returns
    -------
    list of (score, se3_transform, aligned_pts) tuples
    """
    pair_data_list, lam, num_repeats, lr, max_num_steps, verbose = args

    try:
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            'JAX is required for multiprocessing align_with_vol_esp. '
            'Install it with: pip install "shepherd-score[jax]"'
        ) from exc

    from shepherd_score.alignment_jax import optimize_ROCS_esp_overlay_jax_mask

    ref_padded, masks_ref, _orig_refs, _ = _pad_arrays([d[0] for d in pair_data_list])
    fit_padded, masks_fit, orig_fits, _ = _pad_arrays([d[1] for d in pair_data_list])
    ref_ch_padded, _, _, _ = _pad_arrays([d[2] for d in pair_data_list])
    fit_ch_padded, _, _, _ = _pad_arrays([d[3] for d in pair_data_list])
    trans_centers_list = [d[4] for d in pair_data_list]

    results = []
    for ref_pad, fit_pad, ref_ch, fit_ch, mask_ref, mask_fit, orig_fit, trans_centers in zip(
        ref_padded, fit_padded, ref_ch_padded, fit_ch_padded,
        masks_ref, masks_fit, orig_fits, trans_centers_list,
    ):
        aligned_pts, se3_transform, score = optimize_ROCS_esp_overlay_jax_mask(
            ref_points=jnp.array(ref_pad),
            fit_points=jnp.array(fit_pad),
            ref_charges=jnp.array(ref_ch),
            fit_charges=jnp.array(fit_ch),
            mask_ref=jnp.array(mask_ref),
            mask_fit=jnp.array(mask_fit),
            alpha=0.81,
            lam=lam,
            num_repeats=num_repeats,
            trans_centers=trans_centers,
            lr=lr,
            max_num_steps=max_num_steps,
            verbose=verbose,
        )
        results.append((
            float(np.array(score)),
            np.array(se3_transform),
            np.array(aligned_pts)[:orig_fit],
        ))

    return results


def _align_surf_worker(args):
    """Worker for parallel surface alignment (JAX or PyTorch).

    Parameters
    ----------
    args : tuple
        ``(pair_data_list, alpha, num_repeats, lr, max_num_steps,
        use_jax, use_analytical, verbose)`` where each element of
        ``pair_data_list`` is ``(ref_surf, fit_surf, trans_centers)``
        (all plain numpy arrays; ``trans_centers`` may be ``None``).

    Returns
    -------
    list of (score, se3_transform, aligned_pts) tuples
    """
    (pair_data_list, alpha, num_repeats, lr, max_num_steps,
     use_jax, use_analytical, verbose) = args

    if use_jax:
        try:
            import jax.numpy as jnp
        except ImportError as exc:
            raise ImportError(
                'JAX is required for multiprocessing align_with_surf (use_jax=True). '
                'Install it with: pip install "shepherd-score[jax]"'
            ) from exc
        from shepherd_score.alignment_jax import optimize_ROCS_overlay_jax
    else:
        import torch
        from shepherd_score.alignment import (
            optimize_ROCS_overlay_analytical, optimize_ROCS_overlay)

    results = []
    for ref_surf, fit_surf, trans_centers in pair_data_list:
        if use_jax:
            tc = jnp.array(trans_centers) if trans_centers is not None else None
            aligned_pts, se3_transform, score = optimize_ROCS_overlay_jax(
                ref_points=jnp.array(ref_surf),
                fit_points=jnp.array(fit_surf),
                alpha=alpha,
                num_repeats=num_repeats,
                trans_centers=tc,
                lr=lr,
                max_num_steps=max_num_steps,
                verbose=verbose,
            )
            results.append((
                float(np.array(score)),
                np.array(se3_transform),
                np.array(aligned_pts),
            ))
        else:
            _fn = optimize_ROCS_overlay_analytical if use_analytical else optimize_ROCS_overlay
            tc = torch.tensor(trans_centers, dtype=torch.float32) if trans_centers is not None else None
            aligned_pts, se3_transform, score = _fn(
                ref_points=torch.tensor(ref_surf, dtype=torch.float32),
                fit_points=torch.tensor(fit_surf, dtype=torch.float32),
                alpha=alpha,
                num_repeats=num_repeats,
                trans_centers=tc,
                lr=lr,
                max_num_steps=max_num_steps,
                verbose=verbose,
            )
            results.append((
                float(score),
                se3_transform.numpy(),
                aligned_pts.numpy(),
            ))

    return results


def _align_esp_worker(args):
    """Worker for parallel surface ESP alignment (JAX or PyTorch).

    Parameters
    ----------
    args : tuple
        ``(pair_data_list, alpha, lam_scaled, num_repeats, lr, max_num_steps,
        use_jax, use_analytical, verbose)`` where each element of
        ``pair_data_list`` is
        ``(ref_surf, fit_surf, ref_esp, fit_esp, trans_centers)``
        (all plain numpy arrays; ``trans_centers`` may be ``None``).
        ``lam_scaled`` is the pre-scaled charge-weighting parameter
        (``LAM_SCALING * lam``).

    Returns
    -------
    list of (score, se3_transform, aligned_pts) tuples
    """
    (pair_data_list, alpha, lam_scaled, num_repeats, lr, max_num_steps,
     use_jax, use_analytical, verbose) = args

    if use_jax:
        try:
            import jax.numpy as jnp
        except ImportError as exc:
            raise ImportError(
                'JAX is required for multiprocessing align_with_esp (use_jax=True). '
                'Install it with: pip install "shepherd-score[jax]"'
            ) from exc
        from shepherd_score.alignment_jax import optimize_ROCS_esp_overlay_jax
    else:
        import torch
        from shepherd_score.alignment import (
            optimize_ROCS_esp_overlay_analytical, optimize_ROCS_esp_overlay)

    results = []
    for ref_surf, fit_surf, ref_esp, fit_esp, trans_centers in pair_data_list:
        if use_jax:
            tc = jnp.array(trans_centers) if trans_centers is not None else None
            aligned_pts, se3_transform, score = optimize_ROCS_esp_overlay_jax(
                ref_points=jnp.array(ref_surf),
                fit_points=jnp.array(fit_surf),
                ref_charges=jnp.array(ref_esp),
                fit_charges=jnp.array(fit_esp),
                alpha=alpha,
                lam=lam_scaled,
                num_repeats=num_repeats,
                trans_centers=tc,
                lr=lr,
                max_num_steps=max_num_steps,
                verbose=verbose,
            )
            results.append((
                float(np.array(score)),
                np.array(se3_transform),
                np.array(aligned_pts),
            ))
        else:
            _fn = optimize_ROCS_esp_overlay_analytical if use_analytical else optimize_ROCS_esp_overlay
            tc = torch.tensor(trans_centers, dtype=torch.float32) if trans_centers is not None else None
            aligned_pts, se3_transform, score = _fn(
                ref_points=torch.tensor(ref_surf, dtype=torch.float32),
                fit_points=torch.tensor(fit_surf, dtype=torch.float32),
                ref_charges=torch.tensor(ref_esp, dtype=torch.float32),
                fit_charges=torch.tensor(fit_esp, dtype=torch.float32),
                alpha=alpha,
                lam=lam_scaled,
                num_repeats=num_repeats,
                trans_centers=tc,
                lr=lr,
                max_num_steps=max_num_steps,
                verbose=verbose,
            )
            results.append((
                float(score),
                se3_transform.numpy(),
                aligned_pts.numpy(),
            ))

    return results


def _align_pharm_worker(args):
    """Worker for parallel JAX pharmacophore alignment.

    Parameters
    ----------
    args : tuple
        ``(pair_data_list, similarity, extended_points, only_extended,
        num_repeats, lr, max_num_steps, verbose)`` where each element of
        ``pair_data_list`` is
        ``(ref_types, fit_types, ref_ancs, fit_ancs, ref_vecs, fit_vecs,
        trans_centers, init_ref_ancs, init_fit_ancs)``
        (all plain numpy arrays; ``trans_centers`` may be ``None``).

    Returns
    -------
    list of (score, se3_transform, aligned_ancs, aligned_vecs) tuples
    """
    (pair_data_list, similarity, extended_points, only_extended,
     num_repeats, lr, max_num_steps, verbose) = args

    try:
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            'JAX is required for multiprocessing align_with_pharm. '
            'Install it with: pip install "shepherd-score[jax]"'
        ) from exc

    from shepherd_score.alignment_jax import optimize_pharm_overlay_jax_vectorized_mask

    DUMMY_TYPE = 8  # index of 'Dummy' in P_TYPES

    ref_ancs_padded, masks_ref, orig_refs, max_ref_len = _pad_arrays(
        [d[2] for d in pair_data_list])
    fit_ancs_padded, masks_fit, orig_fits, max_fit_len = _pad_arrays(
        [d[3] for d in pair_data_list])
    ref_vecs_padded, _, _, _ = _pad_arrays([d[4] for d in pair_data_list])
    fit_vecs_padded, _, _, _ = _pad_arrays([d[5] for d in pair_data_list])

    ref_types_padded, fit_types_padded = [], []
    for d, orig_ref, orig_fit in zip(pair_data_list, orig_refs, orig_fits):
        rtp = np.full(max_ref_len, DUMMY_TYPE, dtype=np.int32)
        rtp[:orig_ref] = d[0]
        ref_types_padded.append(rtp)
        ftp = np.full(max_fit_len, DUMMY_TYPE, dtype=np.int32)
        ftp[:orig_fit] = d[1]
        fit_types_padded.append(ftp)

    results = []
    for (rtp, ftp,
         ref_ancs_pad, fit_ancs_pad,
         ref_vecs_pad, fit_vecs_pad,
         mask_ref, mask_fit,
         orig_fit, d) in zip(
            ref_types_padded, fit_types_padded,
            ref_ancs_padded, fit_ancs_padded,
            ref_vecs_padded, fit_vecs_padded,
            masks_ref, masks_fit,
            orig_fits, pair_data_list,
    ):
        aligned_ancs, aligned_vecs, se3_transform, score = \
            optimize_pharm_overlay_jax_vectorized_mask(
                ref_pharms=jnp.array(rtp),
                fit_pharms=jnp.array(ftp),
                ref_anchors=jnp.array(ref_ancs_pad),
                fit_anchors=jnp.array(fit_ancs_pad),
                ref_vectors=jnp.array(ref_vecs_pad),
                fit_vectors=jnp.array(fit_vecs_pad),
                mask_ref=jnp.array(mask_ref),
                mask_fit=jnp.array(mask_fit),
                similarity=similarity,
                extended_points=extended_points,
                only_extended=only_extended,
                num_repeats=num_repeats,
                trans_centers=d[6],
                init_ref_anchors=d[7],
                init_fit_anchors=d[8],
                lr=lr,
                max_num_steps=max_num_steps,
                verbose=verbose,
            )
        results.append((
            float(np.array(score)),
            np.array(se3_transform),
            np.array(aligned_ancs)[:orig_fit],
            np.array(aligned_vecs)[:orig_fit],
        ))

    return results
