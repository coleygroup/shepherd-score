"""
Parallel (multi-device) JAX alignment via jax.shard_map.

Each public ``optimize_*_jax_*_shmap`` function accepts flat ``(total, ...)``
arrays (where ``total`` is the number of pairs padded to a multiple of
``len(jax.devices())``) and returns flat ``(total, ...)`` results.  The
leading axis is automatically distributed across devices by
``jax.shard_map`` with ``PartitionSpec('i')``; do **not** pre-reshape to
``(n_devices, B, ...)``.

``XLA_FLAGS=--xla_force_host_platform_device_count=N`` must be set **before
any JAX import** so that ``len(jax.devices()) == N``.
"""
import numpy as np

import jax
from jax import jit, vmap
from jax.sharding import Mesh, PartitionSpec as P

from shepherd_score.alignment._jax import (
    _per_pair_optimize_vol_mask_scan,
    _per_pair_optimize_vol_esp_mask_scan,
    _per_pair_optimize_surf_scan,
    _per_pair_optimize_surf_esp_scan,
    _per_pair_optimize_pharm_mask_scan_factory,
)

# ---------------------------------------------------------------------------
# Volumetric alignment
# ---------------------------------------------------------------------------

# Cache: (max_num_steps, n_devices) -> jit-compiled shard_map function
_shmap_vol_cache: dict = {}


def _make_shmap_vol_fn(max_num_steps: int, mesh: Mesh):
    """Build a ``jit(shard_map(vmap(...)))`` volumetric alignment function.

    Closes over ``max_num_steps`` so ``lax.scan`` sees a static Python int.
    Wraps the shard_map in ``jit`` so XLA compiles it once (shard_map alone
    does not trigger compilation the way pmap does).
    The result is stored in ``_shmap_vol_cache`` and reused for matching calls.
    """
    def _per_shard(ref_b, fit_b, mask_r_b, mask_f_b, se3_b, alpha, VAA_b, VBB_b, lr):
        """Processes one device's shard: shape (B, N/M, 3) etc."""
        def per_pair(ref, fit, mr, mf, s, VAA, VBB):
            return _per_pair_optimize_vol_mask_scan(
                ref, fit, mr, mf, s, alpha, VAA, VBB, lr, max_num_steps
            )
        return vmap(per_pair)(ref_b, fit_b, mask_r_b, mask_f_b, se3_b, VAA_b, VBB_b)

    return jit(jax.shard_map(
        _per_shard,
        mesh=mesh,
        in_specs=(P('i'), P('i'), P('i'), P('i'), P('i'), P(), P('i'), P('i'), P()),
        out_specs=(P('i'), P('i'), P('i')),
        check_vma=False,
    ))


def optimize_ROCS_overlay_jax_vol_shmap(
    ref_batch,
    fit_batch,
    mask_ref_batch,
    mask_fit_batch,
    VAA_batch,
    VBB_batch,
    se3_init_batch,
    alpha: float,
    lr: float,
    max_num_steps: int,
):
    """Volumetric alignment via ``shard_map`` + ``vmap`` across virtual CPU devices.

    All ``*_batch`` arrays use a flat leading axis of size ``total`` (i.e. the
    number of pairs padded to a multiple of ``len(jax.devices())``).  Unlike
    ``pmap``, ``shard_map`` automatically distributes the flat leading axis
    across devices; do **not** pre-reshape to ``(n_devices, B, ...)``.
    Pre-compute self-overlaps ``VAA``/``VBB`` and SE(3) initialisations outside
    this function (they are invariant to the optimisation loop).

    ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` must be set
    **before JAX is first imported** so that ``len(jax.devices()) == N``.

    Parameters
    ----------
    ref_batch       : (total, N, 3)  padded reference positions
    fit_batch       : (total, M, 3)  padded fit positions
    mask_ref_batch  : (total, N)
    mask_fit_batch  : (total, M)
    VAA_batch       : (total,)       pre-computed ref self-overlaps
    VBB_batch       : (total,)       pre-computed fit self-overlaps
    se3_init_batch  : (total, R, 7)  pre-initialised SE(3) params
    alpha           : float
    lr              : float
    max_num_steps   : int (Python int; determines compiled kernel)

    Returns
    -------
    aligned_pts   : (total, M, 3)
    se3_transform : (total, 4, 4)
    scores        : (total,)
    """
    devices = jax.devices()
    mesh = Mesh(np.array(devices), axis_names=('i',))
    cache_key = (max_num_steps, len(devices))
    if cache_key not in _shmap_vol_cache:
        _shmap_vol_cache[cache_key] = _make_shmap_vol_fn(max_num_steps, mesh)
    fn = _shmap_vol_cache[cache_key]
    return fn(
        ref_batch, fit_batch,
        mask_ref_batch, mask_fit_batch,
        se3_init_batch, alpha, VAA_batch, VBB_batch, lr,
    )


# ---------------------------------------------------------------------------
# Masked volumetric ESP alignment
# ---------------------------------------------------------------------------

_shmap_vol_esp_cache: dict = {}


def _make_shmap_vol_esp_fn(max_num_steps: int, mesh: Mesh):
    """Build a ``jit(shard_map(vmap(...)))`` masked volumetric ESP alignment function."""
    def _per_shard(ref_b, fit_b, ref_ch_b, fit_ch_b,
                   mask_r_b, mask_f_b, se3_b, alpha, lam, VAA_b, VBB_b, lr):
        """Processes one device's shard: shape (B, N/M, ...) etc."""
        def per_pair(ref, fit, ref_ch, fit_ch, mr, mf, s, VAA, VBB):
            return _per_pair_optimize_vol_esp_mask_scan(
                ref, fit, ref_ch, fit_ch, mr, mf, s, alpha, lam, VAA, VBB, lr, max_num_steps
            )
        return vmap(per_pair)(ref_b, fit_b, ref_ch_b, fit_ch_b,
                              mask_r_b, mask_f_b, se3_b, VAA_b, VBB_b)

    return jit(jax.shard_map(
        _per_shard,
        mesh=mesh,
        in_specs=(P('i'), P('i'), P('i'), P('i'),
                  P('i'), P('i'), P('i'), P(), P(), P('i'), P('i'), P()),
        out_specs=(P('i'), P('i'), P('i')),
        check_vma=False,
    ))


def optimize_ROCS_esp_overlay_jax_vol_esp_shmap(
    ref_batch,
    fit_batch,
    ref_charges_batch,
    fit_charges_batch,
    mask_ref_batch,
    mask_fit_batch,
    VAA_batch,
    VBB_batch,
    se3_init_batch,
    alpha: float,
    lam: float,
    lr: float,
    max_num_steps: int,
):
    """Masked volumetric ESP alignment via ``shard_map`` + ``vmap`` across virtual CPU devices.

    Parameters
    ----------
    ref_batch          : (total, N, 3)  padded reference positions
    fit_batch          : (total, M, 3)  padded fit positions
    ref_charges_batch  : (total, N, 1)  padded reference charges (column-shaped)
    fit_charges_batch  : (total, M, 1)  padded fit charges
    mask_ref_batch     : (total, N)
    mask_fit_batch     : (total, M)
    VAA_batch          : (total,)       pre-computed ref ESP self-overlaps
    VBB_batch          : (total,)       pre-computed fit ESP self-overlaps
    se3_init_batch     : (total, R, 7)  pre-initialised SE(3) params
    alpha, lam, lr     : float
    max_num_steps      : int (Python int; determines compiled kernel)

    Returns
    -------
    aligned_pts   : (total, M, 3)
    se3_transform : (total, 4, 4)
    scores        : (total,)
    """
    devices = jax.devices()
    mesh = Mesh(np.array(devices), axis_names=('i',))
    cache_key = (max_num_steps, len(devices))
    if cache_key not in _shmap_vol_esp_cache:
        _shmap_vol_esp_cache[cache_key] = _make_shmap_vol_esp_fn(max_num_steps, mesh)
    fn = _shmap_vol_esp_cache[cache_key]
    return fn(
        ref_batch, fit_batch,
        ref_charges_batch, fit_charges_batch,
        mask_ref_batch, mask_fit_batch,
        se3_init_batch, alpha, lam, VAA_batch, VBB_batch, lr,
    )


# ---------------------------------------------------------------------------
# Non-masked surface alignment
# ---------------------------------------------------------------------------

_shmap_surf_cache: dict = {}


def _make_shmap_surf_fn(max_num_steps: int, mesh: Mesh):
    """Build a ``jit(shard_map(vmap(...)))`` non-masked surface alignment function."""
    def _per_shard(ref_b, fit_b, se3_b, alpha, VAA_b, VBB_b, lr):
        """Processes one device's shard."""
        def per_pair(ref, fit, s, VAA, VBB):
            return _per_pair_optimize_surf_scan(
                ref, fit, s, alpha, VAA, VBB, lr, max_num_steps
            )
        return vmap(per_pair)(ref_b, fit_b, se3_b, VAA_b, VBB_b)

    return jit(jax.shard_map(
        _per_shard,
        mesh=mesh,
        in_specs=(P('i'), P('i'), P('i'), P(), P('i'), P('i'), P()),
        out_specs=(P('i'), P('i'), P('i')),
        check_vma=False,
    ))


def optimize_ROCS_overlay_jax_surf_shmap(
    ref_batch,
    fit_batch,
    VAA_batch,
    VBB_batch,
    se3_init_batch,
    alpha: float,
    lr: float,
    max_num_steps: int,
):
    """Non-masked surface alignment via ``shard_map`` + ``vmap`` across virtual CPU devices.

    Surface arrays are uniform size across all pairs so no masking is needed.

    Parameters
    ----------
    ref_batch      : (total, N, 3)  stacked reference surface positions
    fit_batch      : (total, M, 3)  stacked fit surface positions
    VAA_batch      : (total,)       pre-computed ref self-overlaps
    VBB_batch      : (total,)       pre-computed fit self-overlaps
    se3_init_batch : (total, R, 7)  pre-initialised SE(3) params
    alpha, lr      : float
    max_num_steps  : int (Python int; determines compiled kernel)

    Returns
    -------
    aligned_pts   : (total, M, 3)
    se3_transform : (total, 4, 4)
    scores        : (total,)
    """
    devices = jax.devices()
    mesh = Mesh(np.array(devices), axis_names=('i',))
    cache_key = (max_num_steps, len(devices))
    if cache_key not in _shmap_surf_cache:
        _shmap_surf_cache[cache_key] = _make_shmap_surf_fn(max_num_steps, mesh)
    fn = _shmap_surf_cache[cache_key]
    return fn(ref_batch, fit_batch, se3_init_batch, alpha, VAA_batch, VBB_batch, lr)


# ---------------------------------------------------------------------------
# Non-masked surface ESP alignment
# ---------------------------------------------------------------------------

_shmap_surf_esp_cache: dict = {}


def _make_shmap_surf_esp_fn(max_num_steps: int, mesh: Mesh):
    """Build a ``jit(shard_map(vmap(...)))`` non-masked surface ESP alignment function."""
    def _per_shard(ref_b, fit_b, ref_ch_b, fit_ch_b, se3_b, alpha, lam, VAA_b, VBB_b, lr):
        """Processes one device's shard."""
        def per_pair(ref, fit, ref_ch, fit_ch, s, VAA, VBB):
            return _per_pair_optimize_surf_esp_scan(
                ref, fit, ref_ch, fit_ch, s, alpha, lam, VAA, VBB, lr, max_num_steps
            )
        return vmap(per_pair)(ref_b, fit_b, ref_ch_b, fit_ch_b, se3_b, VAA_b, VBB_b)

    return jit(jax.shard_map(
        _per_shard,
        mesh=mesh,
        in_specs=(P('i'), P('i'), P('i'), P('i'), P('i'), P(), P(), P('i'), P('i'), P()),
        out_specs=(P('i'), P('i'), P('i')),
        check_vma=False,
    ))


def optimize_ROCS_esp_overlay_jax_surf_esp_shmap(
    ref_batch,
    fit_batch,
    ref_charges_batch,
    fit_charges_batch,
    VAA_batch,
    VBB_batch,
    se3_init_batch,
    alpha: float,
    lam: float,
    lr: float,
    max_num_steps: int,
):
    """Non-masked surface ESP alignment via ``shard_map`` + ``vmap`` across virtual CPU devices.

    Parameters
    ----------
    ref_batch          : (total, N, 3)  stacked reference surface positions
    fit_batch          : (total, M, 3)  stacked fit surface positions
    ref_charges_batch  : (total, N)     stacked reference ESP values
    fit_charges_batch  : (total, M)     stacked fit ESP values
    VAA_batch          : (total,)       pre-computed ref ESP self-overlaps
    VBB_batch          : (total,)       pre-computed fit ESP self-overlaps
    se3_init_batch     : (total, R, 7)  pre-initialised SE(3) params
    alpha, lam, lr     : float
    max_num_steps      : int

    Returns
    -------
    aligned_pts   : (total, M, 3)
    se3_transform : (total, 4, 4)
    scores        : (total,)
    """
    devices = jax.devices()
    mesh = Mesh(np.array(devices), axis_names=('i',))
    cache_key = (max_num_steps, len(devices))
    if cache_key not in _shmap_surf_esp_cache:
        _shmap_surf_esp_cache[cache_key] = _make_shmap_surf_esp_fn(max_num_steps, mesh)
    fn = _shmap_surf_esp_cache[cache_key]
    return fn(
        ref_batch, fit_batch,
        ref_charges_batch, fit_charges_batch,
        se3_init_batch, alpha, lam, VAA_batch, VBB_batch, lr,
    )


# ---------------------------------------------------------------------------
# Masked pharmacophore alignment
# ---------------------------------------------------------------------------

_shmap_pharm_cache: dict = {}


def _make_shmap_pharm_fn(max_num_steps: int, mesh: Mesh,
                         similarity: str, extended_points: bool, only_extended: bool):
    """Build a ``jit(shard_map(vmap(...)))`` masked pharmacophore alignment function."""
    _per_pair_fn = _per_pair_optimize_pharm_mask_scan_factory(
        similarity, extended_points, only_extended
    )

    def _per_shard(ref_pharms_b, fit_pharms_b,
                   ref_ancs_b, fit_ancs_b,
                   ref_vecs_b, fit_vecs_b,
                   mask_r_b, mask_f_b,
                   se3_b, ref_self_b, fit_self_b, lr):
        """Processes one device's shard."""
        def per_pair(rp, fp, ra, fa, rv, fv, mr, mf, s, rss, fss):
            return _per_pair_fn(
                rp, fp, ra, fa, rv, fv, mr, mf, s, rss, fss, lr, max_num_steps
            )
        return vmap(per_pair)(
            ref_pharms_b, fit_pharms_b,
            ref_ancs_b, fit_ancs_b,
            ref_vecs_b, fit_vecs_b,
            mask_r_b, mask_f_b,
            se3_b, ref_self_b, fit_self_b,
        )

    return jit(jax.shard_map(
        _per_shard,
        mesh=mesh,
        in_specs=(P('i'), P('i'),
                  P('i'), P('i'),
                  P('i'), P('i'),
                  P('i'), P('i'),
                  P('i'), P('i'), P('i'), P()),
        out_specs=(P('i'), P('i'), P('i'), P('i')),
        check_vma=False,
    ))


def optimize_pharm_overlay_jax_pharm_shmap(
    ref_pharms_batch,
    fit_pharms_batch,
    ref_anchors_batch,
    fit_anchors_batch,
    ref_vectors_batch,
    fit_vectors_batch,
    mask_ref_batch,
    mask_fit_batch,
    ref_self_batch,
    fit_self_batch,
    se3_init_batch,
    similarity: str,
    extended_points: bool,
    only_extended: bool,
    lr: float,
    max_num_steps: int,
):
    """Masked pharmacophore alignment via ``shard_map`` + ``vmap`` across virtual CPU devices.

    Parameters
    ----------
    ref_pharms_batch   : (total, N)     padded reference pharmacophore type indices (int32)
    fit_pharms_batch   : (total, M)     padded fit pharmacophore type indices
    ref_anchors_batch  : (total, N, 3)  padded reference anchor positions
    fit_anchors_batch  : (total, M, 3)  padded fit anchor positions
    ref_vectors_batch  : (total, N, 3)  padded reference direction vectors
    fit_vectors_batch  : (total, M, 3)  padded fit direction vectors
    mask_ref_batch     : (total, N)     binary masks
    mask_fit_batch     : (total, M)
    ref_self_batch     : (total,)       pre-computed ref self-overlaps
    fit_self_batch     : (total,)       pre-computed fit self-overlaps
    se3_init_batch     : (total, R, 7)  pre-initialised SE(3) params
    similarity         : str  ('tanimoto', 'tversky_ref', 'tversky_fit')
    extended_points    : bool
    only_extended      : bool
    lr                 : float
    max_num_steps      : int

    Returns
    -------
    aligned_anchors   : (total, M, 3)
    aligned_vectors   : (total, M, 3)
    se3_transform     : (total, 4, 4)
    scores            : (total,)
    """
    devices = jax.devices()
    mesh = Mesh(np.array(devices), axis_names=('i',))
    cache_key = (similarity, extended_points, only_extended, max_num_steps, len(devices))
    if cache_key not in _shmap_pharm_cache:
        _shmap_pharm_cache[cache_key] = _make_shmap_pharm_fn(
            max_num_steps, mesh, similarity, extended_points, only_extended
        )
    fn = _shmap_pharm_cache[cache_key]
    return fn(
        ref_pharms_batch, fit_pharms_batch,
        ref_anchors_batch, fit_anchors_batch,
        ref_vectors_batch, fit_vectors_batch,
        mask_ref_batch, mask_fit_batch,
        se3_init_batch, ref_self_batch, fit_self_batch, lr,
    )
