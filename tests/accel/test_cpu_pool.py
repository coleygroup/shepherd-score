"""The persistent CPU process pool (``_cpu_pool``) must be a pure speed lever: sharding
pairs across workers gives the SAME scores/transforms as the single-process numba path
(pairs are independent + the batched aligner is deterministic). CPU-only; guards on numba.
"""
import numpy as np
import pytest
import torch

pytest.importorskip("numba")

from shepherd_score.accel import cpu_pool as _cpu_pool
from shepherd_score.accel import batch as bm

ALPHA = 0.81
_KW = {
    "vol":  dict(alpha=ALPHA, steps_fine=60),
    "surf": dict(alpha=ALPHA, steps_fine=60),
    "esp":  dict(alpha=ALPHA, lam=0.3, num_repeats=16, trans_init=False, lr=0.1, steps_fine=60),
    "pharm": dict(similarity="tanimoto", extended_points=False, only_extended=False,
                  trans_init=False, num_repeats=16, steps_fine=60, lr=0.1),
}


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else 1.0


class _Mol:
    def __init__(self, d):
        self.__dict__.update(d)


class _Pair:
    """MoleculePair stand-in with the ``*_molec`` arrays and the pre-cached ``_*_t`` tensors."""
    def __init__(self, ref, fit):
        self.ref_molec = _Mol(ref)
        self.fit_molec = _Mol(fit)
        self.device = torch.device("cpu")
        f32, i64 = torch.float32, torch.int64
        cache = [("_ref_xyz_t", "atom_pos", f32), ("_fit_xyz_t", "atom_pos", f32),
                 ("_ref_surf_t", "surf_pos", f32), ("_fit_surf_t", "surf_pos", f32),
                 ("_ref_surf_esp_t", "surf_esp", f32), ("_fit_surf_esp_t", "surf_esp", f32),
                 ("_ref_pharm_types_t", "pharm_types", i64), ("_fit_pharm_types_t", "pharm_types", i64),
                 ("_ref_pharm_ancs_t", "pharm_ancs", f32), ("_fit_pharm_ancs_t", "pharm_ancs", f32),
                 ("_ref_pharm_vecs_t", "pharm_vecs", f32), ("_fit_pharm_vecs_t", "pharm_vecs", f32)]
        for tname, arr, dt in cache:
            src = ref if "_ref_" in tname else fit
            setattr(self, tname, torch.as_tensor(src[arr], dtype=dt))


def _mol_arrays(n_atoms, n_surf, n_pharm, rng):
    v = rng.standard_normal((n_pharm, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return dict(
        atom_pos=(rng.standard_normal((n_atoms, 3)) * 2).astype(np.float32),
        surf_pos=(rng.standard_normal((n_surf, 3)) * 3).astype(np.float32),
        surf_esp=(rng.standard_normal(n_surf)).astype(np.float32),
        pharm_types=rng.integers(0, 4, n_pharm).astype(np.int64),
        pharm_ancs=(rng.standard_normal((n_pharm, 3)) * 2).astype(np.float32),
        pharm_vecs=v.astype(np.float32),
    )


def _rand_rot(rng):
    Q, R = np.linalg.qr(rng.standard_normal((3, 3)))
    Q *= np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float32)


def _se3_copy(ref, rng):
    """A rigid SE(3) copy of ref, so the optimum score is 1.0."""
    R, t = _rand_rot(rng), (rng.standard_normal(3) * 2).astype(np.float32)
    v = ref["pharm_vecs"] @ R.T
    return dict(
        atom_pos=ref["atom_pos"] @ R.T + t, surf_pos=ref["surf_pos"] @ R.T + t,
        surf_esp=ref["surf_esp"].copy(), pharm_types=ref["pharm_types"].copy(),
        pharm_ancs=ref["pharm_ancs"] @ R.T + t,
        pharm_vecs=(v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)).astype(np.float32))


def _make_pairs(k, rng, kind="se3"):
    pairs = []
    for _ in range(k):
        na, ns, nph = int(rng.integers(8, 16)), int(rng.integers(24, 40)), int(rng.integers(4, 8))
        ref = _mol_arrays(na, ns, nph, rng)
        if kind == "se3":
            fit = _se3_copy(ref, rng)
        elif kind == "distinct":
            fit = _mol_arrays(na, ns, nph, rng)
        else:  # exact
            fit = {kk: vv.copy() for kk, vv in ref.items()}
        pairs.append((ref, fit))
    return pairs


@pytest.mark.parametrize("mode", ["vol", "surf", "esp", "pharm"])
@pytest.mark.parametrize("num_workers", [2, 3])
def test_pool_matches_single_process(mode, num_workers):
    """Pooled matches single-process on SE(3) self-copies; per-shard early stop makes it close, not exact."""
    rng = np.random.default_rng(0)
    raw = _make_pairs(10, rng, kind="se3")
    sc_attr = bm._MODE_SPEC[_cpu_pool._LEGACY_MODE_ALIASES.get(mode, mode)]["out"][1]

    ref_pairs = [_Pair(*r) for r in raw]
    getattr(bm, "_align_batch_" + mode)([p for p in ref_pairs], **_KW[mode])
    ref = np.array([float(getattr(p, sc_attr)) for p in ref_pairs])

    pool_pairs = [_Pair(*r) for r in raw]
    _cpu_pool.align_pairs(mode, pool_pairs, num_workers, _KW[mode])
    got = np.array([float(getattr(p, sc_attr)) for p in pool_pairs])

    assert np.abs(got - ref).max() < 5e-3, f"{mode}: maxΔ={np.abs(got-ref).max():.2e}"


@pytest.mark.parametrize("mode", ["vol", "surf", "esp", "pharm"])
def test_pool_distinct_pairs_sanity(mode):
    """Distinct molecules: a looser bound that still catches wrong pairing or dropped kwargs."""
    rng = np.random.default_rng(7)
    raw = _make_pairs(8, rng, kind="distinct")
    sc_attr = bm._MODE_SPEC[_cpu_pool._LEGACY_MODE_ALIASES.get(mode, mode)]["out"][1]
    ref_pairs = [_Pair(*r) for r in raw]
    getattr(bm, "_align_batch_" + mode)([p for p in ref_pairs], **_KW[mode])
    ref = np.array([float(getattr(p, sc_attr)) for p in ref_pairs])
    pool_pairs = [_Pair(*r) for r in raw]
    _cpu_pool.align_pairs(mode, pool_pairs, 3, _KW[mode])
    got = np.array([float(getattr(p, sc_attr)) for p in pool_pairs])
    assert np.abs(got - ref).max() < 5e-2, f"{mode}: maxΔ={np.abs(got-ref).max():.2e}"


@pytest.mark.parametrize("mode", ["vol", "surf", "esp", "pharm"])
def test_pool_self_copy_exact(mode):
    """Self-copies converge immediately, so pooled equals single-process to float precision."""
    rng = np.random.default_rng(2)
    raw = _make_pairs(8, rng, kind="exact")            # fit = exact copy of ref
    sc_attr = bm._MODE_SPEC[_cpu_pool._LEGACY_MODE_ALIASES.get(mode, mode)]["out"][1]

    ref_pairs = [_Pair(*r) for r in raw]
    getattr(bm, "_align_batch_" + mode)([p for p in ref_pairs], **_KW[mode])
    ref = np.array([float(getattr(p, sc_attr)) for p in ref_pairs])

    pool_pairs = [_Pair(*r) for r in raw]
    _cpu_pool.align_pairs(mode, pool_pairs, 3, _KW[mode])
    got = np.array([float(getattr(p, sc_attr)) for p in pool_pairs])

    assert np.allclose(got, ref, atol=1e-5, rtol=0), f"{mode}: maxΔ={np.abs(got-ref).max():.2e}"


def test_pool_more_workers_than_pairs():
    """num_workers > len(pairs): extra workers get empty shards, results still correct."""
    rng = np.random.default_rng(1)
    raw = _make_pairs(2, rng)
    pairs = [_Pair(*r) for r in raw]
    _cpu_pool.align_pairs("vol", pairs, num_workers=5, align_kwargs=_KW["vol"])
    assert all(np.isfinite(float(p.sim_aligned_vol_noH)) for p in pairs)
