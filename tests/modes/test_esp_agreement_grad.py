"""The ShaEP surface-ESP agreement of ``vol_and_surf_esp`` and ``vol_and_surf_esp_tversky``:
its value + SE(3) gradient kernels, and the two modes' optimiser following the gradient of the
score it reports.

Molecules are synthetic (random atoms, charges, radii and surface points, with each surface's
ESP taken from its own atoms), so nothing here needs Open3D or a charge model. The vdW+probe
mask makes the agreement piecewise smooth; finite-difference checks keep only the directions
where differences at two step sizes agree.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from shepherd_score.score.constants import COULOMB_SCALING

LAM, PROBE = 0.001, 1.0


def _molecule(rng, n_atoms, n_surf):
    atoms = rng.normal(scale=1.6, size=(n_atoms, 3))
    charges = rng.normal(scale=0.25, size=n_atoms)
    charges -= charges.mean()
    radii = rng.uniform(1.2, 1.8, size=n_atoms)
    # surface points just outside a random atom's vdW+probe sphere
    owner = rng.integers(0, n_atoms, size=n_surf)
    u = rng.normal(size=(n_surf, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    pts = atoms[owner] + u * (radii[owner] + PROBE + 0.05)[:, None]
    d = np.linalg.norm(pts[:, None, :] - atoms[None, :, :], axis=2)
    esp = COULOMB_SCALING * (charges[None, :] / d).sum(1)
    return atoms, charges, radii, pts, esp


def _padded(rows, pad, width=None):
    out = np.zeros((len(rows), pad) + (() if width is None else (width,)))
    for k, r in enumerate(rows):
        out[k, :len(r)] = r
    return out


def _batch(K=6, seed=0, dtype=torch.float64):
    """Per-pose padded tensors in the term's channel order (cwh, partial, radii, surf, surf_esp)
    for both sides, the real counts, and random unit quaternions / translations."""
    rng = np.random.default_rng(seed)
    ref = [_molecule(rng, int(rng.integers(8, 15)), int(rng.integers(30, 61))) for _ in range(K)]
    fit = [_molecule(rng, int(rng.integers(8, 15)), int(rng.integers(30, 61))) for _ in range(K)]
    t = lambda x: torch.as_tensor(x, dtype=dtype)

    def side(mols):
        na = [len(m[0]) for m in mols]; ns = [len(m[3]) for m in mols]
        A, S = max(na), max(ns)
        chans = (t(_padded([m[0] for m in mols], A, 3)), t(_padded([m[1] for m in mols], A)),
                 t(_padded([m[2] for m in mols], A)), t(_padded([m[3] for m in mols], S, 3)),
                 t(_padded([m[4] for m in mols], S)))
        return chans, torch.tensor(ns, dtype=torch.int32), torch.tensor(na, dtype=torch.int32)

    (r, n_surf, n_atoms), (f, m_surf, m_atoms) = side(ref), side(fit)
    q = torch.nn.functional.normalize(t(rng.normal(size=(K, 4))), dim=1)
    tt = t(rng.normal(scale=1.0, size=(K, 3)))
    return r, f, n_surf, m_surf, n_atoms, m_atoms, q, tt


def _cpu(r, f, n_surf, m_surf, n_atoms, m_atoms, q, t, need_grad=True):
    from shepherd_score.accel.kernels.cpu import esp_agreement_grad_se3_batch
    return esp_agreement_grad_se3_batch(*r, *f, n_surf, m_surf, n_atoms, m_atoms, q, t,
                                        probe_radius=PROBE, lam=LAM, NEED_GRAD=need_grad)


def _rotate(x, q, t):
    from shepherd_score.accel.drivers._common import quaternion_to_rotation_matrix
    R = torch.stack([quaternion_to_rotation_matrix(qq) for qq in q])
    return torch.einsum("bij,bnj->bni", R, x) + t[:, None, :]


def test_value_equals_the_two_one_direction_comparisons():
    """The fused kernel's value is the reference ShaEP term: both one-direction comparisons
    (``score.electrostatic_scoring._esp_comparison``) on the transformed fit, averaged over both
    surfaces."""
    from shepherd_score.score.electrostatic_scoring import _esp_comparison
    r, f, n_surf, m_surf, n_atoms, m_atoms, q, t = _batch(dtype=torch.float32)
    V, _, _ = _cpu(r, f, n_surf, m_surf, n_atoms, m_atoms, q, t, need_grad=False)
    r64, f64 = [x.double() for x in r], [x.double() for x in f]
    cwh2_t, pts2_t = _rotate(f64[0], q.double(), t.double()), _rotate(f64[3], q.double(), t.double())
    ref = []
    for k in range(V.shape[0]):
        ns, ms, na, ma = (int(x[k]) for x in (n_surf, m_surf, n_atoms, m_atoms))
        e1 = _esp_comparison(r64[3][k, :ns], cwh2_t[k, :ma], f64[1][k, :ma], r64[4][k, :ns],
                             f64[2][k, :ma], probe_radius=PROBE, lam=LAM)
        e2 = _esp_comparison(pts2_t[k, :ms], r64[0][k, :na], r64[1][k, :na], f64[4][k, :ms],
                             r64[2][k, :na], probe_radius=PROBE, lam=LAM)
        ref.append((e1 + e2) / (ns + ms))
    ref = torch.stack(ref).to(V.dtype)
    assert torch.allclose(V, ref, atol=2e-6, rtol=0)
    assert (V > 0).all() and (V < 1).all()


def _fd_directional(r, f, counts, q, t, uq, ut, h):
    vp, _, _ = _cpu(r, f, *counts, torch.nn.functional.normalize(q + h * uq, dim=1), t + h * ut,
                    need_grad=False)
    vm, _, _ = _cpu(r, f, *counts, torch.nn.functional.normalize(q - h * uq, dim=1), t - h * ut,
                    need_grad=False)
    return (vp - vm) / (2 * h)


def test_cpu_gradient_matches_finite_differences():
    """dV/dq and dV/dt agree with central differences along random tangent directions (fp64)."""
    r, f, n_surf, m_surf, n_atoms, m_atoms, q, t = _batch(K=24, seed=1)
    counts = (n_surf, m_surf, n_atoms, m_atoms)
    _, dQ, dT = _cpu(r, f, *counts, q, t)
    g = torch.Generator().manual_seed(2)
    uq = torch.randn(q.shape, generator=g, dtype=q.dtype)
    uq = uq - q * (q * uq).sum(1, keepdim=True)
    ut = torch.randn(t.shape, generator=g, dtype=t.dtype)
    d_ana = (dQ * uq).sum(1) + (dT * ut).sum(1)
    f1 = _fd_directional(r, f, counts, q, t, uq, ut, 1e-5)
    f2 = _fd_directional(r, f, counts, q, t, uq, ut, 1e-6)
    smooth = (f1 - f2).abs() <= 1e-3 * f2.abs() + 1e-9
    assert smooth.float().mean() >= 0.75, "too few smooth directions to test"
    rel = (d_ana - f2).abs() / (f2.abs() + 1e-9)
    assert rel[smooth].max() < 1e-4, f"max rel err {rel[smooth].max():.2e}"


@pytest.mark.cuda
def test_triton_matches_numba():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    pytest.importorskip("triton")
    from shepherd_score.accel.kernels.esp_triton import esp_agreement_grad_se3_batch
    r, f, n_surf, m_surf, n_atoms, m_atoms, q, t = _batch(K=32, seed=3, dtype=torch.float32)
    V0, dQ0, dT0 = _cpu(r, f, n_surf, m_surf, n_atoms, m_atoms, q, t)
    cu = lambda x: x.cuda()
    V1, dQ1, dT1 = esp_agreement_grad_se3_batch(
        *map(cu, r), *map(cu, f), cu(n_surf), cu(m_surf), cu(n_atoms), cu(m_atoms), cu(q), cu(t),
        probe_radius=PROBE, lam=LAM, NEED_GRAD=True)
    assert torch.allclose(V1.cpu(), V0, atol=2e-6, rtol=0)
    g0 = torch.cat([dQ0, dT0], 1)
    g1 = torch.cat([dQ1.cpu(), dT1.cpu()], 1)
    rel = (g1 - g0).norm(dim=1) / (g0.norm(dim=1) + 1e-6)
    assert rel.max() < 1e-3, f"max rel grad err {rel.max():.2e}"


def test_every_registered_mode_differentiates_every_term():
    """No mode tracks a score whose terms its optimiser does not follow."""
    from shepherd_score.accel._modes import SPECS
    value_only = [(name, tm.kernel) for name, s in SPECS.items() for tm in s.terms if not tm.grad]
    assert value_only == []


def _combo_problem(mode, device, seed=4, K=8):
    from shepherd_score.accel._modes import SPECS
    from shepherd_score.accel.drivers.engine import assemble
    from .._support.engine import combo_chans
    r, f, n_surf, m_surf, n_atoms, m_atoms, _, _ = _batch(K=K, seed=seed, dtype=torch.float32)
    d = lambda x: x.to(device)
    cwh1, pc1, rad1, pts1, ptc1 = map(d, r)
    cwh2, pc2, rad2, pts2, ptc2 = map(d, f)
    chans = combo_chans(cwh1, cwh2, cwh1, cwh2, pts1, pts2, pc1, pc2, ptc1, ptc2, rad1, rad2,
                        d(n_atoms), d(m_atoms), d(n_atoms), d(m_atoms), d(n_surf), d(m_surf))
    spec = SPECS[mode]
    params = dict(spec.params)
    params["alpha"] = 0.81
    return assemble(spec, chans, params=params, num_seeds=spec.seeds)


@pytest.mark.parametrize("mode", ["vol_and_surf_esp", "vol_and_surf_esp_tversky"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_combo_step_follows_the_reported_score(mode, device, request):
    """The engine's fine-step gradient equals the derivative of the score the mode selects on
    and reports (both terms), on the directions where that score is smooth."""
    if device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        pytest.importorskip("triton")
    import shepherd_score.accel.drivers.engine as EN
    pr = _combo_problem(mode, torch.device(device))
    if device == "cuda":
        mm, old = torch.backends.cuda.matmul, torch.backends.cuda.matmul.allow_tf32
        request.addfinalizer(lambda: setattr(mm, "allow_tf32", old))    # do not leak it
        mm.allow_tf32 = False
    g = torch.Generator().manual_seed(5)
    rn = lambda *s: torch.randn(*s, generator=g).to(pr.q.device, pr.q.dtype)
    P = pr.q.shape[0]
    q = torch.nn.functional.normalize(pr.q + 0.3 * rn(P, 4), dim=1)
    t = pr.t + 0.5 * rn(P, 3)
    st = EN._State(q, t, 0.1)
    EN._step_generic(pr, st, update=False)
    uq = rn(P, 4); uq = uq - q * (q * uq).sum(1, keepdim=True)
    ut = rn(P, 3)
    d_eng = -((st.gq * uq).sum(1) + (st.gt * ut).sum(1)).double()

    def fd(h):
        fp = EN._score_poses(pr, torch.nn.functional.normalize(q + h * uq, dim=1), t + h * ut)
        fm = EN._score_poses(pr, torch.nn.functional.normalize(q - h * uq, dim=1), t - h * ut)
        return ((fp - fm) / (2 * h)).double()
    f1, f2 = fd(1e-3), fd(1e-4)
    smooth = (f1 - f2).abs() <= 0.01 * f2.abs() + 1e-3
    assert smooth.double().mean() >= 0.5, "too few smooth directions to test"
    rel = (d_eng - f1).abs() / (f1.abs() + 1e-3)
    assert (rel[smooth] < 0.01).double().mean() >= 0.95, (
        f"median rel err {rel[smooth].median():.2e}")
