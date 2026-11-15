"""Kernel-level gates for the hard-sphere avoid penalty (``vol_avoid``, ``vol_color_avoid``).

The kernels skip avoid points beyond reach of the posed fit (bounding-sphere cull), skip the
``sqrt`` outside the shell (numba), read one shared avoid row per pair (``A_DIV``), and each cloud
is Z-ordered on its way in. None of that may change the penalty: every path is checked against a
float64 autograd brute force on pocket-shaped clouds where the cull actually removes points.
"""
import numpy as np
import pytest
import torch

from shepherd_score.accel.channels import z_order
from shepherd_score.alignment.utils.se3 import quaternions_to_SE3_batch

D0 = 2.0


def _brute(A, B, q, t, nr, mr):
    """Penalty and its (q, t) gradient in float64 by autograd; A per pose."""
    q = q.double().clone().requires_grad_()
    t = t.double().clone().requires_grad_()
    se3 = quaternions_to_SE3_batch(q, t)
    Bp = B.double() @ se3[:, :3, :3].transpose(1, 2) + se3[:, None, :3, 3]
    d = torch.cdist(A.double(), Bp)
    m = ((torch.arange(A.shape[1])[None, :, None] < nr[:, None, None])
         & (torch.arange(B.shape[1])[None, None, :] < mr[:, None, None]))
    V = (torch.relu((D0 - d) / D0) * m).sum((1, 2))
    gq, gt = torch.autograd.grad(V.sum(), (q, t))
    return V.detach(), gq, gt


def _shell(rng, pos, n, lo, hi):
    """Points whose nearest atom of ``pos`` is ``lo``..``hi`` A away (a pocket around it)."""
    out, c = [], pos.mean(0)
    while len(out) < n:
        x = c + rng.uniform(-hi - 3, hi + 3, size=(4096, 3))
        dm = np.linalg.norm(x[:, None] - pos[None], axis=-1).min(1)
        out.extend(x[(dm > lo) & (dm < hi)])
    return np.asarray(out[:n], np.float32)


def _case(pairs=6, P=4, M=18, K=400, lo=0.5, hi=7.0, seed=0):
    """Per-pair pocket clouds, fits posed near them, ragged real counts."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    Bp = torch.randn(pairs, M, 3) * 1.6
    Ap = torch.tensor(np.stack([_shell(rng, Bp[i].numpy(), K, lo, hi) for i in range(pairs)]))
    npr = torch.randint(K // 2, K + 1, (pairs,), dtype=torch.int32)
    mpr = torch.randint(M // 2, M + 1, (pairs,), dtype=torch.int32)
    R = pairs * P
    q = torch.nn.functional.normalize(torch.eye(4)[0] + 0.2 * torch.randn(R, 4), dim=1)
    t = 0.8 * torch.randn(R, 3)
    return Ap, npr, Bp.repeat_interleave(P, 0), mpr.repeat_interleave(P), q, t, P


def _check(out, ref):
    for a, b in zip(out, ref):
        a = a.double().cpu()
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-4 * (1 + b.abs().max().item())), \
            (a - b).abs().max()


def _devices():
    out = ["cpu"]
    if torch.cuda.is_available():
        try:
            import triton  # noqa: F401
            out.append("cuda")
        except Exception:
            pass
    return out


def _kernel(device):
    if device == "cuda":
        from shepherd_score.accel.kernels import avoid_triton as m
    else:
        from shepherd_score.accel.kernels import cpu as m
    return m.overlap_score_grad_avoid_se3_batch


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("shell", [(0.5, 4.0), (1.0, 9.0), (1.0, 20.0)])
def test_culled_kernel_matches_brute_force(device, shell):
    """Replicated and per-pair (``A_DIV``) layouts, Z-ordered or not, equal the brute force."""
    Ap, npr, B, mr, q, t, P = _case(lo=shell[0], hi=shell[1])
    A = Ap.repeat_interleave(P, 0)
    ref = _brute(A, B, q, t, npr.repeat_interleave(P), mr)
    assert (ref[0] > 0).float().mean() > 0.5, "fixture does not exercise the penalty"
    fn = _kernel(device)
    to = lambda *xs: [x.to(device) for x in xs]
    Az = Ap.clone()                                    # each pair's real points Z-ordered
    for i, n in enumerate(npr.tolist()):
        Az[i, :n] = torch.from_numpy(z_order(Ap[i, :n].numpy()))
    _check(fn(*to(A, B, q, t), N_real=npr.repeat_interleave(P).to(device), M_real=mr.to(device)),
           ref)
    for cloud in (Ap, Az):
        _check(fn(*to(cloud, B, q, t), N_real=npr.to(device), M_real=mr.to(device), A_DIV=P),
               ref)


def test_z_order_is_a_permutation():
    a = _case()[0][0].numpy()
    z = z_order(a)
    assert np.array_equal(a[np.lexsort(a.T)], z[np.lexsort(z.T)])


def test_avoid_launch_chunks_keep_avoid_rows_aligned(monkeypatch):
    """Forced into several launches, the per-pair layout returns what one launch does."""
    import shepherd_score.accel.drivers.terms as T
    Ap, npr, B, mr, q, t, P = _case(pairs=7, P=4, K=60)
    kw = dict(min_dist=D0, NEED_GRAD=True)
    one = T._avoid_chunked(Ap, B, q, t, npr, mr, P, kw)
    monkeypatch.setattr(T, "_CHUNK", 2 * P)                     # 8 poses per launch
    assert T._launch_step(P, (Ap, B)) < q.shape[0]
    many = T._avoid_chunked(Ap, B, q, t, npr, mr, P, kw)
    for a, b in zip(one, many):
        assert torch.allclose(a, b)


def test_layout_rejects_mismatched_rows():
    from shepherd_score.accel.kernels import cpu
    Ap, npr, B, mr, q, t, P = _case(pairs=3, P=4, K=20)
    with pytest.raises(ValueError, match="A_DIV"):
        cpu.overlap_score_grad_avoid_se3_batch(Ap, B, q, t, N_real=npr, M_real=mr, A_DIV=P + 1)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("shared", [False, True])
def test_per_pose_layout_ignores_dedup(device, shared):
    """The avoid term keeps its own layout under ``dedup`` (fit replicated per pose, cloud per
    pair or shared), so it evaluates the same as without dedup and matches the brute force. It
    used to return the un-replicated fit with no ``_a_div`` and raise on the row count."""
    from shepherd_score.accel._modes import SPECS
    from shepherd_score.accel.drivers import terms as T
    from shepherd_score.accel.drivers.engine import _per_pose
    pairs, P = 5, 4
    Ap, npr, B, mr, q, t, _ = _case(pairs=pairs, P=P)
    if shared:
        Ap, npr = Ap[:1], npr[:1]
    term = next(tm for tm in SPECS["vol_avoid"].terms if tm.kernel == "avoid")
    to = lambda x: x.to(device)
    ti = T.TermInputs((to(Ap),), (to(B[::P].contiguous()),), to(npr), to(mr[::P].contiguous()),
                      None, None, {"avoid_min_dist": D0})
    A_full = Ap.expand(pairs, -1, -1) if shared else Ap
    ref = _brute(A_full.repeat_interleave(P, 0), B, q, t,
                 npr.expand(pairs).repeat_interleave(P) if shared else npr.repeat_interleave(P), mr)
    for dedup in (False, True):
        r, f, n, m, tp = _per_pose(term, ti, P, dedup=dedup)
        assert f[0].shape[0] == pairs * P and tp["_a_div"] == P
        out = T.evaluate(term, T.TermInputs(r, f, n, m, None, None, tp), to(q), to(t),
                         seeds_per_mol=P if dedup else 1)
        _check(out, ref)
