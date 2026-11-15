"""One avoid cloud shared by many pairs is carried as a single row (not one padded copy per pair).

Every layout must score the same: the shared row is broadcast to every pose at launch, so it has
to match the replicated per-pair layout on the kernel, through chunked launches, and end to end
(numba, the CPU pool and Triton), against the same clouds passed as distinct per-pair copies.
"""
import numpy as np
import pytest
import torch

from .._support import vca as V

D0 = 2.0


def _cuda():
    try:
        import triton  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


def _pairs(device="cpu", n=6):
    fits = V.SMILES[1:] + V.SMILES[:1]
    return [V.pair(V.SMILES[0], fits[i % len(fits)], device=device) for i in range(n)]


def _cloud(p):
    return V.near_cloud(p.ref_molec)


def test_bucket_keeps_one_row_for_a_shared_cloud():
    """Same array on every pair -> one row; distinct per-pair arrays -> one row per pair."""
    from shepherd_score.accel.batch import aligners as A
    from shepherd_score.accel._modes import SPECS
    from shepherd_score.accel.batch._bucket import plan_buckets
    from shepherd_score.accel.channels import z_order
    spec = SPECS["vol_avoid"]
    params = A.resolve_params(spec, {}, "test")
    names = A.concrete_channels(spec, params)
    dev = torch.device("cpu")
    for shared in (True, False):
        ps = _pairs()
        c = _cloud(ps[0])
        for p in ps:
            p.avoid_points = c if shared else c.copy()
        A.upload_channels(ps, names, dev)
        assert (len({id(p._avoid_pts_t) for p in ps}) == 1) == shared
        pspec, bch = A._pad_spec(spec, params, names, 4, False)
        bk = next(iter(plan_buckets(ps, pspec, dev)))
        ch = A.build_bucket(spec, bk.members, bk, bch, names, dev)["avoid"]
        assert ch.ref.shape[0] == (1 if shared else bk.K)
        assert torch.equal(ch.ref[0, :len(c)], torch.as_tensor(z_order(c)))   # read Z-ordered


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if _cuda() else []))
def test_shared_row_launch_matches_replicated(device, monkeypatch):
    """One shared row equals the same cloud replicated per pose, in one launch and in several."""
    import shepherd_score.accel.drivers.terms as T
    g = torch.Generator().manual_seed(0)
    K, M, Na = 24, 15, 300
    B = (torch.randn(K, M, 3, generator=g) * 1.5).to(device)
    AV = (torch.randn(1, Na, 3, generator=g) * 3.0).to(device)
    q = torch.nn.functional.normalize(torch.eye(4)[0] + 0.3 * torch.randn(K, 4, generator=g),
                                      dim=1).to(device)
    t = (0.5 * torch.randn(K, 3, generator=g)).to(device)
    mr = torch.randint(M // 2, M + 1, (K,), generator=g, dtype=torch.int32).to(device)
    nr = torch.tensor([Na - 7], dtype=torch.int32, device=device)
    kw = dict(min_dist=D0, NEED_GRAD=True)
    rep = T._avoid_chunked(AV.expand(K, -1, -1).contiguous(), B, q, t, nr.expand(K).contiguous(),
                           mr, 1, kw)
    assert (rep[0] > 0).float().mean() > 0.5, "fixture does not exercise the penalty"
    one = T._avoid_chunked(AV, B, q, t, nr, mr, 1, kw)
    monkeypatch.setattr(T, "_CHUNK", 5)                   # ragged slices: 5, 5, 5, 5, 4
    assert T._launch_step(1, (AV, B)) < K
    many = T._avoid_chunked(AV, B, q, t, nr, mr, 1, kw)
    for a, b, c in zip(rep, one, many):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-5)
        assert torch.allclose(b, c)


def _run(mode, device, shared, **kw):
    from shepherd_score.container import MoleculePairBatch
    ps = _pairs(device)
    c = _cloud(ps[0])
    b = MoleculePairBatch(ps)
    fn = getattr(b, f"align_with_{mode}")
    sc, _ = fn(c if shared else [c.copy() for _ in ps], **kw)
    tf = np.stack([torch.as_tensor(getattr(p, f"transform_{mode}")).cpu().numpy() for p in ps])
    return np.asarray(sc, float), tf.astype(float)


_BACKENDS = [("cpu", dict(backend="numba")), ("cpu", dict(backend="numba", num_workers=2))]
if _cuda():
    _BACKENDS.append(("cuda", dict(backend="triton")))


@pytest.mark.parametrize("mode", ["vol_avoid", "vol_color_avoid"])
@pytest.mark.parametrize("device,kw", _BACKENDS, ids=lambda x: str(x))
def test_shared_cloud_scores_like_per_pair_copies(mode, device, kw):
    s_sh, t_sh = _run(mode, device, True, **kw)
    s_pp, t_pp = _run(mode, device, False, **kw)
    assert np.allclose(s_sh, s_pp, atol=1e-5), np.abs(s_sh - s_pp).max()
    assert np.allclose(t_sh, t_pp, atol=1e-4), np.abs(t_sh - t_pp).max()


def test_pool_keeps_the_shared_cloud_one_tensor():
    """The pool reads a shared cloud once; its write-back leaves every pair the same tensor."""
    from shepherd_score.container import MoleculePairBatch
    ps = _pairs()
    b = MoleculePairBatch(ps)
    b.align_with_vol_avoid(_cloud(ps[0]), backend="numba", num_workers=2)
    assert len({id(p._avoid_pts_t) for p in ps}) == 1
