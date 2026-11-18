"""The fp32 poly5-exp2 kernels of ``kernels/cpu_soa.py`` must match the fp64 references in
``kernels/cpu.py`` (value, dQ, dT) to fp32 rounding, including padding, dummy pharm features
and types with no reference feature."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numba")

from shepherd_score.accel.kernels import cpu, cpu_soa  # noqa: E402

RTOL = 2e-5   # relative to the largest |entry| of each output, per row
P, N, M = 6, 23, 17


def _poses(rng):
    q = rng.standard_normal((P, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    t = rng.standard_normal((P, 3)) * 0.7
    return q.astype(np.float32), t.astype(np.float32)


def _counts(rng):
    Nr = rng.integers(N // 2, N + 1, P).astype(np.int64); Nr[0] = N
    Mr = rng.integers(M // 2, M + 1, P).astype(np.int64); Mr[0] = M
    return Nr, Mr


def _close(got, ref, what):
    for g, r, name in zip(got, ref, ("V", "dQ", "dT")):
        g = np.asarray(g, np.float64).reshape(P, -1); r = np.asarray(r, np.float64).reshape(P, -1)
        scale = np.maximum(np.abs(r).max(1, keepdims=True), 1e-6)
        err = (np.abs(g - r) / scale).max()
        assert err < RTOL, f"{what} {name}: max scaled error {err:.2e} >= {RTOL:.0e}"


def _cloud(rng, K, n, spread=1.6):
    return (rng.standard_normal((K, n, 3)) * spread).astype(np.float32)


def test_shape_kernel_matches_fp64():
    rng = np.random.default_rng(0)
    A, B = _cloud(rng, P, N), _cloud(rng, P, M); Nr, Mr = _counts(rng); q, t = _poses(rng)
    for alpha in (0.81, 1.5):
        got = cpu_soa.shape_closure(A, B, Nr, Mr, alpha)(q, t)
        ref = cpu._overlap_grad_kernel(A, B, q, t, Nr, Mr, alpha, True)
        _close(got, ref, f"shape alpha={alpha}")


def test_esp_kernel_matches_fp64():
    rng = np.random.default_rng(1)
    A, B = _cloud(rng, P, N), _cloud(rng, P, M); Nr, Mr = _counts(rng); q, t = _poses(rng)
    CA = (rng.standard_normal((P, N)) * 0.3).astype(np.float32)
    CB = (rng.standard_normal((P, M)) * 0.3).astype(np.float32)
    for lam in (0.1, 0.3):
        got = cpu_soa.esp_closure(A, B, CA, CB, Nr, Mr, 0.81, 1.0 / lam)(q, t)
        ref = cpu._overlap_grad_esp_kernel(A, B, CA, CB, q, t, Nr, Mr, 0.81, 1.0 / lam, True)
        _close(got, ref, f"esp lam={lam}")


def _tables(directionless):
    from shepherd_score.score.analytical_gradients._torch import build_lookup_tables
    al, Ks, cats = build_lookup_tables(torch.device("cpu"), torch.float64,
                                       directionless=directionless)
    return al.numpy().astype(np.float64), Ks.numpy().astype(np.float64), \
        cats.numpy().astype(np.int64)


def _types(rng, K, n, T, skip):
    """Random types drawn from half the table, so some types have no reference feature."""
    pool = rng.permutation(T)[: max(T // 2, 2)]
    ty = rng.choice(pool, (K, n)).astype(np.int64)
    if skip is not None:
        ty[:, 0] = skip
    return ty


def test_color_kernel_matches_fp64():
    rng = np.random.default_rng(2)
    al, Ks, cats = _tables(True); T = len(al)
    A, B = _cloud(rng, P, N), _cloud(rng, P, M); Nr, Mr = _counts(rng); q, t = _poses(rng)
    At, Bt = _types(rng, P, N, T, None), _types(rng, P, M, T, None)
    got = cpu_soa.color_closure(A, B, At, Bt, Nr, Mr, al, Ks, cats)(q, t)
    ref = cpu._pharm_color_grad_kernel(A, B, q, t, At, Bt, al, Ks, cats, Nr, Mr, True)
    _close(got, ref, "color")


def test_pharm_kernel_matches_fp64():
    rng = np.random.default_rng(3)
    al, Ks, cats = _tables(False); T = len(al)
    dummy = [k for k in range(T) if cats[k] == 3]
    A, B = _cloud(rng, P, N), _cloud(rng, P, M); Nr, Mr = _counts(rng); q, t = _poses(rng)
    VA = rng.standard_normal((P, N, 3)).astype(np.float32)   # not unit: both sides normalize
    VB = rng.standard_normal((P, M, 3)).astype(np.float32)
    VA /= np.linalg.norm(VA, axis=-1, keepdims=True); VB /= np.linalg.norm(VB, axis=-1, keepdims=True)
    TA = _types(rng, P, N, T, None)
    TB = _types(rng, P, M, T, dummy[0] if dummy else None)
    TA[:, :4] = TB[:, :4]                                    # guarantee same-type pairs
    got = cpu_soa.pharm_closure(A, VA, TA, B, VB, TB, Nr, Mr, al, Ks, cats)(q, t)
    ref = cpu._pharm_grad_dq_kernel(A, B, q, t, TA, TB, VA, VB, al, Ks, cats, Nr, Mr, True)
    _close(got, ref, "pharm")


def test_exp2_neg_accuracy():
    """``_exp2_neg`` is within fp32 rounding of ``2**-x`` over the range the kernels see."""
    from numba import njit

    @njit
    def ev(x):
        out = np.empty_like(x)
        for i in range(x.size):
            out[i] = cpu_soa._exp2_neg(x[i])
        return out

    x = np.linspace(0.0, 120.0, 200_001).astype(np.float32)
    got = ev(x).astype(np.float64); ref = np.exp2(-x.astype(np.float64))
    assert (np.abs(got - ref) / ref).max() < 5e-7
    assert ev(np.array([200.0, 1e30], np.float32)).max() < 2.0 ** -125   # flushes, no inf/nan
