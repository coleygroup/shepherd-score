"""numba CPU mirrors of the Triton overlap+grad and Adam kernels.

:mod:`~shepherd_score.accel.kernels.dispatch` routes CPU tensors here, so the batched
coarse-to-fine drivers run with no Triton/CUDA. Each kernel replicates its Triton twin op for
op (quaternion ``q=(w,x,y,z)``, fit ``B`` rotated against ref ``A``, ``Vab = K*sum
exp(-alpha/2 r^2)`` with ``K = pi^1.5/(2 alpha)^1.5``, analytic ``dVab/dq`` and ``dVab/dt``);
results match the GPU to rounding (``math.exp`` vs ``exp2``). The O(K*N*M)
work is a fused ``@njit(parallel=True)`` kernel, one pose per prange iteration; the O(K) Adam
bookkeeping stays in torch. ``NUMBA_NUM_THREADS`` sets the thread count.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from numba import njit, prange
from ._svml import enable_svml

from ...score.constants import COULOMB_SCALING, LAM_SCALING

enable_svml()   # before any kernel compiles; pip's icc_rt needs the explicit load

_K_PI = math.pi ** 1.5


def _np_cached(x, dtype):
    """``x`` as a numpy array of ``dtype``, converted once per tensor object.

    The eager fine loop hands the typed kernels the same type, table and count tensors every
    step. The copy is stashed on the tensor, keyed by storage pointer, shape and in-place
    version counter, so a rebuilt or mutated tensor converts again. Always a copy, never a view.
    """
    key = (x.data_ptr(), tuple(x.shape), x._version, dtype)
    hit = getattr(x, "_fss_np", None)
    if hit is not None and hit[0] == key:
        return hit[1]
    arr = x.detach().cpu().numpy().astype(dtype)
    try:
        x._fss_np = (key, arr)
    except AttributeError:                       # a tensor type that refuses attributes
        pass
    return arr


@njit(parallel=True, fastmath=True, cache=True)
def _overlap_grad_kernel(A, B, q, t, Nr, Mr, alpha, need_grad):
    """Fused overlap value + SE(3) gradient, one pose per prange iteration.

    A (K,N,3) ref, B (K,M,3) fit, q (K,4)=(w,x,y,z), t (K,3). Nr/Mr (K,) int real counts.
    Returns V (K,), dQ (K,4)=dVab/dq, dT (K,3)=dVab/dt. fp64 accumulation.
    Mirrors ``shape_triton._gauss_overlap_se3_tiled``.
    """
    K = A.shape[0]
    Kc = _K_PI / (2.0 * alpha) ** 1.5
    a2 = alpha / 2.0
    V = np.zeros(K, dtype=np.float64)
    dQ = np.zeros((K, 4), dtype=np.float64)
    dT = np.zeros((K, 3), dtype=np.float64)
    for k in prange(K):
        qr = q[k, 0]; qi = q[k, 1]; qj = q[k, 2]; qk_ = q[k, 3]
        tx = t[k, 0]; ty = t[k, 1]; tz = t[k, 2]
        # standard unit-quaternion -> rotation matrix (assumes |q|=1, as Triton does)
        r00 = 1.0 - 2.0 * (qj * qj + qk_ * qk_); r01 = 2.0 * (qi * qj - qk_ * qr); r02 = 2.0 * (qi * qk_ + qj * qr)
        r10 = 2.0 * (qi * qj + qk_ * qr); r11 = 1.0 - 2.0 * (qi * qi + qk_ * qk_); r12 = 2.0 * (qj * qk_ - qi * qr)
        r20 = 2.0 * (qi * qk_ - qj * qr); r21 = 2.0 * (qj * qk_ + qi * qr); r22 = 1.0 - 2.0 * (qi * qi + qj * qj)
        n_real = Nr[k]; m_real = Mr[k]
        Vacc = 0.0
        dTx = 0.0; dTy = 0.0; dTz = 0.0
        dQw = 0.0; dQx = 0.0; dQy = 0.0; dQz = 0.0
        for m in range(m_real):
            bx0 = B[k, m, 0]; by0 = B[k, m, 1]; bz0 = B[k, m, 2]
            bx = r00 * bx0 + r01 * by0 + r02 * bz0 + tx
            by = r10 * bx0 + r11 * by0 + r12 * bz0 + ty
            bz = r20 * bx0 + r21 * by0 + r22 * bz0 + tz
            fxj = 0.0; fyj = 0.0; fzj = 0.0
            for n in range(n_real):
                dx = A[k, n, 0] - bx; dy = A[k, n, 1] - by; dz = A[k, n, 2] - bz
                r2 = dx * dx + dy * dy + dz * dz
                g = Kc * math.exp(-a2 * r2)
                Vacc += g
                if need_grad:
                    c = alpha * g            # coeff = 2*half_alpha*g = alpha*g
                    fxj += c * dx; fyj += c * dy; fzj += c * dz
            if need_grad:
                dTx += fxj; dTy += fyj; dTz += fzj
                # dVab/dq, body-frame coords (bx0,by0,bz0); identical to the Triton tail
                dQw += fxj * (-2.0 * qk_ * by0 + 2.0 * qj * bz0) + fyj * (2.0 * qk_ * bx0 - 2.0 * qi * bz0) + fzj * (-2.0 * qj * bx0 + 2.0 * qi * by0)
                dQx += fxj * (2.0 * qj * by0 + 2.0 * qk_ * bz0) + fyj * (2.0 * qj * bx0 - 4.0 * qi * by0 - 2.0 * qr * bz0) + fzj * (2.0 * qk_ * bx0 + 2.0 * qr * by0 - 4.0 * qi * bz0)
                dQy += fxj * (-4.0 * qj * bx0 + 2.0 * qi * by0 + 2.0 * qr * bz0) + fyj * (2.0 * qi * bx0 + 2.0 * qk_ * bz0) + fzj * (-2.0 * qr * bx0 + 2.0 * qk_ * by0 - 4.0 * qj * bz0)
                dQz += fxj * (-4.0 * qk_ * bx0 - 2.0 * qr * by0 + 2.0 * qi * bz0) + fyj * (2.0 * qr * bx0 - 4.0 * qk_ * by0 + 2.0 * qj * bz0) + fzj * (2.0 * qi * bx0 + 2.0 * qj * by0)
        V[k] = Vacc
        dT[k, 0] = dTx; dT[k, 1] = dTy; dT[k, 2] = dTz
        dQ[k, 0] = dQw; dQ[k, 1] = dQx; dQ[k, 2] = dQy; dQ[k, 3] = dQz
    return V, dQ, dT


def overlap_score_grad_se3_batch(A, B, q, t, *, alpha: float = 0.81,
                                 N_real=None, M_real=None, NEED_GRAD: bool = True,
                                 BLOCK=None, num_warps=None, num_stages=None):
    """CPU drop-in for the Triton ``overlap_score_grad_se3_batch``. Returns (V, dQ, dT)
    as torch tensors on A.device with A.dtype. Extra kwargs ignored (GPU-only knobs)."""
    K, N_pad, _ = A.shape
    _, M_pad, _ = B.shape
    dev, dt = A.device, A.dtype
    An = np.ascontiguousarray(A.detach().cpu().numpy())
    Bn = np.ascontiguousarray(B.detach().cpu().numpy())
    qn = np.ascontiguousarray(q.detach().cpu().numpy())
    tn = np.ascontiguousarray(t.detach().cpu().numpy())
    Nr = (np.full(K, N_pad, np.int64) if N_real is None
          else N_real.detach().cpu().numpy().astype(np.int64))
    Mr = (np.full(K, M_pad, np.int64) if M_real is None
          else M_real.detach().cpu().numpy().astype(np.int64))
    V, dQ, dT = _overlap_grad_kernel(An, Bn, qn, tn, Nr, Mr, float(alpha), bool(NEED_GRAD))
    return (torch.as_tensor(V, device=dev, dtype=dt),
            torch.as_tensor(dQ, device=dev, dtype=dt),
            torch.as_tensor(dT, device=dev, dtype=dt))


@njit(parallel=True, fastmath=True, cache=True)
def _avoid_grad_kernel(A, B, q, t, Nr, Mr, min_dist, need_grad):
    """Linear hard-sphere avoid penalty value + SE(3) gradient, one pose per prange iteration.

    A is the fixed avoid cloud in the ref frame (not transformed), B the fit cloud transformed
    by (q, t); same quaternion->R build and dR/dq tail as ``_overlap_grad_kernel``.
    ``A_pen = sum_a sum_b relu((d0 - ||A_a - B'_b||) / d0)``; the force on B'_b is
    ``sum_a (1/(d0 d)) (A_a - B'_b)`` for ``0 < d < d0`` (the d~0 point is skipped, matching
    torch.cdist's zero gradient at coincidence). Returns V (K,), dQ (K,4), dT (K,3), fp64.
    """
    K = A.shape[0]
    inv_d0 = 1.0 / min_dist
    V = np.zeros(K, dtype=np.float64)
    dQ = np.zeros((K, 4), dtype=np.float64)
    dT = np.zeros((K, 3), dtype=np.float64)
    for k in prange(K):
        qr = q[k, 0]; qi = q[k, 1]; qj = q[k, 2]; qk_ = q[k, 3]
        tx = t[k, 0]; ty = t[k, 1]; tz = t[k, 2]
        r00 = 1.0 - 2.0 * (qj * qj + qk_ * qk_); r01 = 2.0 * (qi * qj - qk_ * qr); r02 = 2.0 * (qi * qk_ + qj * qr)
        r10 = 2.0 * (qi * qj + qk_ * qr); r11 = 1.0 - 2.0 * (qi * qi + qk_ * qk_); r12 = 2.0 * (qj * qk_ - qi * qr)
        r20 = 2.0 * (qi * qk_ - qj * qr); r21 = 2.0 * (qj * qk_ + qi * qr); r22 = 1.0 - 2.0 * (qi * qi + qj * qj)
        n_real = Nr[k]; m_real = Mr[k]
        Vacc = 0.0
        dTx = 0.0; dTy = 0.0; dTz = 0.0
        dQw = 0.0; dQx = 0.0; dQy = 0.0; dQz = 0.0
        for m in range(m_real):
            bx0 = B[k, m, 0]; by0 = B[k, m, 1]; bz0 = B[k, m, 2]
            bx = r00 * bx0 + r01 * by0 + r02 * bz0 + tx
            by = r10 * bx0 + r11 * by0 + r12 * bz0 + ty
            bz = r20 * bx0 + r21 * by0 + r22 * bz0 + tz
            fxj = 0.0; fyj = 0.0; fzj = 0.0
            for n in range(n_real):
                dx = A[k, n, 0] - bx; dy = A[k, n, 1] - by; dz = A[k, n, 2] - bz
                r2 = dx * dx + dy * dy + dz * dz
                d = math.sqrt(r2)
                if d < min_dist:
                    Vacc += (min_dist - d) * inv_d0
                    if need_grad and d > 1e-8:
                        c = inv_d0 / d       # (1/d0)/d ; f += c*dx == +dA_pen/dB' (fit-avoid coord)
                        fxj += c * dx; fyj += c * dy; fzj += c * dz
            if need_grad:
                dTx += fxj; dTy += fyj; dTz += fzj
                # dA_pen/dq, body-frame coords (bx0,by0,bz0); identical tail to the shape kernel
                dQw += fxj * (-2.0 * qk_ * by0 + 2.0 * qj * bz0) + fyj * (2.0 * qk_ * bx0 - 2.0 * qi * bz0) + fzj * (-2.0 * qj * bx0 + 2.0 * qi * by0)
                dQx += fxj * (2.0 * qj * by0 + 2.0 * qk_ * bz0) + fyj * (2.0 * qj * bx0 - 4.0 * qi * by0 - 2.0 * qr * bz0) + fzj * (2.0 * qk_ * bx0 + 2.0 * qr * by0 - 4.0 * qi * bz0)
                dQy += fxj * (-4.0 * qj * bx0 + 2.0 * qi * by0 + 2.0 * qr * bz0) + fyj * (2.0 * qi * bx0 + 2.0 * qk_ * bz0) + fzj * (-2.0 * qr * bx0 + 2.0 * qk_ * by0 - 4.0 * qj * bz0)
                dQz += fxj * (-4.0 * qk_ * bx0 - 2.0 * qr * by0 + 2.0 * qi * bz0) + fyj * (2.0 * qr * bx0 - 4.0 * qk_ * by0 + 2.0 * qj * bz0) + fzj * (2.0 * qi * bx0 + 2.0 * qj * by0)
        V[k] = Vacc
        dT[k, 0] = dTx; dT[k, 1] = dTy; dT[k, 2] = dTz
        dQ[k, 0] = dQw; dQ[k, 1] = dQx; dQ[k, 2] = dQy; dQ[k, 3] = dQz
    return V, dQ, dT


def overlap_score_grad_avoid_se3_batch(A, B, q, t, *, min_dist: float = 2.0,
                                       N_real=None, M_real=None, NEED_GRAD: bool = True,
                                       BLOCK=None, num_warps=None, num_stages=None):
    """CPU drop-in for the Triton ``overlap_score_grad_avoid_se3_batch``. A = fixed avoid points
    (ref frame), B = fit points (transformed). Returns (V, dQ, dT) as torch tensors on A.device
    with A.dtype. Extra kwargs (GPU-only knobs) ignored."""
    K, N_pad, _ = A.shape
    _, M_pad, _ = B.shape
    dev, dt = A.device, A.dtype
    An = np.ascontiguousarray(A.detach().cpu().numpy())
    Bn = np.ascontiguousarray(B.detach().cpu().numpy())
    qn = np.ascontiguousarray(q.detach().cpu().numpy())
    tn = np.ascontiguousarray(t.detach().cpu().numpy())
    Nr = (np.full(K, N_pad, np.int64) if N_real is None
          else N_real.detach().cpu().numpy().astype(np.int64))
    Mr = (np.full(K, M_pad, np.int64) if M_real is None
          else M_real.detach().cpu().numpy().astype(np.int64))
    V, dQ, dT = _avoid_grad_kernel(An, Bn, qn, tn, Nr, Mr, float(min_dist), bool(NEED_GRAD))
    return (torch.as_tensor(V, device=dev, dtype=dt),
            torch.as_tensor(dQ, device=dev, dtype=dt),
            torch.as_tensor(dT, device=dev, dtype=dt))


@torch.no_grad()
def fused_adam_qt_with_tangent_proj(q, t, dQ, dT, m_q, v_q, m_t, v_t, lr,
                                    beta1=0.9, beta2=0.999, eps=1e-8):
    """In-place Adam with quaternion tangent projection + renorm. Exact replica of the
    Triton ``_adam_qt_with_tangent_proj`` (no bias correction; eps inside the sqrt)."""
    radial = (dQ * q).sum(dim=1, keepdim=True)
    dq = dQ - q * radial
    m_q.mul_(beta1).add_(dq, alpha=1.0 - beta1)
    v_q.mul_(beta2).addcmul_(dq, dq, value=1.0 - beta2)
    q.add_(-lr * m_q / (v_q + eps).sqrt())
    m_t.mul_(beta1).add_(dT, alpha=1.0 - beta1)
    v_t.mul_(beta2).addcmul_(dT, dT, value=1.0 - beta2)
    t.add_(-lr * m_t / (v_t + eps).sqrt())
    q.div_(q.norm(dim=1, keepdim=True))


@torch.no_grad()
def fused_adam_qt(q, t, dQ, dT, m_q, v_q, m_t, v_t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    """In-place Adam (no tangent projection); CPU replica of Triton ``_adam_qt``."""
    m_q.mul_(beta1).add_(dQ, alpha=1.0 - beta1)
    v_q.mul_(beta2).addcmul_(dQ, dQ, value=1.0 - beta2)
    q.add_(-lr * m_q / (v_q + eps).sqrt())
    m_t.mul_(beta1).add_(dT, alpha=1.0 - beta1)
    v_t.mul_(beta2).addcmul_(dT, dT, value=1.0 - beta2)
    t.add_(-lr * m_t / (v_t + eps).sqrt())
    q.div_(q.norm(dim=1, keepdim=True))


@torch.no_grad()
def _batch_self_overlap(P_pad: torch.Tensor, N_real: torch.Tensor, alpha: float = 0.81):
    """Self-overlap VAA/VBB via the identity pose; CPU replica."""
    K = P_pad.shape[0]
    q_id = torch.zeros(K, 4, device=P_pad.device, dtype=P_pad.dtype); q_id[:, 0] = 1.0
    t_0 = torch.zeros(K, 3, device=P_pad.device, dtype=P_pad.dtype)
    V, _, _ = overlap_score_grad_se3_batch(P_pad, P_pad, q_id, t_0, alpha=alpha,
                                           N_real=N_real, M_real=N_real, NEED_GRAD=False)
    return V

#  ESP-weighted overlap (vol_esp / surf_esp): CPU twin of
#  esp_triton._gauss_overlap_esp_se3_tiled. V = K * sum exp(-a/2 r^2) *
#  exp(-(Ci-Cj)^2/lam); the charge weight is SE(3)-invariant, so the gradient is
#  the shape gradient with the weight folded into g. The two exps are fused.

@njit(parallel=True, fastmath=True, cache=True)
def _overlap_grad_esp_kernel(A, B, CA, CB, q, t, Nr, Mr, alpha, inv_lam, need_grad):
    K = A.shape[0]
    Kc = _K_PI / (2.0 * alpha) ** 1.5
    a2 = alpha / 2.0
    V = np.zeros(K, dtype=np.float64)
    dQ = np.zeros((K, 4), dtype=np.float64)
    dT = np.zeros((K, 3), dtype=np.float64)
    for k in prange(K):
        qr = q[k, 0]; qi = q[k, 1]; qj = q[k, 2]; qk_ = q[k, 3]
        tx = t[k, 0]; ty = t[k, 1]; tz = t[k, 2]
        r00 = 1.0 - 2.0 * (qj * qj + qk_ * qk_); r01 = 2.0 * (qi * qj - qk_ * qr); r02 = 2.0 * (qi * qk_ + qj * qr)
        r10 = 2.0 * (qi * qj + qk_ * qr); r11 = 1.0 - 2.0 * (qi * qi + qk_ * qk_); r12 = 2.0 * (qj * qk_ - qi * qr)
        r20 = 2.0 * (qi * qk_ - qj * qr); r21 = 2.0 * (qj * qk_ + qi * qr); r22 = 1.0 - 2.0 * (qi * qi + qj * qj)
        n_real = Nr[k]; m_real = Mr[k]
        Vacc = 0.0
        dTx = 0.0; dTy = 0.0; dTz = 0.0
        dQw = 0.0; dQx = 0.0; dQy = 0.0; dQz = 0.0
        for m in range(m_real):
            bx0 = B[k, m, 0]; by0 = B[k, m, 1]; bz0 = B[k, m, 2]; cbm = CB[k, m]
            bx = r00 * bx0 + r01 * by0 + r02 * bz0 + tx
            by = r10 * bx0 + r11 * by0 + r12 * bz0 + ty
            bz = r20 * bx0 + r21 * by0 + r22 * bz0 + tz
            fxj = 0.0; fyj = 0.0; fzj = 0.0
            for n in range(n_real):
                dx = A[k, n, 0] - bx; dy = A[k, n, 1] - by; dz = A[k, n, 2] - bz
                r2 = dx * dx + dy * dy + dz * dz
                dc = CA[k, n] - cbm
                g = Kc * math.exp(-a2 * r2 - dc * dc * inv_lam)
                Vacc += g
                if need_grad:
                    c = alpha * g
                    fxj += c * dx; fyj += c * dy; fzj += c * dz
            if need_grad:
                dTx += fxj; dTy += fyj; dTz += fzj
                dQw += fxj * (-2.0 * qk_ * by0 + 2.0 * qj * bz0) + fyj * (2.0 * qk_ * bx0 - 2.0 * qi * bz0) + fzj * (-2.0 * qj * bx0 + 2.0 * qi * by0)
                dQx += fxj * (2.0 * qj * by0 + 2.0 * qk_ * bz0) + fyj * (2.0 * qj * bx0 - 4.0 * qi * by0 - 2.0 * qr * bz0) + fzj * (2.0 * qk_ * bx0 + 2.0 * qr * by0 - 4.0 * qi * bz0)
                dQy += fxj * (-4.0 * qj * bx0 + 2.0 * qi * by0 + 2.0 * qr * bz0) + fyj * (2.0 * qi * bx0 + 2.0 * qk_ * bz0) + fzj * (-2.0 * qr * bx0 + 2.0 * qk_ * by0 - 4.0 * qj * bz0)
                dQz += fxj * (-4.0 * qk_ * bx0 - 2.0 * qr * by0 + 2.0 * qi * bz0) + fyj * (2.0 * qr * bx0 - 4.0 * qk_ * by0 + 2.0 * qj * bz0) + fzj * (2.0 * qi * bx0 + 2.0 * qj * by0)
        V[k] = Vacc
        dT[k, 0] = dTx; dT[k, 1] = dTy; dT[k, 2] = dTz
        dQ[k, 0] = dQw; dQ[k, 1] = dQx; dQ[k, 2] = dQy; dQ[k, 3] = dQz
    return V, dQ, dT


def overlap_score_grad_esp_se3_batch(A, B, charges_A, charges_B, q, t, *,
                                     alpha: float = 0.81, lam: float = 0.3,
                                     N_real=None, M_real=None, NEED_GRAD: bool = True,
                                     BLOCK=None, num_warps=None, num_stages=None):
    """CPU drop-in for the Triton ``overlap_score_grad_esp_se3_batch``."""
    K, N_pad, _ = A.shape
    _, M_pad, _ = B.shape
    dev, dt = A.device, A.dtype
    An = np.ascontiguousarray(A.detach().cpu().numpy())
    Bn = np.ascontiguousarray(B.detach().cpu().numpy())
    CAn = np.ascontiguousarray(charges_A.detach().cpu().numpy())
    CBn = np.ascontiguousarray(charges_B.detach().cpu().numpy())
    qn = np.ascontiguousarray(q.detach().cpu().numpy())
    tn = np.ascontiguousarray(t.detach().cpu().numpy())
    Nr = (np.full(K, N_pad, np.int64) if N_real is None
          else N_real.detach().cpu().numpy().astype(np.int64))
    Mr = (np.full(K, M_pad, np.int64) if M_real is None
          else M_real.detach().cpu().numpy().astype(np.int64))
    V, dQ, dT = _overlap_grad_esp_kernel(An, Bn, CAn, CBn, qn, tn, Nr, Mr,
                                         float(alpha), 1.0 / float(lam), bool(NEED_GRAD))
    return (torch.as_tensor(V, device=dev, dtype=dt),
            torch.as_tensor(dQ, device=dev, dtype=dt),
            torch.as_tensor(dT, device=dev, dtype=dt))

#  ShaEP surface-ESP agreement with its SE(3) gradient (vol_and_surf_esp and its Tversky
#  variant): CPU twin of esp_triton._esp_agreement_grad_kernel. Both directions of the
#  comparison in one pass, with the fit molecule's atoms and surface points moved by
#  R(q) x + t inside the kernel:
#      V = (sum_i g_i + sum_j g_j) / (n_surf + m_surf),  g = keep * exp(-(pe - esp)^2 / lam)
#  where i runs over the reference surface (field from the moved fit atoms) and j over the
#  moved fit surface (field from the reference atoms). The vdW+probe mask is piecewise
#  constant, so it contributes no gradient. fp64 accumulation; Abuf/Fbuf are (K, M_pad, 3)
#  scratch for the moved fit atoms and the forces on them.

@njit(parallel=True, fastmath=True, cache=True)
def _esp_agreement_grad_kernel(P1, PE1, NS1, A1, Q1, R1, NA1, P2, PE2, NS2, A2, Q2, R2, NA2,
                               q, t, inv_lam, coulomb, probe, Abuf, Fbuf, need_grad):
    K = q.shape[0]
    V = np.zeros(K, dtype=np.float64)
    dQ = np.zeros((K, 4), dtype=np.float64)
    dT = np.zeros((K, 3), dtype=np.float64)
    for k in prange(K):
        qr = q[k, 0]; qi = q[k, 1]; qj = q[k, 2]; qk_ = q[k, 3]
        tx = t[k, 0]; ty = t[k, 1]; tz = t[k, 2]
        r00 = 1.0 - 2.0 * (qj * qj + qk_ * qk_); r01 = 2.0 * (qi * qj - qk_ * qr); r02 = 2.0 * (qi * qk_ + qj * qr)
        r10 = 2.0 * (qi * qj + qk_ * qr); r11 = 1.0 - 2.0 * (qi * qi + qk_ * qk_); r12 = 2.0 * (qj * qk_ - qi * qr)
        r20 = 2.0 * (qi * qk_ - qj * qr); r21 = 2.0 * (qj * qk_ + qi * qr); r22 = 1.0 - 2.0 * (qi * qi + qj * qj)
        m2 = NA2[k]
        for m in range(m2):
            bx = A2[k, m, 0]; by = A2[k, m, 1]; bz = A2[k, m, 2]
            Abuf[k, m, 0] = r00 * bx + r01 * by + r02 * bz + tx
            Abuf[k, m, 1] = r10 * bx + r11 * by + r12 * bz + ty
            Abuf[k, m, 2] = r20 * bx + r21 * by + r22 * bz + tz
            Fbuf[k, m, 0] = 0.0; Fbuf[k, m, 1] = 0.0; Fbuf[k, m, 2] = 0.0
        # reference surface vs moved fit atoms
        e1 = 0.0
        for i in range(NS1[k]):
            px = P1[k, i, 0]; py = P1[k, i, 1]; pz = P1[k, i, 2]
            esp = 0.0
            blocked = False
            for m in range(m2):
                dx = px - Abuf[k, m, 0]; dy = py - Abuf[k, m, 1]; dz = pz - Abuf[k, m, 2]
                d = math.sqrt(dx * dx + dy * dy + dz * dz)
                if d < 1e-6:
                    d = 1e-6
                esp += Q2[k, m] / d
                if d < R2[k, m] + probe:
                    blocked = True
            esp *= coulomb
            if not blocked:
                diff = PE1[k, i] - esp
                g = math.exp(-(diff * diff) * inv_lam)
                e1 += g
                if need_grad:
                    c = 2.0 * inv_lam * g * diff * coulomb
                    for m in range(m2):
                        dx = px - Abuf[k, m, 0]; dy = py - Abuf[k, m, 1]; dz = pz - Abuf[k, m, 2]
                        d2 = dx * dx + dy * dy + dz * dz
                        d = math.sqrt(d2)
                        if d < 1e-6:
                            d = 1e-6
                            d2 = 1e-12
                        w = c * Q2[k, m] / (d2 * d)
                        Fbuf[k, m, 0] += w * dx; Fbuf[k, m, 1] += w * dy; Fbuf[k, m, 2] += w * dz
        gTx = 0.0; gTy = 0.0; gTz = 0.0
        gQw = 0.0; gQx = 0.0; gQy = 0.0; gQz = 0.0
        if need_grad:
            for m in range(m2):
                fx = Fbuf[k, m, 0]; fy = Fbuf[k, m, 1]; fz = Fbuf[k, m, 2]
                bx = A2[k, m, 0]; by = A2[k, m, 1]; bz = A2[k, m, 2]
                gTx += fx; gTy += fy; gTz += fz
                gQw += fx * (-2.0 * qk_ * by + 2.0 * qj * bz) + fy * (2.0 * qk_ * bx - 2.0 * qi * bz) + fz * (-2.0 * qj * bx + 2.0 * qi * by)
                gQx += fx * (2.0 * qj * by + 2.0 * qk_ * bz) + fy * (2.0 * qj * bx - 4.0 * qi * by - 2.0 * qr * bz) + fz * (2.0 * qk_ * bx + 2.0 * qr * by - 4.0 * qi * bz)
                gQy += fx * (-4.0 * qj * bx + 2.0 * qi * by + 2.0 * qr * bz) + fy * (2.0 * qi * bx + 2.0 * qk_ * bz) + fz * (-2.0 * qr * bx + 2.0 * qk_ * by - 4.0 * qj * bz)
                gQz += fx * (-4.0 * qk_ * bx - 2.0 * qr * by + 2.0 * qi * bz) + fy * (2.0 * qr * bx - 4.0 * qk_ * by + 2.0 * qj * bz) + fz * (2.0 * qi * bx + 2.0 * qj * by)
        # moved fit surface vs reference atoms
        e2 = 0.0
        m1 = NA1[k]
        for j in range(NS2[k]):
            sx = P2[k, j, 0]; sy = P2[k, j, 1]; sz = P2[k, j, 2]
            px = r00 * sx + r01 * sy + r02 * sz + tx
            py = r10 * sx + r11 * sy + r12 * sz + ty
            pz = r20 * sx + r21 * sy + r22 * sz + tz
            esp = 0.0
            Gx = 0.0; Gy = 0.0; Gz = 0.0
            blocked = False
            for n in range(m1):
                dx = px - A1[k, n, 0]; dy = py - A1[k, n, 1]; dz = pz - A1[k, n, 2]
                d = math.sqrt(dx * dx + dy * dy + dz * dz)
                if d < 1e-6:
                    d = 1e-6
                inv = 1.0 / d
                esp += Q1[k, n] * inv
                if need_grad:
                    w = Q1[k, n] * inv * inv * inv
                    Gx += w * dx; Gy += w * dy; Gz += w * dz
                if d < R1[k, n] + probe:
                    blocked = True
            esp *= coulomb
            if not blocked:
                diff = PE2[k, j] - esp
                g = math.exp(-(diff * diff) * inv_lam)
                e2 += g
                if need_grad:
                    c = -2.0 * inv_lam * g * diff * coulomb
                    fx = c * Gx; fy = c * Gy; fz = c * Gz
                    gTx += fx; gTy += fy; gTz += fz
                    gQw += fx * (-2.0 * qk_ * sy + 2.0 * qj * sz) + fy * (2.0 * qk_ * sx - 2.0 * qi * sz) + fz * (-2.0 * qj * sx + 2.0 * qi * sy)
                    gQx += fx * (2.0 * qj * sy + 2.0 * qk_ * sz) + fy * (2.0 * qj * sx - 4.0 * qi * sy - 2.0 * qr * sz) + fz * (2.0 * qk_ * sx + 2.0 * qr * sy - 4.0 * qi * sz)
                    gQy += fx * (-4.0 * qj * sx + 2.0 * qi * sy + 2.0 * qr * sz) + fy * (2.0 * qi * sx + 2.0 * qk_ * sz) + fz * (-2.0 * qr * sx + 2.0 * qk_ * sy - 4.0 * qj * sz)
                    gQz += fx * (-4.0 * qk_ * sx - 2.0 * qr * sy + 2.0 * qi * sz) + fy * (2.0 * qr * sx - 4.0 * qk_ * sy + 2.0 * qj * sz) + fz * (2.0 * qi * sx + 2.0 * qj * sy)
        nrm = float(NS1[k] + NS2[k])
        V[k] = (e1 + e2) / nrm
        dT[k, 0] = gTx / nrm; dT[k, 1] = gTy / nrm; dT[k, 2] = gTz / nrm
        dQ[k, 0] = gQw / nrm; dQ[k, 1] = gQx / nrm; dQ[k, 2] = gQy / nrm; dQ[k, 3] = gQz / nrm
    return V, dQ, dT


def esp_agreement_grad_se3_batch(ref_atoms, ref_charges, ref_radii, ref_points, ref_point_esp,
                                 fit_atoms, fit_charges, fit_radii, fit_points, fit_point_esp,
                                 n_surf, m_surf, n_atoms, m_atoms, q, t, *,
                                 probe_radius: float = 1.0, lam: float = 0.001,
                                 NEED_GRAD: bool = True):
    """CPU drop-in for the Triton ``esp_agreement_grad_se3_batch``: the ShaEP surface-ESP
    agreement ``V`` (K,) of the reference with the fit moved by ``(q, t)``, and ``dV/dq`` (K, 4),
    ``dV/dt`` (K, 3) (zeros when ``NEED_GRAD`` is False). The fit tensors are in the fit's own
    frame; ``n_surf``/``m_surf`` and ``n_atoms``/``m_atoms`` are the real surface-point and
    with-H atom counts. ``lam`` is the raw weighting parameter; ``LAM_SCALING`` is applied here."""
    dev, dt = q.device, ref_points.dtype
    f = lambda x: np.ascontiguousarray(x.detach().cpu().numpy())
    i = lambda x: x.detach().cpu().numpy().astype(np.int64)
    A2 = f(fit_atoms)
    K = int(q.shape[0])
    Abuf = np.empty((K, A2.shape[1], 3), dtype=np.float64)
    Fbuf = np.empty_like(Abuf)
    V, dQ, dT = _esp_agreement_grad_kernel(
        f(ref_points), f(ref_point_esp), i(n_surf), f(ref_atoms), f(ref_charges), f(ref_radii),
        i(n_atoms), f(fit_points), f(fit_point_esp), i(m_surf), A2, f(fit_charges), f(fit_radii),
        i(m_atoms), f(q), f(t), 1.0 / (LAM_SCALING * float(lam)), float(COULOMB_SCALING),
        float(probe_radius), Abuf, Fbuf, bool(NEED_GRAD))
    return (torch.as_tensor(V, device=dev, dtype=dt),
            torch.as_tensor(dQ, device=dev, dtype=q.dtype),
            torch.as_tensor(dT, device=dev, dtype=t.dtype))

#  Directionless pharmacophore "color" overlap value + quaternion gradient
#  (vol_color): the same same-type-only typed Gaussian as the pharm kernel, but
#  isotropic (w = 1, no vectors, no weight gradient) and emitting dV/dq
#  in-register like _overlap_grad_kernel, so the driver needs no rotation ->
#  quaternion projection. q is assumed unit (Adam renormalises each step).
#  A = ref anchors, B = fit anchors (rotated); At/Bt = ref/fit type indices;
#  dx = A - rot(B), the shape-kernel sign convention.

@njit(parallel=True, fastmath=True, cache=True)
def _pharm_color_grad_kernel(A, B, q, t, At, Bt, alphas, Ks, cats, Nr, Mr, need_grad):
    P = A.shape[0]
    O = np.zeros(P, dtype=np.float64)
    dQ = np.zeros((P, 4), dtype=np.float64)
    dT = np.zeros((P, 3), dtype=np.float64)
    for p in prange(P):
        qr = q[p, 0]; qi = q[p, 1]; qj = q[p, 2]; qk_ = q[p, 3]
        tx = t[p, 0]; ty = t[p, 1]; tz = t[p, 2]
        r00 = 1.0 - 2.0 * (qj * qj + qk_ * qk_); r01 = 2.0 * (qi * qj - qk_ * qr); r02 = 2.0 * (qi * qk_ + qj * qr)
        r10 = 2.0 * (qi * qj + qk_ * qr); r11 = 1.0 - 2.0 * (qi * qi + qk_ * qk_); r12 = 2.0 * (qj * qk_ - qi * qr)
        r20 = 2.0 * (qi * qk_ - qj * qr); r21 = 2.0 * (qj * qk_ + qi * qr); r22 = 1.0 - 2.0 * (qi * qi + qj * qj)
        n_real = Nr[p]; m_real = Mr[p]
        Oacc = 0.0
        dTx = 0.0; dTy = 0.0; dTz = 0.0
        dQw = 0.0; dQx = 0.0; dQy = 0.0; dQz = 0.0
        for m in range(m_real):  # FIT
            bx0 = B[p, m, 0]; by0 = B[p, m, 1]; bz0 = B[p, m, 2]
            ityp = Bt[p, m]
            if cats[ityp] == 3:   # dummy / pad -> skip
                continue
            ialpha = alphas[ityp]; iK = Ks[ityp]
            a2 = ialpha * 0.5
            bx = r00 * bx0 + r01 * by0 + r02 * bz0 + tx
            by = r10 * bx0 + r11 * by0 + r12 * bz0 + ty
            bz = r20 * bx0 + r21 * by0 + r22 * bz0 + tz
            fxj = 0.0; fyj = 0.0; fzj = 0.0
            for n in range(n_real):  # REF (same-type only)
                if At[p, n] != ityp:
                    continue
                dx = A[p, n, 0] - bx; dy = A[p, n, 1] - by; dz = A[p, n, 2] - bz
                r2 = dx * dx + dy * dy + dz * dz
                g = iK * math.exp(-a2 * r2)
                Oacc += g
                if need_grad:
                    c = ialpha * g
                    fxj += c * dx; fyj += c * dy; fzj += c * dz
            if need_grad:
                dTx += fxj; dTy += fyj; dTz += fzj
                dQw += fxj * (-2.0 * qk_ * by0 + 2.0 * qj * bz0) + fyj * (2.0 * qk_ * bx0 - 2.0 * qi * bz0) + fzj * (-2.0 * qj * bx0 + 2.0 * qi * by0)
                dQx += fxj * (2.0 * qj * by0 + 2.0 * qk_ * bz0) + fyj * (2.0 * qj * bx0 - 4.0 * qi * by0 - 2.0 * qr * bz0) + fzj * (2.0 * qk_ * bx0 + 2.0 * qr * by0 - 4.0 * qi * bz0)
                dQy += fxj * (-4.0 * qj * bx0 + 2.0 * qi * by0 + 2.0 * qr * bz0) + fyj * (2.0 * qi * bx0 + 2.0 * qk_ * bz0) + fzj * (-2.0 * qr * bx0 + 2.0 * qk_ * by0 - 4.0 * qj * bz0)
                dQz += fxj * (-4.0 * qk_ * bx0 - 2.0 * qr * by0 + 2.0 * qi * bz0) + fyj * (2.0 * qr * bx0 - 4.0 * qk_ * by0 + 2.0 * qj * bz0) + fzj * (2.0 * qi * bx0 + 2.0 * qj * by0)
        O[p] = Oacc
        dT[p, 0] = dTx; dT[p, 1] = dTy; dT[p, 2] = dTz
        dQ[p, 0] = dQw; dQ[p, 1] = dQx; dQ[p, 2] = dQy; dQ[p, 3] = dQz
    return O, dQ, dT


def pharm_color_score_grad_se3_batch(A, B, q, t, ref_types, fit_types, alphas, Ks, cats, *,
                                     N_real=None, M_real=None, NEED_GRAD: bool = True,
                                     BLOCK=None, num_warps=None, num_stages=None):
    """CPU drop-in for the Triton directionless-color value + quaternion-gradient kernel.
    A = ref anchors (P,N,3), B = fit anchors (P,M,3), q=(P,4), t=(P,3); ref/fit_types
    (P,N)/(P,M). Returns (O, dQ, dT) with dQ = dO/dq (quaternion), like the shape kernel."""
    P, N_pad, _ = A.shape
    _, M_pad, _ = B.shape
    dev, dt = A.device, A.dtype
    An = np.ascontiguousarray(A.detach().cpu().numpy())
    Bn = np.ascontiguousarray(B.detach().cpu().numpy())
    qn = np.ascontiguousarray(q.detach().cpu().numpy())
    tn = np.ascontiguousarray(t.detach().cpu().numpy())
    At = _np_cached(ref_types, np.int64)
    Bt = _np_cached(fit_types, np.int64)
    al = _np_cached(alphas, np.float64)
    Ksn = _np_cached(Ks, np.float64)
    cn = _np_cached(cats, np.int64)
    Nr = np.full(P, N_pad, np.int64) if N_real is None else _np_cached(N_real, np.int64)
    Mr = np.full(P, M_pad, np.int64) if M_real is None else _np_cached(M_real, np.int64)
    O, dQ, dT = _pharm_color_grad_kernel(An, Bn, qn, tn, At, Bt, al, Ksn, cn, Nr, Mr, bool(NEED_GRAD))
    return (torch.as_tensor(O, device=dev, dtype=dt),
            torch.as_tensor(dQ, device=dev, dtype=dt),
            torch.as_tensor(dT, device=dev, dtype=dt))

#  Directional pharmacophore overlap value + quaternion gradient (pharm): CPU twin
#  of pharm_triton._pharm_grad_dq_kernel. Typed Gaussians (per-type alpha/K/cat),
#  directional weight w (cat 1 = (clamp(D,0,1)+2)/3, cat 2 = (|D|+2)/3, else 1),
#  type-match and non-dummy (cat != 3) masking. Takes q (assumed unit) and emits
#  dV/dq directly, by applying the shape-kernel dR/dq tail twice: to the positional
#  force (sum_j aKwE*(rotfit-ref)) with the body-frame fit anchor, and to the weight
#  force (sum_j coeff*ref_vn) with the body-frame fit vector.

@njit(parallel=True, fastmath=True, cache=True)
def _pharm_grad_dq_kernel(A, B, q, t, At, Bt, RvA, FvB, alphas, Ks, cats, Nr, Mr, need_grad):
    P = A.shape[0]
    O = np.zeros(P, dtype=np.float64)
    dQ = np.zeros((P, 4), dtype=np.float64)
    dT = np.zeros((P, 3), dtype=np.float64)
    for p in prange(P):
        qr = q[p, 0]; qi = q[p, 1]; qj = q[p, 2]; qk_ = q[p, 3]
        tx = t[p, 0]; ty = t[p, 1]; tz = t[p, 2]
        r00 = 1.0 - 2.0 * (qj * qj + qk_ * qk_); r01 = 2.0 * (qi * qj - qk_ * qr); r02 = 2.0 * (qi * qk_ + qj * qr)
        r10 = 2.0 * (qi * qj + qk_ * qr); r11 = 1.0 - 2.0 * (qi * qi + qk_ * qk_); r12 = 2.0 * (qj * qk_ - qi * qr)
        r20 = 2.0 * (qi * qk_ - qj * qr); r21 = 2.0 * (qj * qk_ + qi * qr); r22 = 1.0 - 2.0 * (qi * qi + qj * qj)
        n_real = Nr[p]; m_real = Mr[p]
        Oacc = 0.0
        dTx = 0.0; dTy = 0.0; dTz = 0.0
        dQw = 0.0; dQx = 0.0; dQy = 0.0; dQz = 0.0
        for i in range(m_real):  # FIT
            fax0 = B[p, i, 0]; fay0 = B[p, i, 1]; faz0 = B[p, i, 2]
            ityp = Bt[p, i]
            icat = cats[ityp]
            if icat == 3:
                continue
            ialpha = alphas[ityp]; iK = Ks[ityp]
            fvx = FvB[p, i, 0]; fvy = FvB[p, i, 1]; fvz = FvB[p, i, 2]
            fnorm = math.sqrt(fvx * fvx + fvy * fvy + fvz * fvz)
            finv = 1.0 / (fnorm if fnorm > 1e-12 else 1e-12)
            fvxn = fvx * finv; fvyn = fvy * finv; fvzn = fvz * finv
            fatx = r00 * fax0 + r01 * fay0 + r02 * faz0 + tx
            faty = r10 * fax0 + r11 * fay0 + r12 * faz0 + ty
            fatz = r20 * fax0 + r21 * fay0 + r22 * faz0 + tz
            fvtx = r00 * fvxn + r01 * fvyn + r02 * fvzn
            fvty = r10 * fvxn + r11 * fvyn + r12 * fvzn
            fvtz = r20 * fvxn + r21 * fvyn + r22 * fvzn
            fxj = 0.0; fyj = 0.0; fzj = 0.0   # positional force (sum_j aKwE*(rotfit-ref))
            wfx = 0.0; wfy = 0.0; wfz = 0.0   # weight force (sum_j coeff*ref_vn)
            for j in range(n_real):  # REF (same-type only)
                if At[p, j] != ityp:
                    continue
                rax = A[p, j, 0]; ray = A[p, j, 1]; raz = A[p, j, 2]
                rvx = RvA[p, j, 0]; rvy = RvA[p, j, 1]; rvz = RvA[p, j, 2]
                rnorm = math.sqrt(rvx * rvx + rvy * rvy + rvz * rvz)
                rinv = 1.0 / (rnorm if rnorm > 1e-12 else 1e-12)
                rvxn = rvx * rinv; rvyn = rvy * rinv; rvzn = rvz * rinv
                dx = fatx - rax; dy = faty - ray; dz = fatz - raz   # rotfit - ref
                r2 = dx * dx + dy * dy + dz * dz
                E = math.exp(-ialpha * 0.5 * r2)
                D = fvtx * rvxn + fvty * rvyn + fvtz * rvzn
                if icat == 1:
                    Dcl = 0.0 if D < 0.0 else (1.0 if D > 1.0 else D)
                    w = (Dcl + 2.0) / 3.0
                elif icat == 2:
                    w = (abs(D) + 2.0) / 3.0
                else:
                    w = 1.0
                KwE = iK * w * E
                Oacc += KwE
                if need_grad:
                    aKwE = -ialpha * KwE
                    fxj += aKwE * dx; fyj += aKwE * dy; fzj += aKwE * dz
                    if icat == 1:
                        c = 1.0 if (D > 0.0 and D < 1.0) else 0.0
                    elif icat == 2:
                        c = 1.0 if D > 0.0 else (-1.0 if D < 0.0 else 0.0)
                    else:
                        c = 0.0
                    coeff = (1.0 / 3.0) * iK * E * c
                    wfx += coeff * rvxn; wfy += coeff * rvyn; wfz += coeff * rvzn
            if need_grad:
                dTx += fxj; dTy += fyj; dTz += fzj
                # positional dV/dq: shape tail with force (fxj,..) and body-frame fit anchor
                dQw += fxj * (-2.0 * qk_ * fay0 + 2.0 * qj * faz0) + fyj * (2.0 * qk_ * fax0 - 2.0 * qi * faz0) + fzj * (-2.0 * qj * fax0 + 2.0 * qi * fay0)
                dQx += fxj * (2.0 * qj * fay0 + 2.0 * qk_ * faz0) + fyj * (2.0 * qj * fax0 - 4.0 * qi * fay0 - 2.0 * qr * faz0) + fzj * (2.0 * qk_ * fax0 + 2.0 * qr * fay0 - 4.0 * qi * faz0)
                dQy += fxj * (-4.0 * qj * fax0 + 2.0 * qi * fay0 + 2.0 * qr * faz0) + fyj * (2.0 * qi * fax0 + 2.0 * qk_ * faz0) + fzj * (-2.0 * qr * fax0 + 2.0 * qk_ * fay0 - 4.0 * qj * faz0)
                dQz += fxj * (-4.0 * qk_ * fax0 - 2.0 * qr * fay0 + 2.0 * qi * faz0) + fyj * (2.0 * qr * fax0 - 4.0 * qk_ * fay0 + 2.0 * qj * faz0) + fzj * (2.0 * qi * fax0 + 2.0 * qj * fay0)
                # weight dV/dq: shape tail with force (wf*) and body-frame fit vector
                dQw += wfx * (-2.0 * qk_ * fvyn + 2.0 * qj * fvzn) + wfy * (2.0 * qk_ * fvxn - 2.0 * qi * fvzn) + wfz * (-2.0 * qj * fvxn + 2.0 * qi * fvyn)
                dQx += wfx * (2.0 * qj * fvyn + 2.0 * qk_ * fvzn) + wfy * (2.0 * qj * fvxn - 4.0 * qi * fvyn - 2.0 * qr * fvzn) + wfz * (2.0 * qk_ * fvxn + 2.0 * qr * fvyn - 4.0 * qi * fvzn)
                dQy += wfx * (-4.0 * qj * fvxn + 2.0 * qi * fvyn + 2.0 * qr * fvzn) + wfy * (2.0 * qi * fvxn + 2.0 * qk_ * fvzn) + wfz * (-2.0 * qr * fvxn + 2.0 * qk_ * fvyn - 4.0 * qj * fvzn)
                dQz += wfx * (-4.0 * qk_ * fvxn - 2.0 * qr * fvyn + 2.0 * qi * fvzn) + wfy * (2.0 * qr * fvxn - 4.0 * qk_ * fvyn + 2.0 * qj * fvzn) + wfz * (2.0 * qi * fvxn + 2.0 * qj * fvyn)
        O[p] = Oacc
        dT[p, 0] = dTx; dT[p, 1] = dTy; dT[p, 2] = dTz
        dQ[p, 0] = dQw; dQ[p, 1] = dQx; dQ[p, 2] = dQy; dQ[p, 3] = dQz
    return O, dQ, dT


def pharm_grad_dq_se3_batch(q, t, ref_types, fit_types, ref_anchors, fit_anchors,
                            ref_vectors, fit_vectors, alphas, Ks, cats, *,
                            N_real=None, M_real=None, NEED_GRAD: bool = True,
                            BLOCK=None, num_warps=None, num_stages=None):
    """CPU drop-in for the Triton directional pharm value + quaternion-gradient kernel. Takes q
    (not R) and returns (O, dQ, dT) with dQ = dO/dq, so no R->q projection is needed downstream."""
    P, N_pad, _ = ref_anchors.shape
    _, M_pad, _ = fit_anchors.shape
    dev, dt = ref_anchors.device, ref_anchors.dtype

    def _np(x):
        return np.ascontiguousarray(x.detach().cpu().numpy())
    RaA = _np(ref_anchors); FaB = _np(fit_anchors)
    RvA = _np(ref_vectors); FvB = _np(fit_vectors)
    RtA = _np_cached(ref_types, np.int64)
    FtB = _np_cached(fit_types, np.int64)
    qn = _np(q); tn = _np(t)
    al = _np_cached(alphas, np.float64)
    Ksn = _np_cached(Ks, np.float64)
    cn = _np_cached(cats, np.int64)
    Nr = np.full(P, N_pad, np.int64) if N_real is None else _np_cached(N_real, np.int64)
    Mr = np.full(P, M_pad, np.int64) if M_real is None else _np_cached(M_real, np.int64)
    O, dQ, dT = _pharm_grad_dq_kernel(RaA, FaB, qn, tn, RtA, FtB, RvA, FvB, al, Ksn, cn, Nr, Mr, bool(NEED_GRAD))
    return (torch.as_tensor(O, device=dev, dtype=dt),
            torch.as_tensor(dQ, device=dev, dtype=dt),
            torch.as_tensor(dT, device=dev, dtype=dt))


@torch.no_grad()
def _batch_self_overlap_esp(P_pad, charges, N_real, alpha: float = 0.81, lam: float = 0.3):
    """ESP self-overlap via the identity pose; CPU replica of _batch_self_overlap_esp."""
    K = P_pad.shape[0]
    q_id = torch.zeros(K, 4, device=P_pad.device, dtype=P_pad.dtype); q_id[:, 0] = 1.0
    t_0 = torch.zeros(K, 3, device=P_pad.device, dtype=P_pad.dtype)
    V, _, _ = overlap_score_grad_esp_se3_batch(P_pad, P_pad, charges, charges, q_id, t_0,
                                               alpha=alpha, lam=lam, N_real=N_real,
                                               M_real=N_real, NEED_GRAD=False)
    return V
