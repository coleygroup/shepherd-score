"""SoA fp32 CPU kernels for the fused CPU fine loop (cpu_fused.py), one per Gaussian term.

The AoS fp64 kernels in cpu.py are the reference: their layout (``A[k, n, :]``) and fp64
accumulation keep LLVM from vectorizing the inner O(N*M) loop, and ``math.exp`` vectorizes only
with Intel SVML (numba < 0.62 + icc_rt). These twins take coordinates SoA (``A[k, :, n]``,
contiguous in n), work in fp32, and never call ``math.exp``:

* Each Gaussian ``exp(-c * r2)`` is evaluated as ``exp2(-c * log2(e) * r2)``. For the shape and
  ESP kernels the ``sqrt(c * log2(e))`` factor is folded into the reference coordinates (and
  charges) once per align, so the inner loop is ``exp2(-|d'|^2)``; the prefactor and the
  alpha of the force are applied once per pose.
* ``exp2`` is :func:`_exp2_neg`: a degree-5 polynomial on the fractional part (max relative
  error ~1e-7, i.e. fp32 rounding) times ``2**n`` built from exponent bits. It is plain FMA
  arithmetic, so the loop vectorizes with or without SVML.
* The typed kernels (color / pharm) sort each pair's reference features by type once per
  align (``st[p, ty]..st[p, ty+1]`` ranges), so each fit feature only visits its same-type
  references with no type-match branch; padded and dummy fit features get the sentinel type T.

Values and gradients match cpu.py to fp32 rounding (tests/accel/test_cpu_soa_kernels.py).
"""
from __future__ import annotations

import math

import numpy as np
from llvmlite import ir as _ir
from numba import njit, prange
from numba import types as _nbt
from numba.extending import intrinsic

_K_PI = math.pi ** 1.5
_LOG2E = 1.4426950408889634


# exp2 without math.exp

@intrinsic
def _i32_as_f32(typingctx, x):
    """Reinterpret the bits of an int32 as a float32."""
    sig = _nbt.float32(_nbt.int32)

    def codegen(context, builder, signature, args):
        return builder.bitcast(args[0], _ir.FloatType())
    return sig, codegen


@njit(inline="always", fastmath=True)
def _exp2_neg(x):
    """``2**(-x)`` for ``x >= 0`` in fp32 (max relative error ~1e-7; flushes below 2**-126).

    ``y = -x = n + r`` with ``n = trunc(y)`` and ``r`` in (-1, 0]; ``2**r`` is a degree-5
    polynomial fitted for relative error and ``2**n`` is the float with exponent field n+127.
    """
    f = np.float32
    y = max(-x, f(-126.0))
    n = np.int32(y)
    r = y - f(n)
    p = f(1.0) + r * (f(0.6931435465812683) + r * (f(0.2401788979768753) + r * (
        f(0.05529404059052467) + r * (f(0.009202176705002785) + r * f(0.0009435108513571322)))))
    return p * _i32_as_f32((n + np.int32(127)) << np.int32(23))


# shape (vol / surf and the shape channel of the combined modes)

@njit(parallel=True, fastmath=True, cache=True)
def _shape_grad_kernel(A, B, q, t, Nr, Mr, s, Kc, fscale):
    """Shape overlap + SE(3) gradient. ``A`` (K,3,N) is pre-scaled by ``s``; ``B`` (K,3,M) and
    ``t`` are in real units. ``V = Kc * sum e`` and the force is ``fscale * sum e * d'`` with
    ``e = exp2(-|d'|^2)``, ``d' = A' - s * (R b + t)``."""
    K = A.shape[0]; f = np.float32
    V = np.zeros(K, np.float32); dQ = np.zeros((K, 4), np.float32)
    dT = np.zeros((K, 3), np.float32)
    one = f(1.0); two = f(2.0); four = f(4.0)
    for k in prange(K):
        qr = q[k, 0]; qi = q[k, 1]; qj = q[k, 2]; qk_ = q[k, 3]
        tx = t[k, 0]; ty = t[k, 1]; tz = t[k, 2]
        r00 = one - two * (qj * qj + qk_ * qk_); r01 = two * (qi * qj - qk_ * qr); r02 = two * (qi * qk_ + qj * qr)
        r10 = two * (qi * qj + qk_ * qr); r11 = one - two * (qi * qi + qk_ * qk_); r12 = two * (qj * qk_ - qi * qr)
        r20 = two * (qi * qk_ - qj * qr); r21 = two * (qj * qk_ + qi * qr); r22 = one - two * (qi * qi + qj * qj)
        n_real = Nr[k]; m_real = Mr[k]
        Vacc = f(0.0); dTx = f(0.0); dTy = f(0.0); dTz = f(0.0)
        dQw = f(0.0); dQx = f(0.0); dQy = f(0.0); dQz = f(0.0)
        for m in range(m_real):
            bx0 = B[k, 0, m]; by0 = B[k, 1, m]; bz0 = B[k, 2, m]
            bx = s * (r00 * bx0 + r01 * by0 + r02 * bz0 + tx)
            by = s * (r10 * bx0 + r11 * by0 + r12 * bz0 + ty)
            bz = s * (r20 * bx0 + r21 * by0 + r22 * bz0 + tz)
            fxj = f(0.0); fyj = f(0.0); fzj = f(0.0)
            for n in range(n_real):
                dx = A[k, 0, n] - bx; dy = A[k, 1, n] - by; dz = A[k, 2, n] - bz
                e = _exp2_neg(dx * dx + dy * dy + dz * dz)
                Vacc += e
                fxj += e * dx; fyj += e * dy; fzj += e * dz
            dTx += fxj; dTy += fyj; dTz += fzj
            dQw += fxj * (-two * qk_ * by0 + two * qj * bz0) + fyj * (two * qk_ * bx0 - two * qi * bz0) + fzj * (-two * qj * bx0 + two * qi * by0)
            dQx += fxj * (two * qj * by0 + two * qk_ * bz0) + fyj * (two * qj * bx0 - four * qi * by0 - two * qr * bz0) + fzj * (two * qk_ * bx0 + two * qr * by0 - four * qi * bz0)
            dQy += fxj * (-four * qj * bx0 + two * qi * by0 + two * qr * bz0) + fyj * (two * qi * bx0 + two * qk_ * bz0) + fzj * (-two * qr * bx0 + two * qk_ * by0 - four * qj * bz0)
            dQz += fxj * (-four * qk_ * bx0 - two * qr * by0 + two * qi * bz0) + fyj * (two * qr * bx0 - four * qk_ * by0 + two * qj * bz0) + fzj * (two * qi * bx0 + two * qj * by0)
        V[k] = Kc * Vacc
        dT[k, 0] = fscale * dTx; dT[k, 1] = fscale * dTy; dT[k, 2] = fscale * dTz
        dQ[k, 0] = fscale * dQw; dQ[k, 1] = fscale * dQx; dQ[k, 2] = fscale * dQy; dQ[k, 3] = fscale * dQz
    return V, dQ, dT


# ESP-weighted shape (vol_esp / surf_esp): shape Gaussian x charge Gaussian

@njit(parallel=True, fastmath=True, cache=True)
def _esp_grad_kernel(A, B, CA, CB, q, t, Nr, Mr, s, Kc, fscale):
    """As :func:`_shape_grad_kernel`, with charges ``CA`` (K,N) / ``CB`` (K,M) pre-scaled so
    each term is ``exp2(-(|d'|^2 + dc'^2))`` (the charge factor has no pose gradient)."""
    K = A.shape[0]; f = np.float32
    V = np.zeros(K, np.float32); dQ = np.zeros((K, 4), np.float32)
    dT = np.zeros((K, 3), np.float32)
    one = f(1.0); two = f(2.0); four = f(4.0)
    for k in prange(K):
        qr = q[k, 0]; qi = q[k, 1]; qj = q[k, 2]; qk_ = q[k, 3]
        tx = t[k, 0]; ty = t[k, 1]; tz = t[k, 2]
        r00 = one - two * (qj * qj + qk_ * qk_); r01 = two * (qi * qj - qk_ * qr); r02 = two * (qi * qk_ + qj * qr)
        r10 = two * (qi * qj + qk_ * qr); r11 = one - two * (qi * qi + qk_ * qk_); r12 = two * (qj * qk_ - qi * qr)
        r20 = two * (qi * qk_ - qj * qr); r21 = two * (qj * qk_ + qi * qr); r22 = one - two * (qi * qi + qj * qj)
        n_real = Nr[k]; m_real = Mr[k]
        Vacc = f(0.0); dTx = f(0.0); dTy = f(0.0); dTz = f(0.0)
        dQw = f(0.0); dQx = f(0.0); dQy = f(0.0); dQz = f(0.0)
        for m in range(m_real):
            bx0 = B[k, 0, m]; by0 = B[k, 1, m]; bz0 = B[k, 2, m]; cbm = CB[k, m]
            bx = s * (r00 * bx0 + r01 * by0 + r02 * bz0 + tx)
            by = s * (r10 * bx0 + r11 * by0 + r12 * bz0 + ty)
            bz = s * (r20 * bx0 + r21 * by0 + r22 * bz0 + tz)
            fxj = f(0.0); fyj = f(0.0); fzj = f(0.0)
            for n in range(n_real):
                dx = A[k, 0, n] - bx; dy = A[k, 1, n] - by; dz = A[k, 2, n] - bz
                dc = CA[k, n] - cbm
                e = _exp2_neg(dx * dx + dy * dy + dz * dz + dc * dc)
                Vacc += e
                fxj += e * dx; fyj += e * dy; fzj += e * dz
            dTx += fxj; dTy += fyj; dTz += fzj
            dQw += fxj * (-two * qk_ * by0 + two * qj * bz0) + fyj * (two * qk_ * bx0 - two * qi * bz0) + fzj * (-two * qj * bx0 + two * qi * by0)
            dQx += fxj * (two * qj * by0 + two * qk_ * bz0) + fyj * (two * qj * bx0 - four * qi * by0 - two * qr * bz0) + fzj * (two * qk_ * bx0 + two * qr * by0 - four * qi * bz0)
            dQy += fxj * (-four * qj * bx0 + two * qi * by0 + two * qr * bz0) + fyj * (two * qi * bx0 + two * qk_ * bz0) + fzj * (-two * qr * bx0 + two * qk_ * by0 - four * qj * bz0)
            dQz += fxj * (-four * qk_ * bx0 - two * qr * by0 + two * qi * bz0) + fyj * (two * qr * bx0 - four * qk_ * by0 + two * qj * bz0) + fzj * (two * qi * bx0 + two * qj * by0)
        V[k] = Kc * Vacc
        dT[k, 0] = fscale * dTx; dT[k, 1] = fscale * dTy; dT[k, 2] = fscale * dTz
        dQ[k, 0] = fscale * dQw; dQ[k, 1] = fscale * dQx; dQ[k, 2] = fscale * dQy; dQ[k, 3] = fscale * dQz
    return V, dQ, dT


# typed, directionless (vol_color and the other per-type channel modes)

@njit(parallel=True, fastmath=True, cache=True)
def _color_grad_kernel(A, st, B, Bt, q, t, Mr, c2, Kt, KA, T):
    """Typed overlap + gradient. ``A`` (P,3,N) is type-sorted with ranges ``st`` (P,T+1); fit
    feature m of type ``Bt[p, m]`` (``T`` = skip) visits only its type's range. Per type:
    ``c2 = alpha/2 * log2(e)``, ``Kt`` the prefactor and ``KA = Kt * alpha``."""
    P = A.shape[0]; f = np.float32
    V = np.zeros(P, np.float32); dQ = np.zeros((P, 4), np.float32)
    dT = np.zeros((P, 3), np.float32)
    one = f(1.0); two = f(2.0); four = f(4.0)
    for p in prange(P):
        qr = q[p, 0]; qi = q[p, 1]; qj = q[p, 2]; qk_ = q[p, 3]
        tx = t[p, 0]; ty_ = t[p, 1]; tz = t[p, 2]
        r00 = one - two * (qj * qj + qk_ * qk_); r01 = two * (qi * qj - qk_ * qr); r02 = two * (qi * qk_ + qj * qr)
        r10 = two * (qi * qj + qk_ * qr); r11 = one - two * (qi * qi + qk_ * qk_); r12 = two * (qj * qk_ - qi * qr)
        r20 = two * (qi * qk_ - qj * qr); r21 = two * (qj * qk_ + qi * qr); r22 = one - two * (qi * qi + qj * qj)
        Vacc = f(0.0); dTx = f(0.0); dTy = f(0.0); dTz = f(0.0)
        dQw = f(0.0); dQx = f(0.0); dQy = f(0.0); dQz = f(0.0)
        for m in range(Mr[p]):
            ty = Bt[p, m]
            if ty >= T:
                continue
            n0 = st[p, ty]; n1 = st[p, ty + 1]
            if n0 == n1:
                continue
            c = c2[ty]
            bx0 = B[p, 0, m]; by0 = B[p, 1, m]; bz0 = B[p, 2, m]
            bx = r00 * bx0 + r01 * by0 + r02 * bz0 + tx
            by = r10 * bx0 + r11 * by0 + r12 * bz0 + ty_
            bz = r20 * bx0 + r21 * by0 + r22 * bz0 + tz
            vs = f(0.0); fx = f(0.0); fy = f(0.0); fz = f(0.0)
            for n in range(n0, n1):
                dx = A[p, 0, n] - bx; dy = A[p, 1, n] - by; dz = A[p, 2, n] - bz
                e = _exp2_neg(c * (dx * dx + dy * dy + dz * dz))
                vs += e; fx += e * dx; fy += e * dy; fz += e * dz
            Vacc += Kt[ty] * vs
            ka = KA[ty]; fxj = ka * fx; fyj = ka * fy; fzj = ka * fz
            dTx += fxj; dTy += fyj; dTz += fzj
            dQw += fxj * (-two * qk_ * by0 + two * qj * bz0) + fyj * (two * qk_ * bx0 - two * qi * bz0) + fzj * (-two * qj * bx0 + two * qi * by0)
            dQx += fxj * (two * qj * by0 + two * qk_ * bz0) + fyj * (two * qj * bx0 - four * qi * by0 - two * qr * bz0) + fzj * (two * qk_ * bx0 + two * qr * by0 - four * qi * bz0)
            dQy += fxj * (-four * qj * bx0 + two * qi * by0 + two * qr * bz0) + fyj * (two * qi * bx0 + two * qk_ * bz0) + fzj * (-two * qr * bx0 + two * qk_ * by0 - four * qj * bz0)
            dQz += fxj * (-four * qk_ * bx0 - two * qr * by0 + two * qi * bz0) + fyj * (two * qr * bx0 - four * qk_ * by0 + two * qj * bz0) + fzj * (two * qi * bx0 + two * qj * by0)
        V[p] = Vacc
        dT[p, 0] = dTx; dT[p, 1] = dTy; dT[p, 2] = dTz
        dQ[p, 0] = dQw; dQ[p, 1] = dQx; dQ[p, 2] = dQy; dQ[p, 3] = dQz
    return V, dQ, dT


# typed, directional (pharm family)

@njit(parallel=True, fastmath=True, cache=True)
def _pharm_grad_kernel(A, RV, st, B, FV, Bt, q, t, Mr, cats, c2, Kt, al, T):
    """Directional pharmacophore overlap + gradient, the type-sorted twin of
    ``cpu._pharm_grad_dq_kernel``. ``RV`` / ``FV`` are unit vectors (SoA); ``cats`` gives each
    type's vector rule (1 = clamped, 2 = |cos|, else scalar)."""
    P = A.shape[0]; f = np.float32
    V = np.zeros(P, np.float32); dQ = np.zeros((P, 4), np.float32)
    dT = np.zeros((P, 3), np.float32)
    zero = f(0.0); one = f(1.0); two = f(2.0); four = f(4.0); third = f(1.0 / 3.0)
    for p in prange(P):
        qr = q[p, 0]; qi = q[p, 1]; qj = q[p, 2]; qk_ = q[p, 3]
        tx = t[p, 0]; ty_ = t[p, 1]; tz = t[p, 2]
        r00 = one - two * (qj * qj + qk_ * qk_); r01 = two * (qi * qj - qk_ * qr); r02 = two * (qi * qk_ + qj * qr)
        r10 = two * (qi * qj + qk_ * qr); r11 = one - two * (qi * qi + qk_ * qk_); r12 = two * (qj * qk_ - qi * qr)
        r20 = two * (qi * qk_ - qj * qr); r21 = two * (qj * qk_ + qi * qr); r22 = one - two * (qi * qi + qj * qj)
        Oacc = f(0.0); dTx = f(0.0); dTy = f(0.0); dTz = f(0.0)
        dQw = f(0.0); dQx = f(0.0); dQy = f(0.0); dQz = f(0.0)
        for i in range(Mr[p]):
            ty = Bt[p, i]
            if ty >= T:
                continue
            n0 = st[p, ty]; n1 = st[p, ty + 1]
            if n0 == n1:
                continue
            icat = cats[ty]; c = c2[ty]
            fax0 = B[p, 0, i]; fay0 = B[p, 1, i]; faz0 = B[p, 2, i]
            fvxn = FV[p, 0, i]; fvyn = FV[p, 1, i]; fvzn = FV[p, 2, i]
            fatx = r00 * fax0 + r01 * fay0 + r02 * faz0 + tx
            faty = r10 * fax0 + r11 * fay0 + r12 * faz0 + ty_
            fatz = r20 * fax0 + r21 * fay0 + r22 * faz0 + tz
            fvtx = r00 * fvxn + r01 * fvyn + r02 * fvzn
            fvty = r10 * fvxn + r11 * fvyn + r12 * fvzn
            fvtz = r20 * fvxn + r21 * fvyn + r22 * fvzn
            os_ = zero; fx = zero; fy = zero; fz = zero; wx = zero; wy = zero; wz = zero
            for j in range(n0, n1):
                rvx = RV[p, 0, j]; rvy = RV[p, 1, j]; rvz = RV[p, 2, j]
                dx = fatx - A[p, 0, j]; dy = faty - A[p, 1, j]; dz = fatz - A[p, 2, j]
                E = _exp2_neg(c * (dx * dx + dy * dy + dz * dz))
                D = fvtx * rvx + fvty * rvy + fvtz * rvz
                if icat == 1:
                    w = (min(max(D, zero), one) + two) * third
                    cc = one if (D > zero and D < one) else zero
                elif icat == 2:
                    w = (abs(D) + two) * third
                    cc = one if D > zero else (-one if D < zero else zero)
                else:
                    w = one; cc = zero
                wE = w * E
                os_ += wE; fx += wE * dx; fy += wE * dy; fz += wE * dz
                ce = E * cc
                wx += ce * rvx; wy += ce * rvy; wz += ce * rvz
            K = Kt[ty]
            Oacc += K * os_
            s1 = -al[ty] * K; fxj = s1 * fx; fyj = s1 * fy; fzj = s1 * fz
            s2 = K * third; wfx = s2 * wx; wfy = s2 * wy; wfz = s2 * wz
            dTx += fxj; dTy += fyj; dTz += fzj
            dQw += fxj * (-two * qk_ * fay0 + two * qj * faz0) + fyj * (two * qk_ * fax0 - two * qi * faz0) + fzj * (-two * qj * fax0 + two * qi * fay0)
            dQx += fxj * (two * qj * fay0 + two * qk_ * faz0) + fyj * (two * qj * fax0 - four * qi * fay0 - two * qr * faz0) + fzj * (two * qk_ * fax0 + two * qr * fay0 - four * qi * faz0)
            dQy += fxj * (-four * qj * fax0 + two * qi * fay0 + two * qr * faz0) + fyj * (two * qi * fax0 + two * qk_ * faz0) + fzj * (-two * qr * fax0 + two * qk_ * fay0 - four * qj * faz0)
            dQz += fxj * (-four * qk_ * fax0 - two * qr * fay0 + two * qi * faz0) + fyj * (two * qr * fax0 - four * qk_ * fay0 + two * qj * faz0) + fzj * (two * qi * fax0 + two * qj * fay0)
            dQw += wfx * (-two * qk_ * fvyn + two * qj * fvzn) + wfy * (two * qk_ * fvxn - two * qi * fvzn) + wfz * (-two * qj * fvxn + two * qi * fvyn)
            dQx += wfx * (two * qj * fvyn + two * qk_ * fvzn) + wfy * (two * qj * fvxn - four * qi * fvyn - two * qr * fvzn) + wfz * (two * qk_ * fvxn + two * qr * fvyn - four * qi * fvzn)
            dQy += wfx * (-four * qj * fvxn + two * qi * fvyn + two * qr * fvzn) + wfy * (two * qi * fvxn + two * qk_ * fvzn) + wfz * (-two * qr * fvxn + two * qk_ * fvyn - four * qj * fvzn)
            dQz += wfx * (-four * qk_ * fvxn - two * qr * fvyn + two * qi * fvzn) + wfy * (two * qr * fvxn - four * qk_ * fvyn + two * qj * fvzn) + wfz * (two * qi * fvxn + two * qj * fvyn)
        V[p] = Oacc
        dT[p, 0] = dTx; dT[p, 1] = dTy; dT[p, 2] = dTz
        dQ[p, 0] = dQw; dQ[p, 1] = dQx; dQ[p, 2] = dQy; dQ[p, 3] = dQz
    return V, dQ, dT


# per-align marshalling: (q, t) -> (V, dQ, dT) float32 closures used by cpu_fused

def to_soa(x_np):
    """(K,N,3) AoS -> (K,3,N) contiguous float32 (so the inner n-loop is contiguous)."""
    return np.ascontiguousarray(np.transpose(x_np, (0, 2, 1)), dtype=np.float32)


def _gauss_consts(alpha):
    """Prefactor, coordinate pre-scale and force scale for one shape-type Gaussian."""
    Kc = np.float32(_K_PI / (2.0 * alpha) ** 1.5)
    s = np.float32(math.sqrt(alpha * 0.5 * _LOG2E))
    return Kc, s, np.float32(float(Kc) * alpha / float(s))


def shape_closure(A, B, Nr, Mr, alpha):
    """Shape term over AoS ``A`` (K,N,3) / ``B`` (K,M,3)."""
    Kc, s, fscale = _gauss_consts(alpha)
    As = to_soa(A) * s; Bs = to_soa(B)
    return lambda q, t: _shape_grad_kernel(As, Bs, q, t, Nr, Mr, s, Kc, fscale)


def esp_closure(A, B, CA, CB, Nr, Mr, alpha, inv_lam):
    """ESP-weighted shape term; ``CA`` (K,N) / ``CB`` (K,M) are the charges."""
    Kc, s, fscale = _gauss_consts(alpha)
    sc = np.float32(math.sqrt(inv_lam * _LOG2E))
    As = to_soa(A) * s; Bs = to_soa(B)
    CAs = np.ascontiguousarray(CA * sc, dtype=np.float32)
    CBs = np.ascontiguousarray(CB * sc, dtype=np.float32)
    return lambda q, t: _esp_grad_kernel(As, Bs, CAs, CBs, q, t, Nr, Mr, s, Kc, fscale)


def _unit(V):
    n = np.sqrt((V.astype(np.float64) ** 2).sum(-1, keepdims=True))
    return (V / np.maximum(n, 1e-12)).astype(np.float32)


def _sort_ref_by_type(X, types, n_real, T, vecs=None):
    """Sort each row's reference features by type (padding -> T, last). Returns SoA coords,
    the per-type ranges ``st`` (P, T+1) and, if given, SoA unit vectors in the same order."""
    P, N, _ = X.shape
    ty = types.astype(np.int64).copy()
    ty[np.arange(N)[None, :] >= n_real[:, None]] = T
    order = np.argsort(ty, axis=1, kind="stable")
    tys = np.take_along_axis(ty, order, 1)
    st = np.zeros((P, T + 1), np.int64)
    st[:, 1:] = np.cumsum(np.stack([(tys == k).sum(1) for k in range(T)], 1), 1)
    out = [to_soa(np.take_along_axis(X, order[:, :, None], 1)), st]
    if vecs is not None:
        out.append(to_soa(_unit(np.take_along_axis(vecs, order[:, :, None], 1))))
    return out


def _fit_types(types, m_real, cats, T):
    """Fit feature types with padding and dummy (category 3) features mapped to T (skipped)."""
    ty = types.astype(np.int64).copy()
    ty[cats[np.clip(ty, 0, T - 1)] == 3] = T
    ty[np.arange(types.shape[1])[None, :] >= m_real[:, None]] = T
    return ty


def _typed_consts(alphas, Ks):
    al = np.asarray(alphas, np.float64)
    return ((al * 0.5 * _LOG2E).astype(np.float32), np.asarray(Ks, np.float32),
            al.astype(np.float32))


def color_closure(A, B, At, Bt, Nr, Mr, alphas, Ks, cats):
    """Directionless typed term (per-type ``alphas`` / prefactors ``Ks``)."""
    cats = np.asarray(cats, np.int64); T = int(len(alphas))
    As, st = _sort_ref_by_type(A, At, Nr, T)
    Bs = to_soa(B); Bty = _fit_types(Bt, Mr, cats, T)
    c2, Kt, al = _typed_consts(alphas, Ks)
    KA = (Kt * al).astype(np.float32)
    return lambda q, t: _color_grad_kernel(As, st, Bs, Bty, q, t, Mr, c2, Kt, KA, T)


def pharm_closure(A, VA, TA, B, VB, TB, Nr, Mr, alphas, Ks, cats):
    """Directional pharmacophore term (anchors, vectors, types for ref and fit)."""
    cats = np.asarray(cats, np.int64); T = int(len(alphas))
    As, st, RV = _sort_ref_by_type(A, TA, Nr, T, VA)
    Bs = to_soa(B); FV = to_soa(_unit(VB)); Bty = _fit_types(TB, Mr, cats, T)
    c2, Kt, al = _typed_consts(alphas, Ks)
    return lambda q, t: _pharm_grad_kernel(As, RV, st, Bs, FV, Bty, q, t, Mr, cats, c2, Kt,
                                           al, T)
