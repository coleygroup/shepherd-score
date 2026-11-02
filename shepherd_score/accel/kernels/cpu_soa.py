"""SoA fp32 CPU kernels: the vectorizable twins of the AoS fp64 kernels in cpu.py.

The AoS layout (``A[k, n, :]``) and fp64 accumulation in cpu.py keep LLVM from vectorizing
the inner O(N*M) loop. These kernels take coordinates SoA (``A[k, :, n]``, contiguous in n)
and accumulate fp32, so with an SVML-enabled numba (``numba<0.62`` + ``icc_rt``,
``config.USING_SVML``) the inner loop vectorizes and SVML supplies the vector ``exp``. Used
only by the fused CPU driver (cpu_fused.py) when SVML is present. The rotation and dQ tail
stay fp64 per pose; only the inner n-loop is fp32. Math mirrors cpu.py op for op.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange
from ._svml import enable_svml

enable_svml()   # before any kernel compiles; pip's icc_rt needs the explicit load

_K_PI = math.pi ** 1.5

# shape (vol / surf, and vol_color's shape channel)

@njit(parallel=True, fastmath=True, cache=True)
def _overlap_grad_kernel_soa(A, B, q, t, Nr, Mr, alpha, need_grad):
    """SoA fp32 twin of cpu._overlap_grad_kernel. A (K,3,N), B (K,3,M) float32."""
    K = A.shape[0]; f = np.float32
    Kc = f(_K_PI / (2.0 * alpha) ** 1.5); a2 = f(alpha * 0.5); al = f(alpha)
    V = np.zeros(K, np.float32); dQ = np.zeros((K, 4), np.float32); dT = np.zeros((K, 3), np.float32)
    for k in prange(K):
        qr = q[k, 0]; qi = q[k, 1]; qj = q[k, 2]; qk_ = q[k, 3]
        tx = t[k, 0]; ty = t[k, 1]; tz = t[k, 2]
        r00 = 1.0 - 2.0 * (qj * qj + qk_ * qk_); r01 = 2.0 * (qi * qj - qk_ * qr); r02 = 2.0 * (qi * qk_ + qj * qr)
        r10 = 2.0 * (qi * qj + qk_ * qr); r11 = 1.0 - 2.0 * (qi * qi + qk_ * qk_); r12 = 2.0 * (qj * qk_ - qi * qr)
        r20 = 2.0 * (qi * qk_ - qj * qr); r21 = 2.0 * (qj * qk_ + qi * qr); r22 = 1.0 - 2.0 * (qi * qi + qj * qj)
        n_real = Nr[k]; m_real = Mr[k]
        Vacc = f(0.0); dTx = 0.0; dTy = 0.0; dTz = 0.0; dQw = 0.0; dQx = 0.0; dQy = 0.0; dQz = 0.0
        for m in range(m_real):
            bx0 = B[k, 0, m]; by0 = B[k, 1, m]; bz0 = B[k, 2, m]
            bx = f(r00 * bx0 + r01 * by0 + r02 * bz0 + tx)
            by = f(r10 * bx0 + r11 * by0 + r12 * bz0 + ty)
            bz = f(r20 * bx0 + r21 * by0 + r22 * bz0 + tz)
            fxj = f(0.0); fyj = f(0.0); fzj = f(0.0)
            for n in range(n_real):                          # contiguous in n -> SIMD + SVML exp
                dx = A[k, 0, n] - bx; dy = A[k, 1, n] - by; dz = A[k, 2, n] - bz
                g = Kc * f(math.exp(-a2 * (dx * dx + dy * dy + dz * dz)))
                Vacc += g
                if need_grad:
                    cc = al * g
                    fxj += cc * dx; fyj += cc * dy; fzj += cc * dz
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

# ESP-weighted shape (vol_esp / surf_esp): shape kernel × charge Gaussian

@njit(parallel=True, fastmath=True, cache=True)
def _overlap_grad_esp_kernel_soa(A, B, CA, CB, q, t, Nr, Mr, alpha, inv_lam, need_grad):
    """SoA fp32 twin of cpu._overlap_grad_esp_kernel. A (K,3,N)/B (K,3,M); CA (K,N)/CB (K,M)."""
    K = A.shape[0]; f = np.float32
    Kc = f(_K_PI / (2.0 * alpha) ** 1.5); a2 = f(alpha * 0.5); al = f(alpha); il = f(inv_lam)
    V = np.zeros(K, np.float32); dQ = np.zeros((K, 4), np.float32); dT = np.zeros((K, 3), np.float32)
    for k in prange(K):
        qr = q[k, 0]; qi = q[k, 1]; qj = q[k, 2]; qk_ = q[k, 3]
        tx = t[k, 0]; ty = t[k, 1]; tz = t[k, 2]
        r00 = 1.0 - 2.0 * (qj * qj + qk_ * qk_); r01 = 2.0 * (qi * qj - qk_ * qr); r02 = 2.0 * (qi * qk_ + qj * qr)
        r10 = 2.0 * (qi * qj + qk_ * qr); r11 = 1.0 - 2.0 * (qi * qi + qk_ * qk_); r12 = 2.0 * (qj * qk_ - qi * qr)
        r20 = 2.0 * (qi * qk_ - qj * qr); r21 = 2.0 * (qj * qk_ + qi * qr); r22 = 1.0 - 2.0 * (qi * qi + qj * qj)
        n_real = Nr[k]; m_real = Mr[k]
        Vacc = f(0.0); dTx = 0.0; dTy = 0.0; dTz = 0.0; dQw = 0.0; dQx = 0.0; dQy = 0.0; dQz = 0.0
        for m in range(m_real):
            bx0 = B[k, 0, m]; by0 = B[k, 1, m]; bz0 = B[k, 2, m]; cbm = CB[k, m]
            bx = f(r00 * bx0 + r01 * by0 + r02 * bz0 + tx)
            by = f(r10 * bx0 + r11 * by0 + r12 * bz0 + ty)
            bz = f(r20 * bx0 + r21 * by0 + r22 * bz0 + tz)
            fxj = f(0.0); fyj = f(0.0); fzj = f(0.0)
            for n in range(n_real):
                dx = A[k, 0, n] - bx; dy = A[k, 1, n] - by; dz = A[k, 2, n] - bz
                dc = CA[k, n] - cbm
                g = Kc * f(math.exp(-a2 * (dx * dx + dy * dy + dz * dz) - dc * dc * il))
                Vacc += g
                if need_grad:
                    cc = al * g
                    fxj += cc * dx; fyj += cc * dy; fzj += cc * dz
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


def to_soa(x_np):
    """(K,N,3) AoS float32 -> (K,3,N) contiguous float32 (so the inner n-loop is contiguous)."""
    return np.ascontiguousarray(np.transpose(x_np, (0, 2, 1)), dtype=np.float32)
