"""``pharm`` / ``pharm_tversky`` with ``extended_points=True``: an anchor+vector Gaussian
term with no directional weighting and no kernel, so it keeps the eager autograd /
analytical-gradient loop below. The default objective runs the generic engine
(``drivers/engine.py``) on the ``pharm_grad_dq_se3_batch`` kernel.
"""
from __future__ import annotations

import torch

from .pharm_overlap import (batch_pharm_cross_overlap_with_transform, batch_pharm_self_overlap,
                            pharm_similarity_from_overlaps)
from ..kernels.dispatch import fused_adam_qt
from ...score.analytical_gradients import (compute_overlap_and_grad_pharm,
                                           apply_tanimoto_chain_rule, apply_tversky_chain_rule,
                                           project_grad_R_to_quaternion,
                                           _rotation_matrix_from_unit_quat)
from .._stats import record as _record_steps
from ._common import (build_coarse_grid, batched_seeds_torch, apply_se3_transform,
                      apply_so3_transform)

_PHARM_SIGMA_MAP = {'tversky': 0.95, 'tversky_ref': 1.0, 'tversky_fit': 0.05}


def _coarse_fine_pharm_extended(
        anchors_1, anchors_2, vectors_1, vectors_2, types_1, types_2, VAA, VBB, *, similarity,
        only_extended, num_seeds, trans_centers, trans_centers_real, num_repeats_per_trans, topk,
        steps_fine, lr, N_real, M_real, early_stop_patience, early_stop_tol):
    """The eager loop for ``extended_points=True``: the analytical ``grad_R`` path on unpadded
    inputs, the autograd path on padded ones."""
    device = anchors_1.device
    BATCH, N_pad, _ = anchors_1.shape
    _, M_pad, _ = anchors_2.shape
    if N_real is None:
        N_real = anchors_1.new_full((BATCH,), N_pad, dtype=torch.int32)
    if M_real is None:
        M_real = anchors_2.new_full((BATCH,), M_pad, dtype=torch.int32)
    if VAA is None:
        VAA = batch_pharm_self_overlap(anchors_1, vectors_1, types_1, extended_points=True,
                                       only_extended=only_extended, N_real=N_real)
    if VBB is None:
        VBB = batch_pharm_self_overlap(anchors_2, vectors_2, types_2, extended_points=True,
                                       only_extended=only_extended, N_real=M_real)

    # center both clouds on their own real-point centroids (translation-invariant optimisation)
    _N = N_real.to(device=device, dtype=anchors_1.dtype)
    _M = M_real.to(device=device, dtype=anchors_2.dtype)
    _mask_n = (torch.arange(N_pad, device=device)[None] < _N[:, None]).to(anchors_1.dtype)
    _mask_m = (torch.arange(M_pad, device=device)[None] < _M[:, None]).to(anchors_2.dtype)
    c_ref = (anchors_1 * _mask_n.unsqueeze(-1)).sum(1) / _N.clamp(min=1).unsqueeze(-1)
    c_fit = (anchors_2 * _mask_m.unsqueeze(-1)).sum(1) / _M.clamp(min=1).unsqueeze(-1)
    anchors_1 = anchors_1 - c_ref[:, None, :]
    anchors_2 = anchors_2 - c_fit[:, None, :]
    if trans_centers is not None:
        trans_centers = trans_centers - c_ref[:, None, :]

    if trans_centers is None:
        quats, t_seeds = batched_seeds_torch(anchors_1, anchors_2, N_real, M_real, num_seeds=num_seeds)
        P = quats.size(1)
        q_best, t_best = quats.clone(), t_seeds.clone()
    else:
        q_grid, t_grid = build_coarse_grid(
            anchors_1, anchors_2, N_real, M_real, num_seeds=num_seeds,
            trans_centers_batch=trans_centers, trans_centers_real=trans_centers_real,
            num_repeats_per_trans=num_repeats_per_trans)
        G = q_grid.size(1)
        ORI = 10_000
        coarse = torch.empty(BATCH, G, device=device, dtype=anchors_1.dtype)
        with torch.no_grad():
            for o0 in range(0, G, ORI):
                o1 = min(o0 + ORI, G)
                g = o1 - o0
                q_rep = q_grid[:, o0:o1].reshape(-1, 4).contiguous()
                t_rep = t_grid[:, o0:o1].reshape(-1, 3).contiguous()
                ex3 = lambda x, D: x.unsqueeze(1).expand(-1, g, -1, -1).reshape(-1, D, 3)
                ex2 = lambda x, D: x.unsqueeze(1).expand(-1, g, -1).reshape(-1, D)
                VAB = batch_pharm_cross_overlap_with_transform(
                    ex3(anchors_1, N_pad), ex3(anchors_2, M_pad), ex3(vectors_1, N_pad),
                    ex3(vectors_2, M_pad), ex2(types_1, N_pad), ex2(types_2, M_pad), q_rep, t_rep,
                    extended_points=True, only_extended=only_extended,
                    N_real=N_real.repeat_interleave(g), M_real=M_real.repeat_interleave(g))
                sc = pharm_similarity_from_overlaps(VAB, VAA.repeat_interleave(g),
                                                    VBB.repeat_interleave(g), similarity=similarity)
                coarse[:, o0:o1] = sc.view(BATCH, g)
        best_idx = coarse.topk(k=topk, dim=1).indices
        q_best = torch.gather(q_grid, 1, best_idx.unsqueeze(-1).expand(-1, -1, 4)).clone()
        t_best = torch.gather(t_grid, 1, best_idx.unsqueeze(-1).expand(-1, -1, 3)).clone()
        P = topk

    q_param = q_best.reshape(-1, 4).contiguous()
    t_param = t_best.reshape(-1, 3).contiguous()
    ex3 = lambda x, D: x.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, D, 3)
    ex2 = lambda x, D: x.unsqueeze(1).expand(-1, P, -1).reshape(-1, D)
    anchors_1_k, anchors_2_k = ex3(anchors_1, N_pad), ex3(anchors_2, M_pad)
    vectors_1_k, vectors_2_k = ex3(vectors_1, N_pad), ex3(vectors_2, M_pad)
    types_1_k, types_2_k = ex2(types_1, N_pad), ex2(types_2, M_pad)
    N_k, M_k = N_real.repeat_interleave(P), M_real.repeat_interleave(P)
    VAA_k, VBB_k = VAA.repeat_interleave(P), VBB.repeat_interleave(P)

    m_q = torch.zeros_like(q_param); v_q = torch.zeros_like(q_param)
    m_t = torch.zeros_like(t_param); v_t = torch.zeros_like(t_param)
    best_score = torch.full((len(q_param),), -float("inf"), device=device)
    best_q, best_t = q_param.clone(), t_param.clone()
    prev_best = torch.full((BATCH,), -float("inf"), device=device)
    no_improve = 0

    use_analytical = bool((N_k == N_pad).all() and (M_k == M_pad).all())
    if use_analytical:
        _P = anchors_1_k.shape[0]
        _I3 = torch.eye(3, device=device, dtype=anchors_1_k.dtype).expand(_P, 3, 3)
        _z = torch.zeros(_P, 3, device=device, dtype=anchors_1_k.dtype)
        VAA_an, _, _ = compute_overlap_and_grad_pharm(
            _I3, _z, types_1_k, types_1_k, anchors_1_k, anchors_1_k, vectors_1_k, vectors_1_k,
            extended_points=True, only_extended=only_extended)
        VBB_an, _, _ = compute_overlap_and_grad_pharm(
            _I3, _z, types_2_k, types_2_k, anchors_2_k, anchors_2_k, vectors_2_k, vectors_2_k,
            extended_points=True, only_extended=only_extended)

    step = -1
    for step in range(steps_fine):
        if use_analytical:
            q_unit = torch.nn.functional.normalize(q_param, dim=1)
            R = _rotation_matrix_from_unit_quat(q_unit)
            O_AB, grad_R, grad_t = compute_overlap_and_grad_pharm(
                R, t_param, types_1_k, types_2_k, anchors_1_k, anchors_2_k, vectors_1_k,
                vectors_2_k, extended_points=True, only_extended=only_extended)
            score = pharm_similarity_from_overlaps(O_AB, VAA_an, VBB_an, similarity=similarity)
            if similarity == 'tanimoto':
                _, sgrad_R, sgrad_t = apply_tanimoto_chain_rule(O_AB, VAA_an + VBB_an, grad_R, grad_t)
            else:
                sigma = _PHARM_SIGMA_MAP[similarity]
                D = sigma * VAA_an + (1.0 - sigma) * VBB_an
                _, sgrad_R, sgrad_t = apply_tversky_chain_rule(O_AB, D, grad_R, grad_t)
            sgrad_q_unit = project_grad_R_to_quaternion(sgrad_R, q_unit)
            qn = q_param.norm(dim=1, keepdim=True).clamp(min=1e-12)
            dQ = (sgrad_q_unit - q_unit * (q_unit * sgrad_q_unit).sum(1, keepdim=True)) / qn
            dT = sgrad_t
        else:
            q_var = q_param.detach().requires_grad_(True)
            t_var = t_param.detach().requires_grad_(True)
            VAB = batch_pharm_cross_overlap_with_transform(
                anchors_1_k, anchors_2_k, vectors_1_k, vectors_2_k, types_1_k, types_2_k,
                q_var, t_var, extended_points=True, only_extended=only_extended,
                N_real=N_k, M_real=M_k)
            score = pharm_similarity_from_overlaps(VAB, VAA_k, VBB_k, similarity=similarity)
            dQ, dT = torch.autograd.grad(-score.sum(), (q_var, t_var), create_graph=False)
        score_det = score.detach()
        better = score_det > best_score
        best_score = torch.where(better, score_det, best_score)
        mask_q = better.unsqueeze(1)
        best_q = torch.where(mask_q, q_param, best_q)
        best_t = torch.where(mask_q, t_param, best_t)
        if step % 5 == 0:
            cur = best_score.view(BATCH, P).amax(dim=1)
            improved = (cur - prev_best) > early_stop_tol
            if not improved.any():
                no_improve += 1
                if no_improve >= early_stop_patience:
                    break
            else:
                no_improve = 0
            prev_best = torch.where(improved, cur, prev_best)
        radial = (dQ * q_param).sum(dim=1, keepdim=True)
        dQ_tan = dQ - q_param * radial
        fused_adam_qt(q_param, t_param, dQ_tan.detach(), dT.detach(), m_q, v_q, m_t, v_t, lr)
    ran = (step + 1) if steps_fine else 0
    _record_steps(ran, steps_fine, ran < steps_fine)

    final = best_score.view(BATCH, P)
    best = final.argmax(dim=1)
    sel = best + torch.arange(BATCH, device=device) * P
    out_q, out_t = best_q[sel], best_t[sel]
    R_out = _rotation_matrix_from_unit_quat(torch.nn.functional.normalize(out_q, dim=1))
    out_t = out_t - torch.einsum('bij,bj->bi', R_out, c_fit) + c_ref
    return final.flatten()[sel], out_q, out_t


_VBB_MEMO: dict = {}


def _self_overlap_cached(anchors, vectors, types, *, extended_points, only_extended, N_real):
    """``batch_pharm_self_overlap`` memoised on the identity of its inputs (the entry holds a
    reference to what it keyed on, so a data_ptr cannot be recycled under the key)."""
    key = (anchors.data_ptr(), tuple(anchors.shape), vectors.data_ptr(), types.data_ptr(),
           None if N_real is None else N_real.data_ptr(),
           bool(extended_points), bool(only_extended), str(anchors.device), anchors.dtype)
    hit = _VBB_MEMO.get("k")
    if hit == key:
        return _VBB_MEMO["v"]
    v = batch_pharm_self_overlap(anchors, vectors, types, extended_points=extended_points,
                                 only_extended=only_extended, N_real=N_real)
    _VBB_MEMO.clear()
    _VBB_MEMO.update(k=key, v=v, _keep=(anchors, vectors, types, N_real))
    return v


def _self_overlap_shared_ref(anchors, vectors, types, *, extended_points, only_extended,
                             N_real, shared):
    """Self-overlap of a ref side that may be one molecule replicated across the batch."""
    if not shared or anchors.shape[0] <= 1:
        return batch_pharm_self_overlap(anchors, vectors, types, extended_points=extended_points,
                                        only_extended=only_extended, N_real=N_real)
    one = _self_overlap_cached(anchors[:1], vectors[:1], types[:1],
                               extended_points=extended_points, only_extended=only_extended,
                               N_real=None if N_real is None else N_real[:1])
    return one.expand(anchors.shape[0]).contiguous()


def fast_optimize_pharm_overlay_batch(
        ref_pharms_batch, fit_pharms_batch, ref_anchors_batch, fit_anchors_batch,
        ref_vectors_batch, fit_vectors_batch, *, similarity: str = "tanimoto",
        extended_points: bool = False, only_extended: bool = False, num_repeats: int = 50,
        trans_centers_batch=None, trans_centers_real=None, num_repeats_per_trans: int = 10,
        N_real=None, M_real=None, topk: int = 30, steps_fine: int = 100, lr: float = 0.075,
        ref_shared: bool = False):
    """Batched ``extended_points=True`` pharmacophore alignment:
    ``(aligned_anchors, aligned_vectors, q, t, scores)``."""
    if not extended_points:
        raise ValueError("fast_optimize_pharm_overlay_batch runs extended_points=True only; the "
                         "default objective runs on the engine (MoleculePairBatch / _align_batch_pharm)")
    BATCH = ref_anchors_batch.shape[0]
    if N_real is None:
        N_real = ref_anchors_batch.new_full((BATCH,), ref_anchors_batch.shape[1], dtype=torch.int32)
    if M_real is None:
        M_real = fit_anchors_batch.new_full((BATCH,), fit_anchors_batch.shape[1], dtype=torch.int32)
    VAA = _self_overlap_shared_ref(ref_anchors_batch, ref_vectors_batch, ref_pharms_batch,
                                   extended_points=True, only_extended=only_extended,
                                   N_real=N_real, shared=ref_shared)
    VBB = _self_overlap_cached(fit_anchors_batch, fit_vectors_batch, fit_pharms_batch,
                               extended_points=True, only_extended=only_extended, N_real=M_real)
    scores, q_best, t_best = _coarse_fine_pharm_extended(
        ref_anchors_batch, fit_anchors_batch, ref_vectors_batch, fit_vectors_batch,
        ref_pharms_batch, fit_pharms_batch, VAA, VBB, similarity=similarity,
        only_extended=only_extended, num_seeds=num_repeats,
        trans_centers=trans_centers_batch, trans_centers_real=trans_centers_real,
        num_repeats_per_trans=num_repeats_per_trans, topk=topk, steps_fine=steps_fine, lr=lr,
        N_real=N_real, M_real=M_real, early_stop_patience=5, early_stop_tol=1e-5)
    aligned_anchors = apply_se3_transform(fit_anchors_batch, q_best, t_best)
    aligned_vectors = apply_so3_transform(fit_vectors_batch, q_best)
    return aligned_anchors, aligned_vectors, q_best, t_best, scores
