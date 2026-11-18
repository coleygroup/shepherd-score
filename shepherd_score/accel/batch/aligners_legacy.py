"""Autograd fallback for extended-point pharmacophore alignment."""
from __future__ import annotations

import torch

from shepherd_score.alignment.utils.se3 import quaternions_to_SE3_batch

from ._bucket import plan_buckets, PadSpec
from ._pad import _subbatched_align, _scatter_fill
from ..channels import CHANNELS


def _align_batch_pharm_extended(spec, pairs, params, *, n_seeds, steps_fine, trans_init,
                                nrpt, topk):
    from ..drivers.pharm import fast_optimize_pharm_overlay_batch
    from .aligners import upload_channels
    device = pairs[0].device
    names = ("pharm_types", "pharm_ancs", "pharm_vecs")
    upload_channels(pairs, names, device)
    ta, tv, tt = (CHANNELS[n] for n in ("pharm_ancs", "pharm_vecs", "pharm_types"))
    _tc = (lambda p: int(getattr(p, ta.ref_attr).shape[0])) if trans_init else (lambda p: 0)
    pspec = PadSpec(merge={"ref": lambda p: getattr(p, ta.ref_attr).shape[0],
                           "fit": lambda p: getattr(p, ta.fit_attr).shape[0]},
                    seeds=n_seeds, partition={"tc": _tc})
    all_pairs, all_scores, all_q, all_t = [], [], [], []
    for bk in plan_buckets(pairs, pspec, device):
        N_pad, M_pad = bk.pad["ref"], bk.pad["fit"]
        bucket = bk.members
        K = len(bucket)
        ref_types = torch.zeros(K, N_pad, device=device, dtype=torch.int64)
        fit_types = torch.zeros(K, M_pad, device=device, dtype=torch.int64)
        ref_ancs = torch.zeros(K, N_pad, 3, device=device, dtype=torch.float32)
        fit_ancs = torch.zeros(K, M_pad, 3, device=device, dtype=torch.float32)
        ref_vecs = torch.zeros(K, N_pad, 3, device=device, dtype=torch.float32)
        fit_vecs = torch.zeros(K, M_pad, 3, device=device, dtype=torch.float32)
        ra = [getattr(p, ta.ref_attr) for p in bucket]
        fa = [getattr(p, ta.fit_attr) for p in bucket]
        n_list = [t.shape[0] for t in ra]
        m_list = [t.shape[0] for t in fa]
        N_real = torch.tensor(n_list, device=device, dtype=torch.int32)
        M_real = torch.tensor(m_list, device=device, dtype=torch.int32)
        _scatter_fill(ref_types, [getattr(p, tt.ref_attr) for p in bucket], n_list)
        _scatter_fill(fit_types, [getattr(p, tt.fit_attr) for p in bucket], m_list)
        _scatter_fill(ref_ancs, ra, n_list)
        _scatter_fill(fit_ancs, fa, m_list)
        _scatter_fill(ref_vecs, [getattr(p, tv.ref_attr) for p in bucket], n_list)
        _scatter_fill(fit_vecs, [getattr(p, tv.fit_attr) for p in bucket], m_list)
        tcb = ref_ancs if trans_init else None
        tcr = N_real if trans_init else None

        def _proc(_s, _k):
            sl = slice(_s, _s + _k)
            _, _, q, t, sc = fast_optimize_pharm_overlay_batch(
                ref_types[sl], fit_types[sl], ref_ancs[sl], fit_ancs[sl], ref_vecs[sl],
                fit_vecs[sl], similarity=params.get("similarity", "tanimoto"),
                extended_points=True, only_extended=bool(params.get("only_extended", False)),
                num_repeats=n_seeds,
                trans_centers_batch=None if tcb is None else tcb[sl],
                trans_centers_real=None if tcr is None else tcr[sl],
                num_repeats_per_trans=nrpt, N_real=N_real[sl], M_real=M_real[sl],
                topk=topk, steps_fine=steps_fine, lr=float(params["lr"]))
            return sc, q, t
        scores, q_b, t_b = _subbatched_align(_proc, K, key=(spec.name, "ext", N_pad, M_pad, n_seeds),
                                             device=device)
        all_pairs.extend(bucket); all_scores.append(scores); all_q.append(q_b); all_t.append(t_b)
    SE3_all = quaternions_to_SE3_batch(torch.cat(all_q).cpu(), torch.cat(all_t).cpu()).detach().numpy()
    tf_attr, sc_attr = spec.attrs
    for p, s, S in zip(all_pairs, torch.cat(all_scores).cpu().tolist(), SE3_all):
        setattr(p, tf_attr, S)
        setattr(p, sc_attr, s)
