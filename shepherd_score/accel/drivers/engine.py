"""Batched SE(3) optimization driven by mode specifications."""
from __future__ import annotations

from collections import namedtuple
from typing import Optional

import torch
import torch.nn.functional as F

from .._stats import record as _record_steps
from ..kernels.dispatch import fused_adam_qt, fused_adam_qt_with_tangent_proj
from ._common import batched_seeds_torch, build_coarse_grid
from ._graphed import _GraphedFineBase, run_graphed, graph_cap
from ...score.analytical_gradients import _rotation_matrix_from_unit_quat
from . import terms as T

#: Padded per-pair tensors of one channel: ``ref`` (B, N_pad, ...), ``fit`` (B, M_pad, ...) or
#: ``None`` for a pair-level channel, and int32 real counts (``m_real`` ``None`` likewise).
Batch = namedtuple("Batch", "ref fit n_real m_real")

_PHARM_SIGMA = {"tversky": 0.95, "tversky_ref": 1.0, "tversky_fit": 0.05}

# the assembled problem

class _Term:
    """One term's per-pose tensors and reduction constants, in the fine-loop layout.

    ``scratch`` holds the reduction's working buffers so the captured fine step allocates
    nothing. It is allocated lazily on first use, which on the graph path is the warmup in
    ``_GraphedFineBase.capture``, off the capture stream, so every replay reuses it."""
    __slots__ = ("spec", "inputs", "weight", "norm", "C", "k", "guard", "vaa", "vbb", "sigma",
                 "scratch")

    def __init__(self, spec, inputs, weight):
        self.spec = spec
        self.inputs = inputs
        self.weight = weight
        self.norm = self.C = self.k = self.guard = self.vaa = self.vbb = self.sigma = None
        self.scratch = None


def _scratch_for(tm, V):
    """``(den, sim, scl, d2, ng)`` sized like ``V``, or ``None`` when this call's shape is not
    the one the buffers were cut for (``_score_poses`` and ``_coarse_topk`` evaluate a different
    pose count, and must keep the allocating path)."""
    s = tm.scratch
    if s is not None and s[0].shape == V.shape and s[0].dtype == V.dtype:
        return s
    if s is not None:
        return None                       # different shape: do not thrash the fine loop's buffers
    e = lambda: torch.empty_like(V)
    ng = (torch.empty_like(tm.guard) if tm.guard is not None else None)
    tm.scratch = (e(), e(), e(), e(), ng)
    return tm.scratch


class Problem:
    """Everything the fine loop needs for one bucket, already expanded per pose."""
    __slots__ = ("spec", "B", "P", "S_fine", "P_cta", "q", "t", "terms", "device", "dtype",
                 "work", "c_ref", "c_fit", "pads", "params", "similarity")


def _weight_of(term, spec, params, terms):
    """Blend weight of ``term`` as a Python float (negative for a subtracted penalty)."""
    w = term.weight
    if w is None:                                    # complement of the other named weight
        other = next(x.weight for x in terms if isinstance(x.weight, str))
        return 1.0 - float(params[other.lstrip("-")])
    if isinstance(w, str):
        neg = w.startswith("-")
        v = float(params[w.lstrip("-")])
        return -v if neg else v
    return float(w)


def _expand3(x, P, D):
    return x.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, D, 3)


def _expand2(x, P, D):
    return x.unsqueeze(1).expand(-1, P, -1).reshape(-1, D)


def _rep(x, P):
    return x.repeat_interleave(P)


def _expand(x, P):
    if x.dim() == 3:
        return _expand3(x, P, x.shape[1])
    if x.dim() == 2:
        return _expand2(x, P, x.shape[1])
    return _rep(x, P)


def _per_pose(term, ti, P, dedup=False):
    """``(ref, fit, n_real, m_real, params)`` of one term's per-pair inputs laid out for ``P``
    poses per pair: as-is under ``dedup`` (the kernel indexes ``pose // P``), else replicated.
    The avoid term keeps its cloud and counts per pair and records ``_a_div = P``: its kernels
    read avoid row ``pose // P``, so a pair's seeds share one copy of the cloud instead of P.
    A shared cloud (one row for the whole batch, :func:`is_shared_pair`) passes through the
    same way and is broadcast to every pose at launch. The avoid layout ignores ``dedup``: its
    kernels index the fit by pose, with no ``pid // S`` molecule indexing, and every term
    carries its own inputs, so it can sit beside deduped terms in one problem."""
    tp = dict(ti.params)
    for key in ("_n_surf", "_m_surf"):
        if key in tp:
            tp[key] = _rep(tp[key], P)
    if term.kernel == "avoid":
        tp["_a_div"] = int(P)
        return (ti.ref, tuple(_expand(x, P) for x in ti.fit), ti.n_real, _rep(ti.m_real, P),
                tp)
    if dedup:
        return ti.ref, ti.fit, ti.n_real, ti.m_real, tp
    fit = tuple(_expand(x, P) for x in ti.fit)
    m_real = _rep(ti.m_real, P)
    return tuple(_expand(x, P) for x in ti.ref), fit, _rep(ti.n_real, P), m_real, tp


def _masked_centroid(x, n_real):
    n = n_real.to(dtype=x.dtype)
    mask = (torch.arange(x.shape[1], device=x.device)[None] < n[:, None]).to(x.dtype)
    return (x * mask.unsqueeze(-1)).sum(1) / n.clamp(min=1).unsqueeze(-1)


def term_self_overlaps(spec, chans: dict, params: dict, ref_shared: bool = False):
    """``[(vaa, vbb) | None]`` per term over a whole bucket, so a caller that sub-batches the
    bucket computes them once and slices per chunk. Under ``ref_shared`` the reference
    self-overlap is solved on row 0 and expanded."""
    res = {c: spec.resolve_channel(c, params) for c in spec.channels}
    seed = chans[res.get(spec.seed_channel, spec.seed_channel)].ref
    B, device, dtype = int(seed.shape[0]), seed.device, seed.dtype
    out = []
    for tm in spec.terms:
        if tm.reduction not in ("tanimoto", "tversky", "pharm_sim"):
            out.append(None)
            continue
        ref = tuple(chans[res.get(n, n)].ref for n in tm.ref)
        fit = tuple(chans[res.get(n, n)].fit for n in tm.fit)
        n_real = chans[res.get(tm.ref[0], tm.ref[0])].n_real
        m_real = chans[res.get(tm.fit[0], tm.fit[0])].m_real
        tables = T.tables_for(tm, device, dtype, params)
        if ref_shared and B > 1:
            vaa = T.self_overlap(tm, tuple(x[:1] for x in ref), n_real[:1], tables, params)
            vaa = vaa.expand(B).contiguous()
        else:
            vaa = T.self_overlap(tm, ref, n_real, tables, params)
        out.append((vaa, T.self_overlap(tm, fit, m_real, tables, params)))
    return out


def assemble(spec, chans: dict, *, params: dict, num_seeds: int, seeds=None,
             ref_shared: bool = False, trans_centers=None, trans_centers_real=None,
             num_repeats_per_trans: int = 10, topk: int = 30, self_overlaps=None) -> Problem:
    """Resolve channels, center (pharm family), seed, expand per pose, and precompute every
    reduction constant. ``chans`` maps concrete channel names to :class:`Batch`."""
    res = {c: spec.resolve_channel(c, params) for c in spec.channels}
    seed_ch = res[spec.seed_channel] if spec.seed_channel in res else spec.resolve_channel(
        spec.seed_channel, params)
    sb = chans[seed_ch]
    device, dtype = sb.ref.device, sb.ref.dtype
    B = int(sb.ref.shape[0])

    # ---- pharm family: center both clouds on their own real-point centroids ---------------
    c_ref = c_fit = None
    if spec.center_clouds:
        from ..channels import CHANNELS
        c_ref = _masked_centroid(sb.ref, sb.n_real)
        c_fit = _masked_centroid(sb.fit, sb.m_real)
        shifted = {}
        for name, b in chans.items():
            if CHANNELS[name].kind == "points" and not CHANNELS[name].is_pair:
                shifted[name] = Batch(b.ref - c_ref[:, None, :], b.fit - c_fit[:, None, :],
                                      b.n_real, b.m_real)
            else:
                shifted[name] = b
        chans = shifted
        sb = chans[seed_ch]
        if trans_centers is not None:
            trans_centers = trans_centers - c_ref[:, None, :]

    # ---- term inputs (per pair, before expansion) + reduction constants -------------------
    term_objs = []
    for tm in spec.terms:
        ref_names = [res.get(n, n) for n in tm.ref]
        fit_names = [res.get(n, n) for n in tm.fit]
        ref = tuple(chans[n].ref for n in ref_names)
        fit = tuple(chans[n].fit for n in fit_names)
        n_real = chans[ref_names[0]].n_real
        m_real = chans[fit_names[0]].m_real
        tp = dict(params)
        if tm.kernel == "esp_cmp":
            tp["_n_surf"] = chans[res.get("surf", "surf")].n_real
            tp["_m_surf"] = chans[res.get("surf", "surf")].m_real
        tables = T.tables_for(tm, device, dtype, params)
        obj = _Term(tm, T.TermInputs(ref, fit, n_real, m_real, tables, None, tp),
                    _weight_of(tm, spec, params, spec.terms))
        if tm.reduction in ("tanimoto", "tversky", "pharm_sim"):
            pre = None if self_overlaps is None else self_overlaps[len(term_objs)]
            if pre is not None:                       # hoisted per bucket by the caller
                vaa, vbb = pre
            else:
                if ref_shared and B > 1:
                    vaa = T.self_overlap(tm, tuple(x[:1] for x in ref), n_real[:1], tables,
                                         params)
                    vaa = vaa.expand(B).contiguous()
                else:
                    vaa = T.self_overlap(tm, ref, n_real, tables, params)
                vbb = T.self_overlap(tm, fit, m_real, tables, params)
            obj.vaa, obj.vbb = vaa, vbb
        if tm.guard:
            has = n_real > 0
            if m_real is not None:
                has = has & (m_real > 0)
            obj.guard = has
        term_objs.append(obj)

    # ---- seeds ---------------------------------------------------------------------------
    if trans_centers is not None:
        prob0 = _finish(spec, chans, term_objs, params, B, None, None, device, dtype,
                        c_ref, c_fit, seed_ch, ref_shared)
        cb = sb if spec.coarse_channel is None else chans[
            res.get(spec.coarse_channel, spec.coarse_channel)]
        quats, t_seeds = _coarse_topk(prob0, cb, num_seeds, trans_centers, trans_centers_real,
                                      num_repeats_per_trans, topk)
    elif seeds is not None:
        quats, t_seeds = seeds
    else:
        quats, t_seeds = batched_seeds_torch(sb.ref, sb.fit, sb.n_real, sb.m_real,
                                             num_seeds=num_seeds, ref_shared=bool(ref_shared))
    return _finish(spec, chans, term_objs, params, B, quats, t_seeds, device, dtype,
                   c_ref, c_fit, seed_ch, ref_shared)


def _finish(spec, chans, term_objs, params, B, quats, t_seeds, device, dtype, c_ref, c_fit,
            seed_ch, ref_shared):
    """Expand the per-pair term inputs into the fine-loop pose layout."""
    pr = Problem()
    pr.spec, pr.B, pr.device, pr.dtype, pr.params = spec, B, device, dtype, params
    pr.c_ref, pr.c_fit = c_ref, c_fit
    pr.similarity = params.get("similarity", "tanimoto")
    sb = chans[seed_ch]
    pr.pads = (int(sb.ref.shape[1]), int(sb.fit.shape[1]))
    pr.work = pr.pads[0] * pr.pads[1]
    if quats is None:                                # grid-scoring stub: no poses yet
        pr.P, pr.S_fine, pr.P_cta, pr.q, pr.t, pr.terms = 0, 1, 1, None, None, term_objs
        return pr
    P = int(quats.shape[1])
    pr.P = P
    pr.q = quats.reshape(-1, 4).contiguous()
    pr.t = t_seeds.reshape(-1, 3).contiguous()
    # Dedup layout (surf): hand the kernel the molecule blocks once and let it index
    # ``pid // S``; needs the multi-pose shape kernel, so CUDA fp32 single-shape-term only.
    want = int(spec.multipose)
    dedup = (want > 1 and device.type == "cuda" and dtype == torch.float32 and P > 1
             and len(spec.terms) == 1 and spec.terms[0].kernel == "shape")
    pr.S_fine = P if dedup else 1
    pr.P_cta = want if (dedup and P % want == 0) else 1
    out = []
    for obj in term_objs:
        ti = obj.inputs
        ref, fit, n_real, m_real, tp = _per_pose(obj.spec, ti, P, dedup)
        new = _Term(obj.spec, T.TermInputs(ref, fit, n_real, m_real, ti.tables, None, tp),
                    obj.weight)
        red = obj.spec.reduction
        if red == "tanimoto":
            new.norm = _rep(obj.vaa + obj.vbb, P)
        elif red == "tversky":
            ta, tb = float(params["tversky_alpha"]), float(params["tversky_beta"])
            new.k = 1.0 - ta - tb
            new.C = _rep(ta * obj.vaa + tb * obj.vbb, P)
        elif red == "pharm_sim":
            new.vaa, new.vbb = _rep(obj.vaa, P), _rep(obj.vbb, P)
            if pr.similarity == "tanimoto":
                new.norm = new.vaa + new.vbb
            else:
                s = _PHARM_SIGMA[pr.similarity]
                new.sigma = s
                new.C = s * new.vaa + (1.0 - s) * new.vbb
        if obj.guard is not None:
            new.guard = _rep(obj.guard, P)
        out.append(new)
    pr.terms = out
    return pr

# scoring one pose set (value only): the coarse grid

@torch.no_grad()
def _score_poses(pr: Problem, q, t):
    """Total score of ``q``/``t`` (per pose, replicated layout) from value-only evaluations."""
    score = None
    for tm in pr.terms:
        V, _, _ = T.evaluate(tm.spec, tm.inputs, q, t, need_grad=False)
        sim = _reduce_value(tm, V, pr)
        contrib = sim if tm.weight == 1.0 else sim * tm.weight
        score = contrib if score is None else score + contrib
    return score


def _reduce_value(tm, V, pr):
    red = tm.spec.reduction
    if red == "tanimoto":
        sim = V / (tm.norm - V)
    elif red == "tversky":
        sim = V / (tm.k * V + tm.C)
    elif red == "pharm_sim":
        from .pharm_overlap import pharm_similarity_from_overlaps
        sim = pharm_similarity_from_overlaps(V, tm.vaa, tm.vbb, similarity=pr.similarity)
    else:
        sim = V
    if tm.guard is not None:
        sim = torch.where(tm.guard, sim, torch.zeros_like(sim))
    return sim


@torch.no_grad()
def _coarse_topk(pr0, cb, num_seeds, trans_centers, trans_centers_real, nrpt, topk):
    """Legacy ``trans_init`` path: a coarse grid of poses scored value-only, top-k kept.

    ``cb`` is the mode's ``coarse_channel`` batch, which is the seed channel for every mode but
    the combo pair (see ``ModeSpec.coarse_channel``)."""
    q_grid, t_grid = build_coarse_grid(cb.ref, cb.fit, cb.n_real, cb.m_real, num_seeds=num_seeds,
                                       trans_centers_batch=trans_centers,
                                       trans_centers_real=trans_centers_real,
                                       num_repeats_per_trans=nrpt)
    B, G = q_grid.shape[0], q_grid.shape[1]
    ORI = 5_000
    coarse = torch.empty(B, G, device=q_grid.device, dtype=q_grid.dtype)
    for o0 in range(0, G, ORI):
        o1 = min(o0 + ORI, G)
        g = o1 - o0
        # expand the per-pair term inputs g times (replicated layout, no dedup)
        qs = q_grid[:, o0:o1].reshape(-1, 4).contiguous()
        ts = t_grid[:, o0:o1].reshape(-1, 3).contiguous()
        tmp = Problem()
        tmp.spec, tmp.similarity, tmp.params = pr0.spec, pr0.similarity, pr0.params
        tmp.terms = []
        for obj in pr0.terms:
            ti = obj.inputs
            ref, fit, n_real, m_real, tp = _per_pose(obj.spec, ti, g)
            new = _Term(obj.spec, T.TermInputs(ref, fit, n_real, m_real, ti.tables, None, tp),
                        obj.weight)
            red = obj.spec.reduction
            if red == "tanimoto":
                new.norm = _rep(obj.vaa + obj.vbb, g)
            elif red == "tversky":
                ta, tb = float(pr0.params["tversky_alpha"]), float(pr0.params["tversky_beta"])
                new.k, new.C = 1.0 - ta - tb, _rep(ta * obj.vaa + tb * obj.vbb, g)
            elif red == "pharm_sim":
                new.vaa, new.vbb = _rep(obj.vaa, g), _rep(obj.vbb, g)
            if obj.guard is not None:
                new.guard = _rep(obj.guard, g)
            tmp.terms.append(new)
        coarse[:, o0:o1] = _score_poses(tmp, qs, ts).view(B, g)
    best_idx = coarse.topk(k=topk, dim=1).indices
    q_best = torch.gather(q_grid, 1, best_idx.unsqueeze(-1).expand(-1, -1, 4)).clone()
    t_best = torch.gather(t_grid, 1, best_idx.unsqueeze(-1).expand(-1, -1, 3)).clone()
    return q_best, t_best

# the fine step: value+grad of every term -> score, best-pose tracking, descent gradient, Adam

class _State:
    """Loop-carried buffers (updated in place, so one body serves eager and graph replay)."""
    __slots__ = ("q", "t", "mq", "vq", "mt", "vt", "best", "bq", "bt", "gq", "gt", "lr")

    def __init__(self, q0, t0, lr):
        self.q = q0.clone()
        self.t = t0.clone()
        self.mq = torch.zeros_like(self.q); self.vq = torch.zeros_like(self.q)
        self.mt = torch.zeros_like(self.t); self.vt = torch.zeros_like(self.t)
        self.best = torch.full((q0.shape[0],), -float("inf"), device=q0.device, dtype=q0.dtype)
        self.bq = q0.clone(); self.bt = t0.clone()
        self.gq = torch.empty_like(self.q); self.gt = torch.empty_like(self.t)
        self.lr = float(lr)

    def reset(self, q0, t0):
        self.q.copy_(q0); self.t.copy_(t0)
        self.mq.zero_(); self.vq.zero_(); self.mt.zero_(); self.vt.zero_()
        self.best.fill_(-float("inf")); self.bq.copy_(q0); self.bt.copy_(t0)


def _track_best(st: _State, score):
    better = score > st.best
    torch.where(better, score, st.best, out=st.best)
    bm = better.unsqueeze(1)
    torch.where(bm, st.q, st.bq, out=st.bq)
    torch.where(bm, st.t, st.bt, out=st.bt)


def _reduce_grad_term(tm, V):
    """``(sim, scale)`` for a gradient-bearing term: the similarity and d(sim)/dV.

    Written through the term's ``scratch`` buffers when they fit this call's shape, so the
    captured step allocates nothing; the allocating branch serves off-shape callers.
    ``~tm.guard`` is recomputed each step because ``_load`` copies a new bucket's guard into
    the same tensor, so a cached negation would go stale."""
    red = tm.spec.reduction
    buf = _scratch_for(tm, V)
    if buf is not None:
        den, sim, scl, d2, ng = buf
        if tm.guard is not None:
            torch.logical_not(tm.guard, out=ng)
        if red == "tanimoto":
            torch.sub(tm.norm, V, out=den)
            if tm.guard is not None:
                den.masked_fill_(ng, 1.0)
            torch.div(V, den, out=sim)
            torch.mul(den, den, out=d2)
            torch.div(tm.norm, d2, out=scl)
        elif red == "tversky":
            torch.mul(V, tm.k, out=den)
            den.add_(tm.C)
            if tm.guard is not None:
                den.masked_fill_(ng, 1.0)
            torch.div(V, den, out=sim)
            torch.mul(den, den, out=d2)
            torch.div(tm.C, d2, out=scl)
        else:                                        # "raw": the value itself, unit scale
            if tm.guard is None:
                return V, None
            sim.copy_(V).masked_fill_(ng, 0.0)
            scl.copy_(tm.guard)                      # bool -> dtype, same as guard.to(dtype)
            return sim, scl
        if tm.guard is not None:
            sim.masked_fill_(ng, 0.0)
            scl.masked_fill_(ng, 0.0)
        return sim, scl

    if red == "tanimoto":
        denom = tm.norm - V
        if tm.guard is not None:
            denom = denom.masked_fill(~tm.guard, 1.0)
        sim = V / denom
        scale = tm.norm / (denom * denom)
    elif red == "tversky":
        denom = tm.k * V + tm.C
        if tm.guard is not None:
            denom = denom.masked_fill(~tm.guard, 1.0)
        sim = V / denom
        scale = tm.C / (denom * denom)
    else:                                            # "raw": the value itself, unit scale
        sim = V
        scale = None
    if tm.guard is not None:
        sim = sim.masked_fill(~tm.guard, 0.0)
        if scale is None:
            scale = tm.guard.to(V.dtype)
        else:
            scale = scale.masked_fill(~tm.guard, 0.0)
    return sim, scale


def _step_generic(pr: Problem, st: _State, *, update=True):
    """One fine step of a non-pharm-style mode. Evaluates every term (value-only terms without
    their gradient), tracks the best pose on the blended score, forms the blended descent
    gradient and (when ``update``) applies the tangent-projected Adam step."""
    # Shape+color modes fuse their two gradient kernels into one launch where it fits; None
    # when it does not apply (CPU tensors, or a pad past the fused kernel's single tile).
    fused = None
    if pr.spec.fused_pair and len(pr.terms) == 2:
        fused = T.evaluate_fused_pair(pr.terms[0].spec, pr.terms[1].spec, pr.terms[0].inputs,
                                      pr.terms[1].inputs, st.q, st.t, pr.params)
    score = None
    first = True
    for ti, tm in enumerate(pr.terms):
        if not tm.spec.grad:
            V, _, _ = T.evaluate(tm.spec, tm.inputs, st.q, st.t, need_grad=False)
            sim = _reduce_value(tm, V, pr)
            contrib = sim if tm.weight == 1.0 else sim * tm.weight
            score = contrib if score is None else score + contrib
            continue
        if fused is not None:
            V, dQ, dT = fused[ti]
        else:
            V, dQ, dT = T.evaluate(tm.spec, tm.inputs, st.q, st.t, seeds_per_mol=pr.S_fine,
                                   poses_per_cta=pr.P_cta)
        sim, scale = _reduce_grad_term(tm, V)
        contrib = sim if tm.weight == 1.0 else sim * tm.weight
        score = contrib if score is None else score + contrib
        sc = None if scale is None else scale.unsqueeze(1)
        if first:
            if sc is None:
                st.gq.copy_(dQ); st.gt.copy_(dT)
            else:
                torch.mul(dQ, sc, out=st.gq); torch.mul(dT, sc, out=st.gt)
            st.gq.neg_(); st.gt.neg_()
            if tm.weight != 1.0:
                st.gq.mul_(tm.weight); st.gt.mul_(tm.weight)
            first = False
        else:
            tq = dQ if sc is None else dQ * sc
            tt = dT if sc is None else dT * sc
            st.gq.add_(tq * (-tm.weight) if tm.weight != 1.0 else -tq)
            st.gt.add_(tt * (-tm.weight) if tm.weight != 1.0 else -tt)
    _track_best(st, score)
    if update:
        fused_adam_qt_with_tangent_proj(st.q, st.t, st.gq, st.gt, st.mq, st.vq, st.mt, st.vt, st.lr)
    return score


def _step_pharm(pr: Problem, st: _State, *, update=True):
    """One fine step of a pharm-style mode (unit-normalised ``q`` into the kernel, the
    normalisation Jacobian on the way back, the guarded / clamped pharmacophore similarity,
    an explicit tangent projection and the un-projected fused Adam)."""
    tm = pr.terms[0]
    q_unit = F.normalize(st.q, dim=1)
    O, dQ_raw, dT_raw = T.evaluate(tm.spec, tm.inputs, q_unit, st.t)
    if pr.similarity == "tanimoto":
        denom = tm.norm - O
        score = torch.where(denom > 1e-8, O / denom, torch.zeros_like(O))
        scale = -tm.norm / (denom * denom)
    else:
        D = tm.C
        score = torch.clamp_max(torch.where(D > 1e-8, O / D, torch.zeros_like(O)), 1.0)
        active = (O < D).to(dtype=dQ_raw.dtype)
        scale = -active / D
    sgrad_q = scale.unsqueeze(1) * dQ_raw
    sgrad_t = scale.unsqueeze(1) * dT_raw
    qn = st.q.norm(dim=1, keepdim=True).clamp(min=1e-12)
    dQ = (sgrad_q - q_unit * (q_unit * sgrad_q).sum(1, keepdim=True)) / qn
    _track_best(st, score)
    if update:
        radial = (dQ * st.q).sum(dim=1, keepdim=True)
        dQ_tan = dQ - st.q * radial
        fused_adam_qt(st.q, st.t, dQ_tan, sgrad_t, st.mq, st.vq, st.mt, st.vt, st.lr)
    else:
        st.gq.copy_(dQ); st.gt.copy_(sgrad_t)
    return score


def _apply_adam_pharm(st: _State):
    radial = (st.gq * st.q).sum(dim=1, keepdim=True)
    dQ_tan = st.gq - st.q * radial
    fused_adam_qt(st.q, st.t, dQ_tan, st.gt, st.mq, st.vq, st.mt, st.vt, st.lr)

# CUDA-graph fine loop

class _GraphedFineTerms(_GraphedFineBase):
    """Capture one generic fine step; replay = N steps. Persistent buffers hold every term's
    inputs and constants for the bucket shape; ``_load`` copies a bucket in."""

    def __init__(self, pr: Problem, steps, lr):
        f = lambda x: torch.empty_like(x)
        self.bufs = []
        self.terms = []
        for tm in pr.terms:
            ti = tm.inputs
            ref = tuple(f(x) for x in ti.ref)
            fit = tuple(f(x) for x in ti.fit)
            tp = dict(ti.params)
            for key in ("_n_surf", "_m_surf"):
                if key in tp:
                    tp[key] = f(tp[key])
            new = _Term(tm.spec, T.TermInputs(ref, fit, f(ti.n_real), f(ti.m_real), ti.tables,
                                               None, tp), tm.weight)
            for a in ("norm", "C", "guard", "vaa", "vbb"):
                v = getattr(tm, a)
                setattr(new, a, None if v is None else f(v))
            new.k, new.sigma = tm.k, tm.sigma
            self.terms.append(new)
        self.pr = Problem()
        self.pr.spec, self.pr.terms, self.pr.similarity = pr.spec, self.terms, pr.similarity
        self.pr.S_fine, self.pr.P_cta, self.pr.params = pr.S_fine, pr.P_cta, pr.params
        self.qs = f(pr.q); self.ts = f(pr.t)
        self.st = _State(pr.q, pr.t, lr)
        self.pharm = bool(pr.spec.pharm_style)
        super().__init__(steps)

    def _step(self):
        if self.pharm:
            _step_pharm(self.pr, self.st)
        else:
            _step_generic(self.pr, self.st)

    def _load(self, pr: Problem):
        for dst, src in zip(self.terms, pr.terms):
            for d, s in zip(dst.inputs.ref, src.inputs.ref):
                d.copy_(s)
            for d, s in zip(dst.inputs.fit, src.inputs.fit):
                d.copy_(s)
            dst.inputs.n_real.copy_(src.inputs.n_real)
            dst.inputs.m_real.copy_(src.inputs.m_real)
            for key in ("_n_surf", "_m_surf"):
                if key in dst.inputs.params:
                    dst.inputs.params[key].copy_(src.inputs.params[key])
            for a in ("norm", "C", "guard", "vaa", "vbb"):
                v = getattr(src, a)
                if v is not None:
                    getattr(dst, a).copy_(v)
        self.qs.copy_(pr.q); self.ts.copy_(pr.t)

    def _reset(self):
        self.st.reset(self.qs, self.ts)

    def _result(self):
        return self.st.best, self.st.bq, self.st.bt

    @property
    def best(self):                                  # the base class's early-stop reads this
        return self.st.best


def _graph_key(pr: Problem, steps, lr):
    p = pr.params
    key = [pr.device.index, pr.spec.name]
    for tm in pr.terms:
        key += [tuple(tuple(x.shape) for x in tm.inputs.ref),      # avoid: per pair, or 1 shared
                tuple(int(x.shape[1]) for x in tm.inputs.fit)]
    key += [int(pr.q.shape[0]), pr.P, int(steps), round(float(lr), 5), pr.S_fine, pr.P_cta,
            pr.similarity]
    for name in sorted(p):
        v = p[name]
        if isinstance(v, (int, float, bool)):
            key.append((name, round(float(v), 6)))
        elif isinstance(v, str):
            key.append((name, v))
    return tuple(key)

# eager fine loop

def _eager(pr: Problem, steps_fine, lr, es_patience, es_tol):
    st = _State(pr.q, pr.t, lr)
    B, P = pr.B, pr.P
    prev_best = torch.full((B,), -float("inf"), device=pr.device, dtype=st.best.dtype)
    no_improve = 0
    step = -1
    for step in range(steps_fine):
        if pr.spec.pharm_style:
            _step_pharm(pr, st, update=False)
        else:
            _step_generic(pr, st, update=False)
        # Early-stop check every 5 steps, per pair (each pair's own best over its seeds), so
        # one converged pair cannot halt the rest of the bucket. One host sync per check.
        if step % 5 == 0:
            cur = st.best.view(B, P).amax(dim=1)
            improved = (cur - prev_best) > es_tol
            if not improved.any():
                no_improve += 1
                if no_improve >= es_patience:
                    break
            else:
                no_improve = 0
            prev_best = torch.where(improved, cur, prev_best)
        if pr.spec.pharm_style:
            _apply_adam_pharm(st)
        else:
            fused_adam_qt_with_tangent_proj(st.q, st.t, st.gq, st.gt, st.mq, st.vq, st.mt,
                                            st.vt, st.lr)
    ran = (step + 1) if steps_fine else 0
    _record_steps(ran, steps_fine, ran < steps_fine)
    return st.best, st.bq, st.bt

# entry point

def align(spec, chans: dict, *, params: dict, num_seeds: int, steps_fine: int, lr: float,
          early_stop_patience: int, early_stop_tol: float = 1e-5, seeds=None,
          ref_shared: bool = False, trans_centers=None, trans_centers_real=None,
          num_repeats_per_trans: int = 10, topk: int = 30, self_overlaps=None):
    """Batched coarse-to-fine alignment of one bucket of pairs in mode ``spec``.

    Returns ``(score (B,), q (B,4), t (B,3))`` -- each pair's best over its seeds. The graph
    loop, the fused CPU loop and the eager loop are tried in that order under the mode's gates
    (``graph_budget`` / ``cpu_fused``), and every one of them runs the same per-step arithmetic.
    """
    pr = assemble(spec, chans, params=params, num_seeds=num_seeds, seeds=seeds,
                  ref_shared=ref_shared, trans_centers=trans_centers,
                  trans_centers_real=trans_centers_real,
                  num_repeats_per_trans=num_repeats_per_trans, topk=topk,
                  self_overlaps=self_overlaps)
    B, P = pr.B, pr.P
    PK = int(pr.q.shape[0])
    best = bq = bt = None

    # --- CUDA-graph fast path -----------------------------------------------------------------
    if (pr.device.type == "cuda" and pr.dtype == torch.float32 and spec.graph_budget is not None
            and PK <= graph_cap(pr.work, budget=spec.graph_budget)):
        try:
            best, bq, bt = run_graphed(
                lambda: _GraphedFineTerms(pr, steps_fine, lr), _graph_key(pr, steps_fine, lr),
                (pr,), es_patience=(0 if spec.graph_full_steps else early_stop_patience),
                es_tol=early_stop_tol, es_seeds=P)
        except Exception:
            best = None                                   # capture failed -> next path

    # --- fused CPU (numba) fast path -----------------------------------------------------------
    if (best is None and pr.device.type == "cpu" and pr.dtype == torch.float32
            and spec.cpu_fused and pr.S_fine == 1
            and (spec.cpu_fused_max_pad is None or max(pr.pads) <= spec.cpu_fused_max_pad)):
        try:
            from ..kernels.cpu_fused import run_fused
            best, bq, bt = run_fused(pr, steps_fine, lr, early_stop_patience, early_stop_tol)
        except Exception:
            best = None                                   # fused failed -> eager

    if best is None:
        best, bq, bt = _eager(pr, steps_fine, lr, early_stop_patience, early_stop_tol)

    # --- gather each pair's best seed ---------------------------------------------------------
    final = best.view(B, P)
    idx = final.argmax(dim=1)
    ar = torch.arange(B, device=pr.device)
    out_score = final[ar, idx]
    out_q = bq.view(B, P, 4)[ar, idx]
    out_t = bt.view(B, P, 3)[ar, idx]
    if pr.c_ref is not None:
        # fold the centring back so the transform maps original fit -> original ref
        R = _rotation_matrix_from_unit_quat(F.normalize(out_q, dim=1))
        out_t = out_t - torch.einsum("bij,bj->bi", R, pr.c_fit) + pr.c_ref
    return out_score, out_q, out_t
