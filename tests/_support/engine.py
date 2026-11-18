"""Run the accel engine on padded channel tensors, for tests that bypass ``MoleculePair``.

``run`` keeps the defaults of the per-mode driver functions these tests used to call
(``num_seeds=50, steps_fine=100, lr=0.075``, the mode's early-stop patience).
"""
import torch

from shepherd_score.accel._modes import SPECS
from shepherd_score.accel.drivers.engine import Batch, align


def batch(ref, fit, n_real=None, m_real=None) -> Batch:
    """A :class:`Batch` with real counts defaulting to the padded widths."""
    B = ref.shape[0]
    if n_real is None:
        n_real = ref.new_full((B,), ref.shape[1], dtype=torch.int32)
    if m_real is None and fit is not None:
        m_real = fit.new_full((B,), fit.shape[1], dtype=torch.int32)
    return Batch(ref, fit, n_real, m_real)


def run(mode, chans, *, num_seeds=50, steps_fine=100, lr=0.075, early_stop_patience=None,
        early_stop_tol=1e-5, seeds=None, topk=30, **params):
    """``engine.align`` for ``mode`` on ``chans``; returns ``(scores, q, t)``."""
    spec = SPECS[mode]
    p = dict(spec.params)
    p.update(params)
    p["lr"] = float(lr)
    patience = spec.patience if early_stop_patience is None else early_stop_patience
    return align(spec, chans, params=p, num_seeds=int(num_seeds), steps_fine=int(steps_fine),
                 lr=float(lr), early_stop_patience=int(patience),
                 early_stop_tol=float(early_stop_tol), seeds=seeds, topk=topk)


def combo_chans(cwh_1, cwh_2, centers_1, centers_2, points_1, points_2, partial_1, partial_2,
                surf_esp_1, surf_esp_2, radii_1, radii_2, n_centers, m_centers, n_cwh, m_cwh,
                n_surf, m_surf):
    """The engine's channel dict for ``vol_and_surf_esp``. The spec reads ``centers`` from
    ``atoms`` at ``alpha == 0.81`` and from ``surf`` otherwise."""
    wh = batch(cwh_1, cwh_2, n_cwh, m_cwh)
    sf = batch(points_1, points_2, n_surf, m_surf)
    return {"cwh": wh, "partial": batch(partial_1, partial_2, wh.n_real, wh.m_real),
            "radii": batch(radii_1, radii_2, wh.n_real, wh.m_real),
            "surf": sf, "surf_esp": batch(surf_esp_1, surf_esp_2, sf.n_real, sf.m_real),
            "atoms": batch(centers_1, centers_2, n_centers, m_centers)}
