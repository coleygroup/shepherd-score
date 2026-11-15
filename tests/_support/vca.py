"""Shared fixtures for the ``vol_color_avoid`` accel tests: molecules, avoid clouds, padded
engine channels built straight from the channel registry, and the reference score at an engine
pose."""
import warnings

import numpy as np
import torch

SMILES = ("CC(C)Cc1ccc(cc1)C(C)C(=O)O",          # ibuprofen
          "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",        # caffeine
          "c1ccccc1CCO",
          "OC(=O)c1ccccc1O",
          "CC(=O)Nc1ccc(O)cc1")


def mol(smiles, seed=0):
    from shepherd_score.conformer_generation import embed_conformer_from_smiles
    from shepherd_score.container import Molecule
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rd = embed_conformer_from_smiles(smiles, MMFF_optimize=True, random_seed=seed)
    return Molecule(rd, pharm_multi_vector=False)


def pair(ref_smi, fit_smi, device="cpu"):
    from shepherd_score.container import MoleculePair
    ref, fit = mol(ref_smi), mol(fit_smi)
    for m in (ref, fit):              # centered up front, so clouds built from them are input-frame
        m.center_to(m.atom_pos.mean(0))
    return MoleculePair(ref, fit, device=torch.device(device))


def far_cloud(m):
    """One point 100 A away: never intrudes, so the penalty is 0."""
    return (np.asarray(m.atom_pos).mean(0) + 100.0).astype(np.float32)[None, :]


def near_cloud(m, n=None):
    """Points on one side of the reference shape, inside it, so the penalty is active."""
    pos = np.asarray(m.atom_pos, dtype=np.float64)
    c = pos.mean(0)
    axis = np.linalg.svd(pos - c)[2][0]
    side = pos[(pos - c) @ axis > 0] + 0.3 * axis
    return (side if n is None else side[:n]).astype(np.float32)


def _pad(arrs, pad, dtype, device):
    W = max(max(len(a) for a in arrs), 1)
    shape = (len(arrs), W) + tuple(np.asarray(arrs[0]).shape[1:])
    out = np.full(shape, pad, dtype=dtype)
    for i, a in enumerate(arrs):
        out[i, :len(a)] = a
    n = torch.tensor([len(a) for a in arrs], dtype=torch.int32, device=device)
    return torch.as_tensor(out, device=device), n


def chans(pairs, clouds, device="cpu"):
    """Padded engine channels (``atoms``, ``pharm_ancs``, ``pharm_types``, ``avoid``) for
    ``pairs``, read through the channel registry, with one avoid cloud per pair."""
    from shepherd_score.accel.channels import CHANNELS
    from shepherd_score.accel.drivers.engine import Batch
    out = {}
    for name in ("atoms", "pharm_ancs", "pharm_types"):
        ch = CHANNELS[name]
        dt = np.int64 if ch.dtype == "int64" else np.float32
        r, nr = _pad([ch.read(p.ref_molec) for p in pairs], ch.pad, dt, device)
        f, mr = _pad([ch.read(p.fit_molec) for p in pairs], ch.pad, dt, device)
        out[name] = Batch(r, f, nr, mr)
    out["pharm_types"] = Batch(out["pharm_types"].ref, out["pharm_types"].fit,
                               out["pharm_ancs"].n_real, out["pharm_ancs"].m_real)
    av, na = _pad([np.asarray(c, np.float32) for c in clouds], 0.0, np.float32, device)
    out["avoid"] = Batch(av, None, na, None)
    return out


def ref_score_at(p, cloud, q, t, **mode_kw):
    """The PyTorch reference ``vol_color_avoid`` score of ``p`` with the fit moved by the engine
    pose ``(q, t)`` (one pose)."""
    from shepherd_score.alignment.utils.se3 import quaternions_to_SE3_batch
    from shepherd_score.modes import vol_color_avoid
    mode = vol_color_avoid(avoid_points=cloud, **mode_kw)
    se3 = quaternions_to_SE3_batch(q.detach().cpu().double()[None], t.detach().cpu().double()[None])[0]
    ref, fit = p._mode_inputs(mode, p._as_tensor)
    posed = mode.pose(fit, se3.to(torch.float32))
    with torch.no_grad():
        return float(mode.score(ref, posed))
