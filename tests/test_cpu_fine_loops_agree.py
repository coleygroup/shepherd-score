"""The fused numba CPU fine loop and the eager torch fine loop must agree for every mode.

``drivers/engine.align`` silently falls back from the fused loop to the eager one, so the two
must score alike. Bounds cover both the SVML (fp32 SoA) and non-SVML (fp64 AoS) kernels;
``surf_esp``, the most shape-degenerate mode, gets its own.
"""
from pathlib import Path

import numpy as np
import pytest

try:
    import torch
    TORCH = True
except ImportError:
    TORCH = False

pytestmark = pytest.mark.skipif(not TORCH, reason="PyTorch required")

# (max |delta|, max relative %) per mode: roughly an order of magnitude above the fp32 drift
# seen in either kernel environment, so a failure means the loops genuinely diverged.
TOL_DEFAULT = (1.0e-5, 0.01)
TOL = {"surf_esp": (1.0e-3, 0.2)}

SMILES = ["CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
          "CC(=O)Oc1ccccc1C(=O)O", "c1ccc2c(c1)cccc2O"]
N_SURF = 60

# Conformers are frozen on disk: re-embedding with ETKDG gives new geometries on every RDKit
# release, and on some of those the SVML kernels' fp32 drift exceeds the bounds above.
_SDF = Path(__file__).parent / "data" / "cpu_fine_loops_mols.sdf"


def _load_conformers():
    from rdkit import Chem
    mols = {m.GetProp("smiles"): m for m in Chem.SDMolSupplier(str(_SDF), removeHs=False)}
    assert list(mols) == SMILES, f"{_SDF.name} is out of sync with SMILES"
    return mols


_CONFORMERS = _load_conformers()

# Modes whose call needs an explicit keyword; everything else takes the registry defaults.
KW = {"surf": {"alpha": 0.81}, "surf_esp": {"alpha": 0.81, "lam": 0.3},
      "surf_tversky": {"alpha": 0.81}, "surf_esp_tversky": {"alpha": 0.81, "lam": 0.3},
      "vol_and_surf_esp": {"alpha": 0.81}, "vol_and_surf_esp_tversky": {"alpha": 0.81}}


def _mol(smiles, seed, n_surf=None):
    from shepherd_score.container import Molecule
    rd = _CONFORMERS[smiles]
    rng = np.random.default_rng(seed)
    n = rd.GetNumAtoms()
    ns = N_SURF if n_surf is None else int(n_surf)
    return Molecule(rd, pharm_multi_vector=False,
                    surface_points=(rng.standard_normal((ns, 3)) * 3.0).astype(np.float32),
                    electrostatics=rng.standard_normal(ns).astype(np.float32),
                    fukui=(rng.standard_normal(n) * 0.1).astype(np.float32),
                    partial_charges=(rng.standard_normal(n) * 0.2).astype(np.float32))


@pytest.fixture(scope="module")
def mols():
    return [_mol(s, i) for i, s in enumerate(SMILES)]


def _scores(mode, mols, force_eager):
    """Align mols[0] against every mol, optionally with the fused loop disabled."""
    from shepherd_score.accel._modes import MODE_ATTRS
    from shepherd_score.accel.kernels import cpu_fused
    from shepherd_score.container import MoleculePair, MoleculePairBatch

    real = cpu_fused.run_fused
    if force_eager:
        def boom(*a, **k):
            raise RuntimeError("fused disabled by the test")
        cpu_fused.run_fused = boom
    try:
        pairs = [MoleculePair(mols[0], f, do_center=True, device=torch.device("cpu"))
                 for f in mols]
        batch = MoleculePairBatch(pairs)
        extra = {}
        if mode == "vol_avoid":                       # the one mode with a third input
            extra["avoid_points"] = np.asarray(mols[0].atom_pos, dtype=np.float32) + 40.0
        batch._align_mode(mode, backend="numba", **extra, **KW.get(mode, {}))
        return [float(getattr(p, MODE_ATTRS[mode][1])) for p in pairs]
    finally:
        cpu_fused.run_fused = real


def _all_modes():
    from shepherd_score.accel._modes import CANONICAL_MODES
    return list(CANONICAL_MODES)


@pytest.mark.parametrize("mode", _all_modes())
def test_fused_and_eager_cpu_loops_agree(mode, mols):
    from shepherd_score.accel._modes import spec_of

    fused = _scores(mode, mols, force_eager=False)
    eager = _scores(mode, mols, force_eager=True)

    assert all(np.isfinite(fused)) and all(np.isfinite(eager)), (fused, eager)
    d = max(abs(a - b) for a, b in zip(fused, eager))
    rel = max(abs(a - b) / max(abs(a), 1e-12) for a, b in zip(fused, eager)) * 100

    max_abs, max_rel = TOL.get(mode, TOL_DEFAULT)

    assert d <= max_abs and rel <= max_rel, (
        f"{mode}: fused vs eager max|d| {d:.3e} ({rel:.4f}% relative) exceeds "
        f"{max_abs:.1e} / {max_rel}%.\n  fused {fused}\n  eager {eager}\n"
        "  (SVML widens this gap; see the module docstring for both measurements)")

    if not spec_of(mode).cpu_fused:
        assert fused == eager, \
            f"{mode} sets cpu_fused=False, so both runs take the eager loop and must be identical"


def test_disabling_the_fused_loop_actually_changes_the_path(mols, monkeypatch):
    """Forcing the fused loop off must reach the eager loop, or the parity test is vacuous."""
    from shepherd_score.accel.drivers import engine

    seen = []
    real = engine._eager

    def spy(*a, **k):
        seen.append(1)
        return real(*a, **k)

    monkeypatch.setattr(engine, "_eager", spy)
    _scores("vol", mols, force_eager=False)
    assert not seen, "vol took the eager loop even with the fused loop available"
    _scores("vol", mols, force_eager=True)
    assert seen, "forcing the fused loop off did not reach the eager loop"


# --- surf_esp at realistic surface sizes -----------------------------------------------------
# The tests above run at N_SURF; these check surf_esp at 128-300 surface points.

@pytest.mark.parametrize("n_surf", [128, 200, 300])
def test_surf_esp_stays_fused_above_the_old_cap(n_surf, monkeypatch):
    """surf_esp must take the fused CPU loop at realistic surface-point counts."""
    from shepherd_score.accel.drivers import engine
    from shepherd_score.accel.kernels import cpu_fused
    from shepherd_score.container import MoleculePair, MoleculePairBatch

    seen = []
    rf, re_ = cpu_fused.run_fused, engine._eager
    monkeypatch.setattr(cpu_fused, "run_fused",
                        lambda *a, **k: (seen.append("fused"), rf(*a, **k))[1])
    monkeypatch.setattr(engine, "_eager", lambda *a, **k: (seen.append("eager"), re_(*a, **k))[1])

    a, b = _mol(SMILES[0], 0, n_surf=n_surf), _mol(SMILES[1], 1, n_surf=n_surf)
    pair = MoleculePair(a, b, do_center=True, device=torch.device("cpu"))
    MoleculePairBatch([pair]).align_with_surf_esp(backend="numba", alpha=0.81)
    assert "fused" in seen and "eager" not in seen, (
        "surf_esp at %d surface points took %s; it must take the fused CPU loop"
        % (n_surf, seen or ["?"]))


@pytest.mark.parametrize("n_surf", [128, 200])
def test_surf_esp_pose_agrees_above_the_old_cap(n_surf):
    """Fused and eager surf_esp must land in the same pose basin, not merely the same score."""
    from shepherd_score.accel.kernels import cpu_fused
    from shepherd_score.container import MoleculePair, MoleculePairBatch

    mols = [_mol(s, i, n_surf=n_surf) for i, s in enumerate(SMILES)]

    def poses(force_eager):
        real = cpu_fused.run_fused
        if force_eager:
            def boom(*a, **k):
                raise RuntimeError("fused disabled by the test")
            cpu_fused.run_fused = boom
        try:
            pairs = [MoleculePair(mols[0], f, do_center=True, device=torch.device("cpu"))
                     for f in mols]
            MoleculePairBatch(pairs).align_with_surf_esp(backend="numba", alpha=0.81)
            return [np.asarray(p.transform_surf_esp, dtype=np.float64) for p in pairs]
        finally:
            cpu_fused.run_fused = real

    worst = 0.0
    for A, B in zip(poses(False), poses(True)):
        R = A[:3, :3] @ B[:3, :3].T
        worst = max(worst, float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))))
    assert worst < 5.0, (
        "surf_esp fused vs eager rotated by %.3f deg at %d surface points: the fused loop has "
        "landed in a different basin, which is what the old cpu_fused_max_pad=100 guarded "
        "against. Measured 0.045 deg on real surfaces when the cap was lifted." % (worst, n_surf))
