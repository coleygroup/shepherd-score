"""Gates for the translation-seeded accelerated path (``trans_init=True``): the coarse grid is
built from the cloud each mode's ``coarse_channel`` names, the grid size follows the legacy
contract, and self-copies still score ~1.0. Surfaces and charges are injected so the file needs
neither open3d nor xtb.
"""
import warnings

import numpy as np
import pytest

try:
    import torch
    TORCH = True
except ImportError:
    TORCH = False

pytestmark = pytest.mark.skipif(not TORCH, reason="PyTorch required")

IBU = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
CAF = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
N_SURF = 75

# Modes checked for ``trans_init`` -> the cloud their coarse grid is built from.
# Hardcoded on purpose: deriving it from the spec would only restate the code under test.
GRID_CLOUD = {
    "vol": "atoms",
    "surf": "surf",
    "vol_esp": "atoms",
    "surf_esp": "surf",
    "vol_and_surf_esp": "surf",
    "pharm": "pharm_ancs",
    "vol_color": "atoms",
}

KW = {
    "vol": dict(alpha=0.81),
    "surf": dict(alpha=0.81),
    "vol_esp": dict(alpha=0.81, lam=0.3),
    "surf_esp": dict(alpha=0.81, lam=0.3),
    "vol_and_surf_esp": dict(alpha=0.81),
    "pharm": {},
    "vol_color": dict(alpha=0.81),
}


def _mol(smiles, seed=0):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from shepherd_score.container import Molecule
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rd = Chem.AddHs(Chem.MolFromSmiles(smiles))
        p = AllChem.ETKDGv3()
        p.randomSeed = seed
        assert AllChem.EmbedMolecule(rd, p) == 0, smiles
    rng = np.random.default_rng(seed)
    return Molecule(rd, pharm_multi_vector=False,
                    surface_points=(rng.standard_normal((N_SURF, 3)) * 3.0).astype(np.float32),
                    electrostatics=rng.standard_normal(N_SURF).astype(np.float32),
                    partial_charges=(rng.standard_normal(rd.GetNumAtoms()) * 0.2
                                     ).astype(np.float32))


def _pair(a, b):
    from shepherd_score.container import MoleculePair
    return MoleculePair(a, b, do_center=True, device=torch.device("cpu"))


def _widths(mol):
    """Real-point count of each cloud the coarse grid could be built from."""
    return {"atoms": int(np.asarray(mol.atom_pos).shape[0]),
            "surf": int(np.asarray(mol.surf_pos).shape[0]),
            "pharm_ancs": int(np.asarray(mol.pharm_ancs).shape[0])}


def _spy(monkeypatch):
    """Record the reference cloud size and grid size the coarse path is called with."""
    from shepherd_score.accel.drivers import engine
    seen = {}
    real = engine.build_coarse_grid

    def spy(A_batch, B_batch, N_real, M_real, **kw):
        q, t = real(A_batch, B_batch, N_real, M_real, **kw)
        seen.setdefault("n", int(N_real[0].item()))
        seen.setdefault("g", int(q.shape[1]))
        return q, t

    monkeypatch.setattr(engine, "build_coarse_grid", spy)
    return seen


@pytest.mark.parametrize("mode", sorted(GRID_CLOUD))
def test_coarse_grid_is_built_from_the_documented_cloud(mode, monkeypatch):
    """The grid must be built from the mode's ``coarse_channel`` cloud."""
    from shepherd_score.container import MoleculePairBatch

    ref, fit = _mol(IBU), _mol(CAF, seed=1)
    w = _widths(ref)
    assert len(set(w.values())) == len(w), f"fixture clouds are not distinguishable: {w}"

    seen = _spy(monkeypatch)
    MoleculePairBatch([_pair(ref, fit)])._align_mode(
        mode, backend="numba", trans_init=True, **KW[mode])

    assert seen, f"{mode}: trans_init=True did not reach the coarse-grid path"
    want = GRID_CLOUD[mode]
    assert seen["n"] == w[want], (
        f"{mode}: coarse grid built from a {seen['n']}-point cloud, expected the {want} cloud "
        f"({w[want]} points). Cloud widths: {w}")


def test_grid_size_follows_the_legacy_contract(monkeypatch):
    """G = 10*P + 5 for P translation centers, with P taken from the reference molecule's atoms."""
    from shepherd_score.container import MoleculePairBatch

    ref, fit = _mol(IBU), _mol(CAF, seed=1)
    p_count = int(np.asarray(ref.atom_pos).shape[0])
    seen = _spy(monkeypatch)
    MoleculePairBatch([_pair(ref, fit)]).align_with_vol_esp(
        backend="numba", trans_init=True, alpha=0.81, lam=0.3)
    assert seen["g"] == 10 * p_count + 5, \
        f"grid size {seen['g']} != 10*{p_count}+5"


def _self_copy(mode):
    from shepherd_score.accel._modes import MODE_ATTRS
    from shepherd_score.container import MoleculePairBatch
    p = _pair(_mol(IBU), _mol(IBU))
    MoleculePairBatch([p])._align_mode(
        mode, backend="numba", trans_init=True, **KW[mode])
    return float(getattr(p, MODE_ATTRS[mode][1]))


@pytest.mark.parametrize("mode", [m for m in sorted(GRID_CLOUD) if m != "vol_and_surf_esp"])
def test_self_copy_scores_one_through_the_coarse_path(mode):
    """A self-copy still scores ~1.0 when the fine-loop starts come from the coarse grid."""
    s = _self_copy(mode)
    assert 0.97 <= s <= 1.0 + 1e-5, f"{mode} trans_init self-copy scored {s}"


def test_combo_self_copy_falls_short_of_one_as_it_always_has():
    """``vol_and_surf_esp`` cannot reach identity from grid starts; pinned so any change is noticed."""
    s = _self_copy("vol_and_surf_esp")
    assert 0.45 <= s < 0.97, (
        f"vol_and_surf_esp trans_init self-copy scored {s}; the shipped behaviour is ~0.54. "
        "Above 0.97 means the coarse path now reaches identity -- a real improvement, but check "
        "it against the rest of the parity gates before widening this bound.")


@pytest.mark.cuda
@pytest.mark.skipif(not (TORCH and torch.cuda.is_available()), reason="CUDA required")
@pytest.mark.parametrize("mode", sorted(GRID_CLOUD))
def test_triton_coarse_grid_matches_numba(mode, monkeypatch):
    """The GPU builds the grid from the same cloud and lands in the same basin as the CPU."""
    from shepherd_score.accel._modes import MODE_ATTRS
    from shepherd_score.container import MoleculePairBatch

    ref, fit = _mol(IBU), _mol(CAF, seed=1)
    out, clouds = [], []
    for backend in ("numba", "triton"):
        seen = _spy(monkeypatch)
        p = _pair(ref, fit)
        MoleculePairBatch([p])._align_mode(
            mode, backend=backend, trans_init=True, **KW[mode])
        out.append(float(getattr(p, MODE_ATTRS[mode][1])))
        clouds.append(seen.get("n"))

    assert clouds[0] == clouds[1],         f"{mode}: triton built its grid from a {clouds[1]}-point cloud, numba from {clouds[0]}"
    assert abs(out[0] - out[1]) <= 2e-3,         f"{mode}: trans_init triton {out[1]} vs numba {out[0]}"
