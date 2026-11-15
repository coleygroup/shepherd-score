"""Batched-numba gates for the CPU-testable experimental modes: self-copy scores 1.0, and the
batched path matches the per-pair torch reference on a distinct pair (loose tolerance, since
the two use different multi-start seed sets). Triton parity is CUDA-only. ``vol_fukui`` reads a
synthetic field injected via ``Molecule(fukui=...)``, so no xTB binary is needed.
"""
import warnings
import numpy as np
import pytest

from .utils import align_per_pair

try:
    import torch
    TORCH = True
except ImportError:
    TORCH = False

pytestmark = pytest.mark.skipif(not TORCH, reason="PyTorch required")

IBU = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
CAF = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


def _synthetic_fukui(rd):
    """Deterministic signed per-atom field in with-H order (the real field needs xTB)."""
    z = np.array([a.GetAtomicNum() for a in rd.GetAtoms()], dtype=np.float32)
    return (0.2 * ((z % 4) - 1.5)).astype(np.float32)


def _mol(smiles):
    from shepherd_score.conformer_generation import embed_conformer_from_smiles
    from shepherd_score.container import Molecule
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rd = embed_conformer_from_smiles(smiles, MMFF_optimize=True, random_seed=0)
    # pharm_multi_vector=False builds the pharmacophores the pharm/color modes need
    return Molecule(rd, pharm_multi_vector=False, fukui=_synthetic_fukui(rd))


@pytest.fixture(scope="module")
def mols():
    return _mol(IBU), _mol(CAF)


# per mode: (align kwarg name for repeats, steps) come from MODE_SEEDS/MODE_STEPS
MODES = [
    ("vol_tversky", "sim_aligned_vol_tversky"),
    ("vol_lipo", "sim_aligned_vol_lipo"),
    ("vol_fukui", "sim_aligned_vol_fukui"),
    ("vol_esp_tversky", "sim_aligned_vol_esp_tversky"),
    # surface-free SI modes only; the surface modes need Open3D
    ("vol_mr", "sim_aligned_vol_mr"),
    ("vol_lipo_tversky", "sim_aligned_vol_lipo_tversky"),
    ("vol_color_tversky", "sim_aligned_vol_color_tversky"),
    ("vol_atomtype", "sim_aligned_vol_atomtype"),
    ("vol_pharm", "sim_aligned_vol_pharm"),
    ("pharm_tversky", "sim_aligned_pharm_tversky"),
]


@pytest.mark.parametrize("mode,attr", MODES)
def test_numba_self_copy_is_one(mode, attr, mols):
    """A molecule aligned to a copy of itself scores 1.0 under the numba backend."""
    from shepherd_score.container import MoleculePair, MoleculePairBatch
    ibu, _ = mols
    b = MoleculePairBatch([MoleculePair(_mol(IBU), _mol(IBU), do_center=True,
                                        device=torch.device("cpu"))])
    scores, *_ = b._align_mode(mode, backend="numba")
    assert np.isclose(float(scores[0]), 1.0, atol=1e-4), f"{mode} numba self-copy {scores[0]} != 1.0"


@pytest.mark.parametrize("mode,attr", MODES)
def test_numba_batched_matches_per_pair(mode, attr, mols):
    """Batched-numba lands in the per-pair reference's basin at the shipped seed/step budget."""
    from shepherd_score.accel._modes import MODE_SEEDS, MODE_STEPS
    from shepherd_score.container import MoleculePair, MoleculePairBatch
    ibu, caf = mols
    nr, ns = MODE_SEEDS[mode], MODE_STEPS[mode]

    mp = MoleculePair(_mol(IBU), _mol(CAF), do_center=True, device=torch.device("cpu"))
    ref = align_per_pair(mp, mode, num_repeats=nr, max_num_steps=ns)

    b = MoleculePairBatch([MoleculePair(_mol(IBU), _mol(CAF), do_center=True,
                                        device=torch.device("cpu"))])
    scores, *_ = b._align_mode(mode, backend="numba")
    bat = float(scores[0])

    assert abs(ref - bat) < 1e-2, f"{mode}: per-pair {ref:.5f} vs batched-numba {bat:.5f}"


@pytest.mark.cuda
@pytest.mark.skipif(not (TORCH and torch.cuda.is_available()), reason="CUDA required")
@pytest.mark.parametrize("mode,attr", MODES)
def test_triton_matches_numba(mode, attr, mols):
    """The Triton batched driver agrees with the numba one on the same pair."""
    from shepherd_score.container import MoleculePair, MoleculePairBatch
    bn = MoleculePairBatch([MoleculePair(_mol(IBU), _mol(CAF), do_center=True,
                                         device=torch.device("cpu"))])
    sc_n, *_ = bn._align_mode(mode, backend="numba")
    bt = MoleculePairBatch([MoleculePair(_mol(IBU), _mol(CAF), do_center=True,
                                         device=torch.device("cuda"))])
    sc_t, *_ = bt._align_mode(mode, backend="triton")
    assert abs(float(sc_n[0]) - float(sc_t[0])) < 1e-2, (
        f"{mode}: numba {float(sc_n[0]):.5f} vs triton {float(sc_t[0]):.5f}")
