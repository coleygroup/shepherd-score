"""Tests for the directionless "color" pharmacophore scoring and the ``vol_color`` combined
shape + color alignment mode: torch vs NumPy parity, self-overlap, precomputed directionless
self overlaps, the combo scorer, RDKit BaseFeatures featurization, and the alignment path.
"""
import warnings

import numpy as np
import pytest

from shepherd_score.score.pharmacophore_scoring_np import (
    get_overlap_pharm_np,
    get_pharm_combo_score as get_pharm_combo_score_np,
)
from ..scoring.test_pharmacophore_scoring import TestDataGenerator

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available - skipping torch vol_color tests")

if TORCH_AVAILABLE:
    from shepherd_score.score.pharmacophore_scoring import (
        get_overlap_pharm as get_overlap_pharm_torch,
        get_pharm_combo_score as get_pharm_combo_score_torch,
    )

RTOL = 1e-5
ATOL = 1e-7

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")


# --------------------------------------------------------------------------- #
# Directionless scoring parity (torch vs NumPy oracle)
# --------------------------------------------------------------------------- #
class TestDirectionlessScoring:

    @pytest.mark.parametrize("sizes", [(5, 3), (10, 8), (20, 15), (1, 1)])
    @pytest.mark.parametrize("similarity", ['tanimoto', 'tversky', 'tversky_ref', 'tversky_fit'])
    def test_directionless_torch_matches_np(self, sizes, similarity):
        """Directionless overlap: torch must match the NumPy oracle for all types."""
        n1, n2 = sizes
        ptype_1, ptype_2, anchors_1, anchors_2, vectors_1, vectors_2 = \
            TestDataGenerator.generate_pharmacophore_data(n1, n2, seed=42)

        result_np = get_overlap_pharm_np(
            ptype_1, ptype_2, anchors_1, anchors_2, vectors_1, vectors_2,
            similarity=similarity, directionless=True,
        )
        result_torch = get_overlap_pharm_torch(
            torch.from_numpy(ptype_1), torch.from_numpy(ptype_2),
            torch.from_numpy(anchors_1), torch.from_numpy(anchors_2),
            torch.from_numpy(vectors_1), torch.from_numpy(vectors_2),
            similarity=similarity, directionless=True,
        )
        assert np.allclose(result_torch.numpy(), result_np, rtol=RTOL, atol=ATOL), \
            f"torch {result_torch.numpy()} != np {result_np} (directionless, {similarity}, {sizes})"

    def test_directionless_differs_from_directional(self):
        """Sanity: with directional types present, directionless != directional."""
        ptype_1, ptype_2, anchors_1, anchors_2, vectors_1, vectors_2 = \
            TestDataGenerator.generate_pharmacophore_data(12, 10, seed=1)
        directional = get_overlap_pharm_np(ptype_1, ptype_2, anchors_1, anchors_2,
                                           vectors_1, vectors_2, directionless=False)
        directionless = get_overlap_pharm_np(ptype_1, ptype_2, anchors_1, anchors_2,
                                             vectors_1, vectors_2, directionless=True)
        assert not np.isclose(directional, directionless), \
            "directionless scoring unexpectedly identical to directional"

    def test_directionless_self_overlap_is_one(self):
        """Identical molecules scored directionless must give Tanimoto 1.0."""
        ptype_1 = np.array([0, 1, 2, 4, 5], dtype=np.int64)  # acceptor/donor/aromatic/halogen/cation
        anchors_1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
                             dtype=np.float32)
        vectors_1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]],
                             dtype=np.float32)
        vectors_1 = vectors_1 / np.linalg.norm(vectors_1, axis=1, keepdims=True)

        res_np = get_overlap_pharm_np(ptype_1, ptype_1, anchors_1, anchors_1,
                                      vectors_1, vectors_1, directionless=True)
        assert np.allclose(res_np, 1.0, rtol=1e-10, atol=1e-10)

        res_torch = get_overlap_pharm_torch(
            torch.from_numpy(ptype_1), torch.from_numpy(ptype_1),
            torch.from_numpy(anchors_1), torch.from_numpy(anchors_1),
            torch.from_numpy(vectors_1), torch.from_numpy(vectors_1),
            directionless=True,
        )
        assert np.allclose(res_torch.numpy(), 1.0, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("similarity", ['tanimoto', 'tversky', 'tversky_ref', 'tversky_fit', (0.7, 0.3)])
    def test_directionless_precomputed_self_overlaps(self, similarity):
        """Directionless self overlaps from compute_self_overlaps_pharm give the same score as
        recomputing them, unbatched and batched, and match the NumPy oracle."""
        from shepherd_score.score.analytical_gradients import compute_self_overlaps_pharm
        ptype_1, ptype_2, anchors_1, anchors_2, vectors_1, vectors_2 = \
            TestDataGenerator.generate_pharmacophore_data(12, 10, seed=3)
        args = [torch.from_numpy(x) for x in (ptype_1, ptype_2, anchors_1, anchors_2, vectors_1, vectors_2)]
        cache = compute_self_overlaps_pharm(*args, directionless=True)

        recomputed = get_overlap_pharm_torch(*args, similarity=similarity, directionless=True)
        cached = get_overlap_pharm_torch(*args, similarity=similarity, directionless=True,
                                         precomputed_self_overlaps=cache)
        assert torch.allclose(cached, recomputed, rtol=RTOL, atol=ATOL)

        batch = [x.repeat((3,) + (1,) * x.dim()) for x in args]
        cached_batch = get_overlap_pharm_torch(*batch, similarity=similarity, directionless=True,
                                               precomputed_self_overlaps=cache)
        assert torch.allclose(cached_batch, recomputed.expand(3), rtol=RTOL, atol=ATOL)

        if isinstance(similarity, str):
            res_np = get_overlap_pharm_np(ptype_1, ptype_2, anchors_1, anchors_2, vectors_1, vectors_2,
                                          similarity=similarity, directionless=True)
            assert np.allclose(cached.numpy(), res_np, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("color_weight", [0.0, 0.3, 0.5, 1.0])
    def test_combo_color_weight_matches_np(self, color_weight):
        """Combined shape+color score: torch matches NumPy oracle for directionless mode."""
        ptype_1, ptype_2, anchors_1, anchors_2, vectors_1, vectors_2 = \
            TestDataGenerator.generate_pharmacophore_data(10, 8, seed=42)
        centers_1, centers_2 = TestDataGenerator.generate_shape_data(15, 13)

        res_np = get_pharm_combo_score_np(
            centers_1, centers_2, ptype_1, ptype_2, anchors_1, anchors_2,
            vectors_1, vectors_2, color_weight=color_weight, directionless=True,
        )
        res_torch = get_pharm_combo_score_torch(
            torch.from_numpy(centers_1), torch.from_numpy(centers_2),
            torch.from_numpy(ptype_1), torch.from_numpy(ptype_2),
            torch.from_numpy(anchors_1), torch.from_numpy(anchors_2),
            torch.from_numpy(vectors_1), torch.from_numpy(vectors_2),
            color_weight=color_weight, directionless=True,
        )
        assert np.allclose(res_torch.numpy(), res_np, rtol=RTOL, atol=ATOL)

    def test_combo_default_is_unweighted_average(self):
        """color_weight=0.5 reproduces the previous (pharm+shape)/2 default."""
        ptype_1, ptype_2, anchors_1, anchors_2, vectors_1, vectors_2 = \
            TestDataGenerator.generate_pharmacophore_data(8, 6, seed=11)
        centers_1, centers_2 = TestDataGenerator.generate_shape_data(13, 11)
        # default call (color_weight=0.5, directionless=False) == legacy average
        res = get_pharm_combo_score_np(centers_1, centers_2, ptype_1, ptype_2,
                                       anchors_1, anchors_2, vectors_1, vectors_2)
        pharm = get_overlap_pharm_np(ptype_1, ptype_2, anchors_1, anchors_2,
                                     vectors_1, vectors_2)
        # shape via the same path the combo uses
        from shepherd_score.score.gaussian_overlap_np import VAB_2nd_order_np
        vab = VAB_2nd_order_np(centers_1, centers_2, 0.81)
        vaa = VAB_2nd_order_np(centers_1, centers_1, 0.81)
        vbb = VAB_2nd_order_np(centers_2, centers_2, 0.81)
        shape = vab / (vaa + vbb - vab)
        assert np.allclose(res, (pharm + shape) / 2, rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------- #
# RDKit BaseFeatures.fdef featurization + vol_color alignment (need a real mol)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def embedded_mol():
    """Embed one drug-like molecule once (ibuprofen: aromatic/hydrophobe/donor/acceptor/anion)."""
    try:
        from shepherd_score.conformer_generation import embed_conformer_from_smiles
        mol = embed_conformer_from_smiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O",
                                          MMFF_optimize=True, random_seed=0)
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"could not embed test molecule: {e}")
    if mol is None:
        pytest.skip("embedding returned None")
    return mol


def _rotated_translated_copy(mol, seed=0):
    """Return a rigid SE(3) copy of `mol` (proper rotation + translation)."""
    from rdkit.Chem import Mol
    from rdkit.Geometry import Point3D
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))          # make deterministic
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]               # ensure a proper rotation (det +1)
    t = rng.standard_normal(3) * 3.0
    m = Mol(mol)
    conf = m.GetConformer()
    pos = conf.GetPositions() @ Q.T + t
    for i in range(m.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(*[float(v) for v in pos[i]]))
    return m


class TestRDKitBaseFeaturization:

    def test_rdkit_base_directionless_vectors_zero(self, embedded_mol):
        from shepherd_score.pharm_utils.pharmacophore import get_pharmacophores
        X, P, V = get_pharmacophores(embedded_mol, multi_vector=False,
                                     feature_set='rdkit_base', directionless=True)
        assert len(X) > 0, "no features extracted"
        # all vectors zero (isotropic color atoms)
        assert np.allclose(V, 0.0), "directionless featurization left nonzero vectors"
        # only the 6 ROCS/ROSHAMBO color types: indices for
        # Acceptor=0, Donor=1, Aromatic=2, Hydrophobe=3, Cation=5, Anion=6
        assert set(np.unique(X)).issubset({0, 1, 2, 3, 5, 6}), \
            f"unexpected feature types {set(np.unique(X))}"
        # ibuprofen should at least have an aromatic ring feature
        assert 2 in set(np.unique(X)), "expected an aromatic feature"

    def test_shepherd_default_unchanged(self, embedded_mol):
        """Default feature_set='shepherd' still computes orientation vectors (regression)."""
        from shepherd_score.pharm_utils.pharmacophore import get_pharmacophores
        X, P, V = get_pharmacophores(embedded_mol, multi_vector=False)
        assert len(X) > 0
        # at least some nonzero vectors remain (donor/acceptor/aromatic have orientation)
        assert not np.allclose(V, 0.0)

    def test_molecule_directionless_kwarg(self, embedded_mol):
        """Molecule(feature_set='rdkit_base', directionless=True) stores zero vectors."""
        from shepherd_score.container import Molecule
        mol = Molecule(embedded_mol, pharm_multi_vector=False,
                       feature_set='rdkit_base', directionless=True)
        assert mol.pharm_types is not None and len(mol.pharm_types) > 0
        assert np.allclose(mol.pharm_vecs, 0.0)
        assert set(np.unique(mol.pharm_types)).issubset({0, 1, 2, 3, 5, 6})


class TestVolColorAlignment:

    def test_self_copy_recovers_one(self, embedded_mol):
        """A rigid SE(3) self-copy must recover combined vol+color similarity ~ 1.0."""
        from shepherd_score.container import Molecule, MoleculePair

        ref = Molecule(embedded_mol, pharm_multi_vector=False, feature_set='rdkit_base')
        fit_rd = _rotated_translated_copy(embedded_mol, seed=7)
        fit = Molecule(fit_rd, pharm_multi_vector=False, feature_set='rdkit_base')

        mp = MoleculePair(ref, fit, do_center=True, device=torch.device('cpu'))
        mp.align_with_vol_color(num_repeats=20, max_num_steps=200)
        score = float(mp.sim_aligned_vol_color)
        assert score > 0.95, f"vol_color self-copy recovered only {score:.3f}"

    def test_requires_pharmacophores(self, embedded_mol):
        """align_with_vol_color must raise if the molecules carry no pharmacophores."""
        from shepherd_score.container import Molecule, MoleculePair
        ref = Molecule(embedded_mol)            # no pharm_multi_vector -> no pharm
        fit = Molecule(embedded_mol)
        mp = MoleculePair(ref, fit, device=torch.device('cpu'))
        with pytest.raises(ValueError):
            mp.align_with_vol_color(num_repeats=2, max_num_steps=2)


class TestVolColorBatch:
    """The batched vol_color driver (shape kernel + directionless color)."""

    def test_batched_numba_self_copy(self, embedded_mol):
        """The batched numba vol_color driver recovers self-copies at ~1.0."""
        try:
            import numba  # noqa: F401
        except ImportError:
            pytest.skip("numba required for the CPU batched path")
        from shepherd_score.container import Molecule, MoleculePair, MoleculePairBatch

        pairs = []
        for k in range(3):
            ref = Molecule(embedded_mol, pharm_multi_vector=False, feature_set='rdkit_base')
            fit_rd = _rotated_translated_copy(embedded_mol, seed=k + 1)
            fit = Molecule(fit_rd, pharm_multi_vector=False, feature_set='rdkit_base')
            pairs.append(MoleculePair(ref, fit, do_center=True, device=torch.device('cpu')))

        b = MoleculePairBatch(pairs)
        scores, _ = b.align_with_vol_color(backend="numba", num_repeats=20, max_num_steps=150)
        assert np.all(np.asarray(scores) > 0.95), f"batched vol_color self-copy low: {scores}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
