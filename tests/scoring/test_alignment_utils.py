"""
Unit tests for alignment and related fucntions.
"""
import pytest
import numpy as np
import torch
from .._support.utils import _configure_jax_platform
from shepherd_score.alignment.utils.se3_np import (
    quaternions_to_rotation_matrix_np,
    get_SE3_transform_np,
    apply_SE3_transform_np,
)
from shepherd_score.alignment.utils.se3 import (
    quaternions_to_rotation_matrix,
    get_SE3_transform,
    apply_SE3_transform,
)


# Attempt to import JAX and related modules
JAX_AVAILABLE = False
jnp = None
quaternions_to_rotation_matrix_jax = None
get_SE3_transform_jax = None
apply_SE3_transform_jax = None

try:
    # Configure JAX platform before import to avoid GPU initialization errors
    _gpu_detected = _configure_jax_platform()

    import jax.numpy as jnp
    from shepherd_score.alignment.utils.se3_jax import (
        quaternions_to_rotation_matrix_jax,
        get_SE3_transform_jax,
        apply_SE3_transform_jax
    )

    JAX_AVAILABLE = True
except ImportError:
    # JAX not available - JAX-specific tests will be skipped
    pass


class TestSE3:
    """
    Test SE(3) related transformations.
    Checks all versions: NumPy, Jax, Torch (batched and unbatched)
    """
    ex_quaternion = np.array([5., 0.2, -3.4, 0.4])
    # Numbers from from https://www.andre-gaschler.com/rotationconverter/
    sol_rot_matrix = np.array([
        [0.3623504, -0.1458107, -0.9205658],
        [0.0718172,  0.9891186, -0.1284004],
        [0.9292709, -0.0195865,  0.3688792]
    ])

    ex_se3_params = np.array([5., 0.2, -3.4, 0.4, 4., 5., 6.])
    sol_se3_transform = np.array([
        [0.3623504, -0.1458107, -0.9205658, 4.],
        [0.0718172,  0.9891186, -0.1284004, 5.],
        [0.9292709, -0.0195865,  0.3688792, 6.],
        [0.       , 0.        , 0.        , 1.]
    ])

    ex_set_of_points = np.array([
        [1., 2., 3.],
        [-1., 2., -3.],
        [0.4, -0.4, -0.5]
    ])
    sol_points_transformed = (sol_se3_transform[:3, :3] @ ex_set_of_points.T).T + sol_se3_transform[:3, 3]

    # NUMPY functions
    def test_quaternion_to_rotation_matrix_np(self):
        """ Test quaternions_to_rotation_matrix_np """
        out_rot_matrix = quaternions_to_rotation_matrix_np(self.ex_quaternion)
        assert np.allclose(out_rot_matrix, self.sol_rot_matrix)

    def test_se3_matrix_from_params_np(self):
        """ Test get_SE3_transform_np """
        out_se3_matrix = get_SE3_transform_np(self.ex_se3_params)
        assert np.allclose(out_se3_matrix, self.sol_se3_transform)

    def test_apply_se3_transform_np(self):
        """ Test apply_SE3_transform_np  """
        out_transformed = apply_SE3_transform_np(self.ex_set_of_points, self.sol_se3_transform)
        assert np.allclose(out_transformed, self.sol_points_transformed)

    # TORCH functions (single instance)
    def test_quaternion_to_rotation_matrix_torch(self):
        """ Test quaternions_to_rotation_matrix (Torch) """
        out_rot_matrix = quaternions_to_rotation_matrix(torch.Tensor(self.ex_quaternion))
        assert torch.allclose(out_rot_matrix, torch.Tensor(self.sol_rot_matrix))

    def test_se3_matrix_from_params_torch(self):
        """ Test get_SE3_transform (torch) """
        out_se3_matrix = get_SE3_transform(torch.Tensor(self.ex_se3_params))
        assert torch.allclose(out_se3_matrix, torch.Tensor(self.sol_se3_transform))

    def test_apply_se3_transform_torch(self):
        """ Test apply_SE3_transform torch  """
        out_transformed = apply_SE3_transform(torch.Tensor(self.ex_set_of_points), torch.Tensor(self.sol_se3_transform))
        assert torch.allclose(out_transformed, torch.Tensor(self.sol_points_transformed))

    # TORCH functions (batched)
    def test_quaternion_to_rotation_matrix_torch_batched(self):
        """ Test quaternions_to_rotation_matrix (Torch batched) """
        out_rot_matrix = quaternions_to_rotation_matrix(torch.Tensor(self.ex_quaternion).repeat((2, 1)))
        assert torch.allclose(out_rot_matrix, torch.Tensor(self.sol_rot_matrix).repeat((2,1,1)))

    def test_se3_matrix_from_params_torch_batched(self):
        """ Test get_SE3_transform (torch batched) """
        out_se3_matrix = get_SE3_transform(torch.Tensor(self.ex_se3_params).repeat((2,1)))
        assert torch.allclose(out_se3_matrix, torch.Tensor(self.sol_se3_transform).repeat((2, 1, 1)))

    def test_apply_se3_transform_torch_batched(self):
        """ Test apply_SE3_transform (torch batched)  """
        points_repeated = torch.Tensor(self.ex_set_of_points).repeat((2, 1, 1))
        transform_repeated = torch.Tensor(self.sol_se3_transform).repeat((2, 1, 1))
        out_transformed = apply_SE3_transform(points_repeated, transform_repeated)
        sol_repeated = torch.Tensor(self.sol_points_transformed).repeat((2, 1, 1))
        assert torch.allclose(out_transformed, sol_repeated)

    def test_apply_se3_transform_torch_R1_stays_batched(self):
        """A singleton batch (R==1) keeps its batch dimension, as upstream does."""
        pts_single = torch.Tensor(self.ex_set_of_points)
        tf_single = torch.Tensor(self.sol_se3_transform)
        out_r1 = apply_SE3_transform(pts_single.unsqueeze(0), tf_single.unsqueeze(0))
        assert out_r1.shape == (1, self.ex_set_of_points.shape[0], 3), out_r1.shape
        assert torch.allclose(out_r1[0], torch.Tensor(self.sol_points_transformed))

    # Jax functions (single instance)
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
    def test_quaternion_to_rotation_matrix_jax(self):
        """ Test quaternions_to_rotation_matrix (Jax) """
        out_rot_matrix = quaternions_to_rotation_matrix_jax(jnp.array(self.ex_quaternion))
        assert jnp.allclose(out_rot_matrix, jnp.array(self.sol_rot_matrix))

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
    def test_se3_matrix_from_params_jax(self):
        """ Test get_SE3_transform (Jax) """
        out_se3_matrix = get_SE3_transform_jax(jnp.array(self.ex_se3_params))
        assert jnp.allclose(out_se3_matrix, jnp.array(self.sol_se3_transform))

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
    def test_apply_se3_transform_jax(self):
        """ Test apply_SE3_transform (Jax)  """
        out_transformed = apply_SE3_transform_jax(jnp.array(self.ex_set_of_points), jnp.array(self.sol_se3_transform))
        assert jnp.allclose(out_transformed, jnp.array(self.sol_points_transformed))



class TestPCA:
    """
    Test principal component analysis
    """

    def test_rotation_axis_np_antiparallel_is_finite_perpendicular(self):
        """Antiparallel vectors give a finite unit axis perpendicular to v1, not a 0/0 NaN."""
        from shepherd_score.alignment.utils.pca_np import rotation_axis_np
        for v1 in ([1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0.6, -0.8, 0.0]):
            v1 = np.asarray(v1)
            ax = rotation_axis_np(v1, -v1)                       # antiparallel
            assert np.all(np.isfinite(ax)), f"non-finite axis for v1={v1}: {ax}"
            assert np.isclose(np.linalg.norm(ax), 1.0, atol=1e-6)
            assert np.isclose(np.dot(ax, v1), 0.0, atol=1e-6), "axis must be perpendicular to v1"

    def test_rotation_axis_np_generic_unchanged(self):
        """Non-parallel vectors give the plain normalized cross product."""
        from shepherd_score.alignment.utils.pca_np import rotation_axis_np
        v1, v2 = np.array([1., 0., 0.]), np.array([0., 1., 0.])
        np.testing.assert_allclose(rotation_axis_np(v1, v2), [0., 0., 1.], atol=1e-7)

    def test_degenerate_molecules_align_without_eigh_crash(self):
        """Linear, single-heavy-atom and symmetric-top molecules self-align to ~1.0 per pair."""
        pytest.importorskip("rdkit")
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from shepherd_score.container import Molecule, MoleculePair

        def build(smi):
            m = Chem.AddHs(Chem.MolFromSmiles(smi))
            p = AllChem.ETKDGv3(); p.randomSeed = 0
            assert AllChem.EmbedMolecule(m, p) == 0
            return Molecule(m)

        for smi in ("N#N", "O", "C1CCCCC1", "CC(C)(C)C", "C#C"):   # N2, water, cyclohexane, neopentane, acetylene
            mol = build(smi)
            mp = MoleculePair(mol, mol, do_center=True)
            torch.manual_seed(0)
            mp.align_with_vol()                                    # must not raise
            assert float(np.asarray(mp.sim_aligned_vol_noH)) > 0.99, f"self-overlap too low for {smi}"

    def test_mmff_unparameterizable_raises_clear_error(self):
        """A molecule MMFF cannot parameterize (H2) raises a clear ValueError on the charge property."""
        pytest.importorskip("rdkit")
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from shepherd_score.container import Molecule

        m = Chem.AddHs(Chem.MolFromSmiles("[H][H]"))
        AllChem.EmbedMolecule(m, randomSeed=0)
        # charges are lazy and the default model is xtb, so force the MMFF path and touch the
        # property to exercise the guard
        mol = Molecule(m, charge_model="mmff")
        with pytest.raises(ValueError, match="MMFF94 could not parameterize"):
            _ = mol.partial_charges
