"""
Cross-platform test suite for pharmacophore scoring implementations.
Tests PyTorch and JAX implementations against NumPy ground truth.
"""
import pytest
import numpy as np
import warnings
import os
from typing import Tuple, List

# Configure JAX platform before import to avoid GPU errors
def _configure_jax_platform():
    """Configure JAX platform based on GPU availability."""
    try:
        # Try to detect GPU through other means first
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        gpu_detected = result.returncode == 0
    except (FileNotFoundError, subprocess.SubprocessError):
        gpu_detected = False
    
    if not gpu_detected:
        # Force JAX to use CPU only if no GPU detected
        os.environ['JAX_PLATFORMS'] = 'cpu'
    
    return gpu_detected

# Pre-configure JAX platform
_gpu_detected_early = _configure_jax_platform()

# Try to import all implementations
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available - skipping PyTorch tests")

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    JAX_AVAILABLE = True
    
    # Check for GPU more safely now
    try:
        gpu_devices = jax.devices('gpu')
        GPU_AVAILABLE = len(gpu_devices) > 0 and _gpu_detected_early
    except:
        GPU_AVAILABLE = False
        
except ImportError:
    JAX_AVAILABLE = False
    GPU_AVAILABLE = False
    warnings.warn("JAX not available - skipping JAX tests")

# Import implementations conditionally
if TORCH_AVAILABLE:
    from shepherd_score.score.pharmacophore_scoring import (
        get_overlap_pharm as get_overlap_pharm_torch,
        get_pharm_combo_score as get_pharm_combo_score_torch
    )

if JAX_AVAILABLE:
    from shepherd_score.score.pharmacophore_scoring_jax import (
        get_overlap_pharm_jax,
        get_pharm_combo_score_jax
    )

# Always import NumPy implementation (ground truth)
from shepherd_score.score.pharmacophore_scoring_np import (
    get_overlap_pharm_np,
    get_pharm_combo_score as get_pharm_combo_score_np
)

# Test tolerances
RTOL = 1e-5
ATOL = 1e-7

class TestDataGenerator:
    """Generate consistent test data across all implementations."""
    
    @staticmethod
    def generate_pharmacophore_data(n_pharm_1: int = 10, n_pharm_2: int = 8, 
                                  seed: int = 42) -> Tuple[np.ndarray, ...]:
        """Generate test pharmacophore data."""
        np.random.seed(seed)
        
        # Pharmacophore types (indices 0-7 for the 8 pharmacophore types)
        ptype_1 = np.random.randint(0, 8, size=n_pharm_1)
        ptype_2 = np.random.randint(0, 8, size=n_pharm_2)
        
        # Anchor coordinates
        anchors_1 = np.random.normal(0, 5, size=(n_pharm_1, 3)).astype(np.float32)
        anchors_2 = np.random.normal(0, 5, size=(n_pharm_2, 3)).astype(np.float32)
        
        # Vector directions (normalized)
        vectors_1 = np.random.normal(0, 1, size=(n_pharm_1, 3)).astype(np.float32)
        vectors_1 = vectors_1 / np.linalg.norm(vectors_1, axis=1, keepdims=True)
        
        vectors_2 = np.random.normal(0, 1, size=(n_pharm_2, 3)).astype(np.float32)
        vectors_2 = vectors_2 / np.linalg.norm(vectors_2, axis=1, keepdims=True)
        
        return ptype_1, ptype_2, anchors_1, anchors_2, vectors_1, vectors_2
    
    @staticmethod
    def generate_shape_data(n_atoms_1: int = 20, n_atoms_2: int = 15, 
                          seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Generate test shape data (atomic coordinates)."""
        np.random.seed(seed + 100)  # Different seed for shape data
        
        centers_1 = np.random.normal(0, 3, size=(n_atoms_1, 3)).astype(np.float32)
        centers_2 = np.random.normal(0, 3, size=(n_atoms_2, 3)).astype(np.float32)
        
        return centers_1, centers_2

# Fixtures for test data
@pytest.fixture(params=[
    (5, 3),    # Small
    (10, 8),   # Medium
    (20, 15),  # Large
    (1, 1),    # Single pharmacophore each
])
def pharmacophore_sizes(request):
    """Different pharmacophore array sizes for testing."""
    return request.param

@pytest.fixture(params=[
    'tanimoto',
    'tversky', 
    'tversky_ref',
    'tversky_fit'
])
def similarity_func(request):
    """Different similarity functions for testing."""
    return request.param

@pytest.fixture(params=[
    (False, False),  # No extended points
    (True, False),   # Extended points, include anchors
    (True, True),    # Extended points only
])
def extended_points_config(request):
    """Different extended points configurations."""
    extended_points, only_extended = request.param
    return extended_points, only_extended

@pytest.fixture
def test_data(pharmacophore_sizes):
    """Generate test data for given sizes."""
    n1, n2 = pharmacophore_sizes
    data = TestDataGenerator.generate_pharmacophore_data(n1, n2)
    shape_data = TestDataGenerator.generate_shape_data(n1 + 5, n2 + 5)
    return data + shape_data

class TestPharmacophoreScoring:
    """Test pharmacophore scoring implementations."""

    def test_single_pharmacophore_type(self):
        """Test with all pharmacophores of the same type."""
        # All hydrophobic pharmacophores
        ptype_1 = np.array([0, 0, 0], dtype=np.int32)
        ptype_2 = np.array([0, 0], dtype=np.int32)
        
        anchors_1 = np.random.normal(0, 2, size=(3, 3)).astype(np.float32)
        anchors_2 = np.random.normal(0, 2, size=(2, 3)).astype(np.float32)
        vectors_1 = np.random.normal(0, 1, size=(3, 3)).astype(np.float32)
        vectors_2 = np.random.normal(0, 1, size=(2, 3)).astype(np.float32)
        
        # Normalize vectors
        vectors_1 = vectors_1 / np.linalg.norm(vectors_1, axis=1, keepdims=True)
        vectors_2 = vectors_2 / np.linalg.norm(vectors_2, axis=1, keepdims=True)
        
        # NumPy result
        result_np = get_overlap_pharm_np(
            ptype_1, ptype_2, anchors_1, anchors_2, vectors_1, vectors_2
        )
        
        # Compare with other implementations
        if TORCH_AVAILABLE:
            result_torch = get_overlap_pharm_torch(
                torch.from_numpy(ptype_1), torch.from_numpy(ptype_2),
                torch.from_numpy(anchors_1), torch.from_numpy(anchors_2),
                torch.from_numpy(vectors_1), torch.from_numpy(vectors_2)
            )
            assert np.allclose(result_torch.numpy(), result_np, rtol=RTOL, atol=ATOL)
        
        if JAX_AVAILABLE:
            result_jax = get_overlap_pharm_jax(
                jnp.array(ptype_1), jnp.array(ptype_2),
                jnp.array(anchors_1), jnp.array(anchors_2),
                jnp.array(vectors_1), jnp.array(vectors_2)
            )
            assert np.allclose(np.array(result_jax), result_np, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("similarity", ['tanimoto', 'tversky', 'tversky_ref', 'tversky_fit'])
    def test_pharmacophore_scoring_similarity_functions(self, test_data, similarity):
        """Test different similarity functions."""
        ptype_1, ptype_2, anchors_1, anchors_2, vectors_1, vectors_2, centers_1, centers_2 = test_data
        
        # NumPy result (ground truth)
        result_np = get_overlap_pharm_np(
            ptype_1, ptype_2, anchors_1, anchors_2, vectors_1, vectors_2,
            similarity=similarity
        )
        
        # PyTorch comparison
        if TORCH_AVAILABLE:
            result_torch = get_overlap_pharm_torch(
                torch.from_numpy(ptype_1), torch.from_numpy(ptype_2),
                torch.from_numpy(anchors_1), torch.from_numpy(anchors_2),
                torch.from_numpy(vectors_1), torch.from_numpy(vectors_2),
                similarity=similarity
            )
            assert np.allclose(result_torch.numpy(), result_np, rtol=RTOL, atol=ATOL), \
                f"PyTorch result {result_torch.numpy()} != NumPy result {result_np} for {similarity}"
        
        # JAX comparison
        if JAX_AVAILABLE:
            result_jax = get_overlap_pharm_jax(
                jnp.array(ptype_1), jnp.array(ptype_2),
                jnp.array(anchors_1), jnp.array(anchors_2),
                jnp.array(vectors_1), jnp.array(vectors_2),
                similarity=similarity
            )
            assert np.allclose(np.array(result_jax), result_np, rtol=RTOL, atol=ATOL), \
                f"JAX result {np.array(result_jax)} != NumPy result {result_np} for {similarity}"

    def test_extended_points_configurations(self, test_data, extended_points_config):
        """Test extended points configurations."""
        extended_points, only_extended = extended_points_config
        ptype_1, ptype_2, anchors_1, anchors_2, vectors_1, vectors_2, centers_1, centers_2 = test_data
        
        # NumPy result (ground truth)
        result_np = get_overlap_pharm_np(
            ptype_1, ptype_2, anchors_1, anchors_2, vectors_1, vectors_2,
            extended_points=extended_points, only_extended=only_extended
        )
        
        # PyTorch comparison
        if TORCH_AVAILABLE:
            result_torch = get_overlap_pharm_torch(
                torch.from_numpy(ptype_1), torch.from_numpy(ptype_2),
                torch.from_numpy(anchors_1), torch.from_numpy(anchors_2),
                torch.from_numpy(vectors_1), torch.from_numpy(vectors_2),
                extended_points=extended_points, only_extended=only_extended
            )
            assert np.allclose(result_torch.numpy(), result_np, rtol=RTOL, atol=ATOL), \
                f"PyTorch result != NumPy result for extended_points={extended_points}, only_extended={only_extended}"
        
        # JAX comparison
        if JAX_AVAILABLE:
            result_jax = get_overlap_pharm_jax(
                jnp.array(ptype_1), jnp.array(ptype_2),
                jnp.array(anchors_1), jnp.array(anchors_2),
                jnp.array(vectors_1), jnp.array(vectors_2),
                extended_points=extended_points, only_extended=only_extended
            )
            assert np.allclose(np.array(result_jax), result_np, rtol=RTOL, atol=ATOL), \
                f"JAX result != NumPy result for extended_points={extended_points}, only_extended={only_extended}"

class TestComboScoring:
    """Test combined pharmacophore + shape scoring."""
    
    def test_combo_scoring_basic(self, test_data):
        """Test basic combo scoring functionality."""
        ptype_1, ptype_2, anchors_1, anchors_2, vectors_1, vectors_2, centers_1, centers_2 = test_data
        
        # NumPy result (ground truth)
        result_np = get_pharm_combo_score_np(
            centers_1, centers_2, ptype_1, ptype_2, 
            anchors_1, anchors_2, vectors_1, vectors_2
        )
        
        # PyTorch comparison
        if TORCH_AVAILABLE:
            result_torch = get_pharm_combo_score_torch(
                torch.from_numpy(centers_1), torch.from_numpy(centers_2),
                torch.from_numpy(ptype_1), torch.from_numpy(ptype_2),
                torch.from_numpy(anchors_1), torch.from_numpy(anchors_2),
                torch.from_numpy(vectors_1), torch.from_numpy(vectors_2)
            )
            assert np.allclose(result_torch.numpy(), result_np, rtol=RTOL, atol=ATOL)
        
        # JAX comparison
        if JAX_AVAILABLE:
            result_jax = get_pharm_combo_score_jax(
                jnp.array(centers_1), jnp.array(centers_2),
                jnp.array(ptype_1), jnp.array(ptype_2),
                jnp.array(anchors_1), jnp.array(anchors_2),
                jnp.array(vectors_1), jnp.array(vectors_2)
            )
            assert np.allclose(np.array(result_jax), result_np, rtol=RTOL, atol=ATOL)

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_mismatched_pharmacophore_types(self):
        """Test with completely different pharmacophore types."""
        # Molecule 1 has only hydrophobic (0)
        ptype_1 = np.array([0, 0, 0], dtype=np.int32)
        # Molecule 2 has only aromatic (4)  
        ptype_2 = np.array([4, 4], dtype=np.int32)
        
        anchors_1 = np.random.normal(0, 1, size=(3, 3)).astype(np.float32)
        anchors_2 = np.random.normal(0, 1, size=(2, 3)).astype(np.float32)
        vectors_1 = np.random.normal(0, 1, size=(3, 3)).astype(np.float32)
        vectors_2 = np.random.normal(0, 1, size=(2, 3)).astype(np.float32)
        
        # Normalize vectors
        vectors_1 = vectors_1 / np.linalg.norm(vectors_1, axis=1, keepdims=True)
        vectors_2 = vectors_2 / np.linalg.norm(vectors_2, axis=1, keepdims=True)
        
        # Should give low/zero similarity
        result_np = get_overlap_pharm_np(
            ptype_1, ptype_2, anchors_1, anchors_2, vectors_1, vectors_2
        )
        
        # Result should be very small (likely 0) since no overlapping pharmacophore types
        assert result_np >= 0.0  # Similarity should be non-negative
        
        # Test consistency across implementations
        if TORCH_AVAILABLE:
            result_torch = get_overlap_pharm_torch(
                torch.from_numpy(ptype_1), torch.from_numpy(ptype_2),
                torch.from_numpy(anchors_1), torch.from_numpy(anchors_2),
                torch.from_numpy(vectors_1), torch.from_numpy(vectors_2)
            )
            assert np.allclose(result_torch.numpy(), result_np, rtol=RTOL, atol=ATOL)
        
        if JAX_AVAILABLE:
            result_jax = get_overlap_pharm_jax(
                jnp.array(ptype_1), jnp.array(ptype_2),
                jnp.array(anchors_1), jnp.array(anchors_2),
                jnp.array(vectors_1), jnp.array(vectors_2)
            )
            assert np.allclose(np.array(result_jax), result_np, rtol=RTOL, atol=ATOL)

    def test_identical_molecules(self):
        """Test scoring identical molecules (should give similarity = 1.0)."""
        ptype_1 = np.array([0, 1, 2, 4], dtype=np.int32)  # Mix of types
        anchors_1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        vectors_1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float32)
        vectors_1 = vectors_1 / np.linalg.norm(vectors_1, axis=1, keepdims=True)
        
        # Use same data for both molecules
        ptype_2, anchors_2, vectors_2 = ptype_1, anchors_1, vectors_1
        
        result_np = get_overlap_pharm_np(
            ptype_1, ptype_2, anchors_1, anchors_2, vectors_1, vectors_2,
            similarity='tanimoto'
        )
        
        # Should be exactly 1.0 for identical molecules with Tanimoto similarity
        assert np.allclose(result_np, 1.0, rtol=1e-10, atol=1e-10)
        
        # Test other implementations
        if TORCH_AVAILABLE:
            result_torch = get_overlap_pharm_torch(
                torch.from_numpy(ptype_1), torch.from_numpy(ptype_2),
                torch.from_numpy(anchors_1), torch.from_numpy(anchors_2),
                torch.from_numpy(vectors_1), torch.from_numpy(vectors_2),
                similarity='tanimoto'
            )
            assert np.allclose(result_torch.numpy(), 1.0, rtol=1e-10, atol=1e-10)
        
        if JAX_AVAILABLE:
            result_jax = get_overlap_pharm_jax(
                jnp.array(ptype_1), jnp.array(ptype_2),
                jnp.array(anchors_1), jnp.array(anchors_2),
                jnp.array(vectors_1), jnp.array(vectors_2),
                similarity='tanimoto'
            )
            assert np.allclose(np.array(result_jax), 1.0, rtol=1e-10, atol=1e-10)

# Performance benchmarking (optional, for development)
class TestPerformance:
    """Performance comparison tests (marked as slow)."""
    
    @pytest.mark.slow
    @pytest.mark.skipif(not (TORCH_AVAILABLE and JAX_AVAILABLE), 
                       reason="Both PyTorch and JAX needed for performance comparison")
    def test_performance_comparison(self):
        """Compare performance of different implementations."""
        import time
        
        # Generate larger test data
        ptype_1, ptype_2, anchors_1, anchors_2, vectors_1, vectors_2 = \
            TestDataGenerator.generate_pharmacophore_data(50, 40, seed=42)
        
        n_runs = 10
        
        # NumPy timing
        np_times = []
        for _ in range(n_runs):
            start = time.time()
            result_np = get_overlap_pharm_np(
                ptype_1, ptype_2, anchors_1, anchors_2, vectors_1, vectors_2
            )
            np_times.append(time.time() - start)
        np_time = np.mean(np_times)
        
        # PyTorch timing
        torch_ptype_1 = torch.from_numpy(ptype_1)
        torch_ptype_2 = torch.from_numpy(ptype_2)
        torch_anchors_1 = torch.from_numpy(anchors_1)
        torch_anchors_2 = torch.from_numpy(anchors_2)
        torch_vectors_1 = torch.from_numpy(vectors_1)
        torch_vectors_2 = torch.from_numpy(vectors_2)
        
        torch_times = []
        for _ in range(n_runs):
            start = time.time()
            result_torch = get_overlap_pharm_torch(
                torch_ptype_1, torch_ptype_2, torch_anchors_1, torch_anchors_2,
                torch_vectors_1, torch_vectors_2
            )
            torch_times.append(time.time() - start)
        torch_time = np.mean(torch_times)
        
        # JAX timing (with warmup)
        jax_ptype_1 = jnp.array(ptype_1)
        jax_ptype_2 = jnp.array(ptype_2)
        jax_anchors_1 = jnp.array(anchors_1)
        jax_anchors_2 = jnp.array(anchors_2)
        jax_vectors_1 = jnp.array(vectors_1)
        jax_vectors_2 = jnp.array(vectors_2)
        
        # Warmup for JIT compilation
        get_overlap_pharm_jax(
            jax_ptype_1, jax_ptype_2, jax_anchors_1, jax_anchors_2,
            jax_vectors_1, jax_vectors_2
        ).block_until_ready()
        
        jax_times = []
        for _ in range(n_runs):
            start = time.time()
            result_jax = get_overlap_pharm_jax(
                jax_ptype_1, jax_ptype_2, jax_anchors_1, jax_anchors_2,
                jax_vectors_1, jax_vectors_2
            ).block_until_ready()
            jax_times.append(time.time() - start)
        jax_time = np.mean(jax_times)
        
        print(f"\nPerformance comparison ({n_runs} runs):")
        print(f"NumPy:   {np_time:.4f}s")
        print(f"PyTorch: {torch_time:.4f}s ({torch_time/np_time:.2f}x vs NumPy)")
        print(f"JAX:     {jax_time:.4f}s ({jax_time/np_time:.2f}x vs NumPy)")
        
        # Verify results are still consistent
        assert np.allclose(result_torch.numpy(), result_np, rtol=RTOL, atol=ATOL)
        assert np.allclose(np.array(result_jax), result_np, rtol=RTOL, atol=ATOL)

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 