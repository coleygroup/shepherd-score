"""
Tests for the opt-in mesh-free smooth+stochastic surfacer (`method='smooth_sdf'`) and the
surface diagnostics. These run WITHOUT Open3D (the default 'mesh' path needs it; those checks
are skipped when Open3D is absent).
"""
import importlib.util
import inspect

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from shepherd_score.container import Molecule
from shepherd_score import generate_point_cloud as gpc
from shepherd_score.generate_point_cloud import (
    get_molecular_surface,
    get_molecular_surface_smooth_sdf,
    _get_masked_surface_candidates,
    _smoothmin_sdf_project,
    _farthest_point_sample,
    get_atomic_vdw_radii,
)
from .._support import surface_diagnostics as sd

HAS_OPEN3D = importlib.util.find_spec("open3d") is not None
PROBE = 1.2


def _embed(smiles, seed=0xC0FFEE):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed, maxAttempts=400)
    AllChem.MMFFOptimizeMolecule(m)
    return m


def _geom(m):
    return (m.GetConformer().GetPositions().astype(np.float64),
            get_atomic_vdw_radii(m).astype(np.float64))


# --------------------------------------------------------------------------- backwards compat
def test_default_is_mesh():
    """The default surface method must remain 'mesh' everywhere (unchanged behavior)."""
    assert inspect.signature(get_molecular_surface).parameters["method"].default == "mesh"
    assert inspect.signature(Molecule.__init__).parameters["surface_method"].default == "mesh"
    m = _embed("c1ccccc1")
    # num_surf_points=None builds no surface, so this works without Open3D and must record 'mesh'
    assert Molecule(m, num_surf_points=None).surface_method == "mesh"


def test_unknown_method_raises():
    c, r = _geom(_embed("CC"))
    with pytest.raises(ValueError):
        get_molecular_surface(c, r, num_points=100, method="bogus")


@pytest.mark.skipif(HAS_OPEN3D, reason="needs an Open3D-free env to prove mesh path is untouched")
def test_default_mesh_still_uses_open3d():
    """The default 'mesh' path still requires Open3D, so it was not rerouted to the mesh-free path."""
    c, r = _geom(_embed("c1ccccc1"))
    with pytest.raises(ModuleNotFoundError):
        get_molecular_surface(c, r, num_points=100)  # method defaults to 'mesh'


def test_fast_alias_matches_smooth_sdf():
    """'fast' is an opt-in alias for the mesh-free smooth surfacer."""
    c, r = _geom(_embed("c1ccccc1"))
    a = get_molecular_surface(c, r, num_points=150, method="fast", seed=7)
    b = get_molecular_surface(c, r, num_points=150, method="smooth_sdf", seed=7)
    assert a.shape == (150, 3) and np.allclose(a, b)


# --------------------------------------------------------------------------- smooth_sdf surfacer
def test_smooth_sdf_no_open3d_and_exact_count():
    import sys
    c, r = _geom(_embed("CC(=O)Oc1ccccc1C(=O)O"))  # aspirin
    # open3d may already be imported by earlier tests; the claim is that this call does not import it
    had_open3d = "open3d" in sys.modules
    surf = get_molecular_surface(c, r, num_points=200, method="smooth_sdf", seed=1)
    assert surf.shape == (200, 3)
    assert surf.dtype == np.float32
    assert np.all(np.isfinite(surf))
    assert ("open3d" in sys.modules) == had_open3d  # the smooth path never imports Open3D


def test_molecule_smooth_sdf_end_to_end():
    m = _embed("Cn1cnc2c1c(=O)n(C)c(=O)n2C")  # caffeine
    mol = Molecule(m, num_surf_points=200, surface_method="smooth_sdf")
    assert mol.surf_pos.shape == (200, 3)
    assert mol.surf_esp.shape == (200,)
    assert np.all(np.isfinite(mol.surf_esp))


def test_stochastic_unseeded_reproducible_seeded():
    c, r = _geom(_embed("c1ccccc1"))
    a = get_molecular_surface_smooth_sdf(c, r, 200, seed=None)
    b = get_molecular_surface_smooth_sdf(c, r, 200, seed=None)
    assert not np.allclose(a, b)  # stochastic
    assert np.allclose(get_molecular_surface_smooth_sdf(c, r, 200, seed=7),
                       get_molecular_surface_smooth_sdf(c, r, 200, seed=7))  # reproducible


def test_knn_cutoff_is_exact():
    c, r = _geom(_embed("CC(=O)Oc1ccccc1C(=O)O"))
    cand = _get_masked_surface_candidates(c, r, seed=5)
    a = r + PROBE
    full = _smoothmin_sdf_project(cand, c, a, s=10, iters=6, knn=len(c))
    knn = _smoothmin_sdf_project(cand, c, a, s=10, iters=6, knn=8)
    # the k=8 cutoff agrees with the full LSE far below the surface noise floor
    assert np.max(np.linalg.norm(full - knn, axis=1)) < 1e-3


def test_edge_cases():
    # tiny molecule
    c, r = _geom(_embed("O"))
    s = get_molecular_surface_smooth_sdf(c, r, 200, seed=1)
    assert len(s) > 0 and np.all(np.isfinite(s))
    # num_points=None returns the full envelope (more than a typical request)
    c, r = _geom(_embed("c1ccccc1"))
    assert len(get_molecular_surface_smooth_sdf(c, r, None, seed=1)) > 200
    # density + smooth_sdf is guarded
    with pytest.raises(ValueError):
        Molecule(_embed("CC"), density=0.5, surface_method="smooth_sdf")


# --------------------------------------------------------------------------- anti-leak / crimps
def test_smooth_defeats_leak_relative_to_on_sphere():
    """On-sphere surfaces recover atom centers exactly (~0); the smooth surface must not."""
    c, r = _geom(_embed("CC(=O)Oc1ccccc1C(=O)O"))
    on_sphere = _farthest_point_sample(
        _get_masked_surface_candidates(c, r, stochastic=False), 200)
    smooth = get_molecular_surface_smooth_sdf(c, r, 200, seed=3)
    rec_on = sd.center_recovery_attack(on_sphere, c, r)["median_center_error"]
    rec_sm = sd.center_recovery_attack(smooth, c, r)["median_center_error"]
    assert rec_on < 0.01          # on-sphere leaks (centers recoverable)
    assert rec_sm > 0.02          # smooth hides them better
    assert sd.leak_metrics(smooth, c, r)["median_residual"] > \
           sd.leak_metrics(on_sphere, c, r)["median_residual"]


def test_crimp_detection_localizes_at_seam():
    """Two equal spheres on the x-axis: detected crimp points must cluster at the x=mid seam plane."""
    d = 1.6
    centers = np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
    radii = np.array([1.7, 1.7])
    pts = _get_masked_surface_candidates(centers, radii, num_samples_per_atom=30,
                                         stochastic=False)
    is_crimp, owner = sd.crimp_points(pts, centers, radii)
    assert is_crimp.sum() > 0
    assert set(np.unique(owner)) == {0, 1}            # both atoms own surface
    mid = d / 2.0
    ax_crimp = np.abs(pts[is_crimp][:, 0] - mid)
    ax_other = np.abs(pts[~is_crimp][:, 0] - mid)
    # crimp points sit much closer to the intersection plane than non-crimp points
    assert np.median(ax_crimp) < np.median(ax_other)


def test_summarize_runs():
    c, r = _geom(_embed("c1ccccc1"))
    out = sd.summarize(get_molecular_surface_smooth_sdf(c, r, 200, seed=1), c, r)
    for key in ("median_residual", "median_center_error", "crimp_fraction",
                "curvature_at_crimps", "curvature_elsewhere"):
        assert key in out
