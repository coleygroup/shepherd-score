"""
Module contains functions to generate point clouds of
molecular surfaces.

Surface generation workflow adapted from Open Drug Discovery Toolkit:
    https://oddt.readthedocs.io/en/latest/_modules/oddt/surface.html
"""
from __future__ import annotations   # annotations as strings -> o3d not needed at import

from typing import Tuple, List, Union

import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial import distance
from scipy.special import logsumexp

from shepherd_score.score.constants import COULOMB_SCALING


# Open3D is imported lazily, on first use: its import is slow and it is fork-hostile
# (importing it before a fork breaks CUDA in the child, which the fork-based multi-GPU
# pool in shepherd_score.accel.multi_gpu relies on). Only surface generation needs it.
class _LazyOpen3D:
    def __getattr__(self, attr):
        import open3d as _o3d
        globals()["o3d"] = _o3d          # swap the proxy out for the real module
        return getattr(_o3d, attr)


o3d = _LazyOpen3D()

PT = Chem.GetPeriodicTable()

# Defaults for the opt-in mesh-free surfacer (get_molecular_surface_smooth_sdf); only used
# when method='smooth_sdf' is requested.
SMOOTH_SDF_S = 10.0       # smooth-min sharpness: smaller rounds the concave atom-border seams
                          # more, larger approaches the sharp sphere union
SMOOTH_SDF_NSPA = 15      # candidate samples per atom (the mesh path needs 25 to mesh)
SMOOTH_SDF_ITERS = 6      # Newton projection steps
SMOOTH_SDF_KNN = 8        # nearest atoms per point in the smooth-min (cost ~ O(M*knn))
SMOOTH_SDF_JITTER = 0.0   # extra off-sphere jitter (Angstrom) to mimic mesh facet noise
SMOOTH_SDF_EVEN = 'fps'   # 'fps' (even, blue-noise-like) or 'random' resample-to-count


def get_atom_coords(mol: rdkit.Chem.Mol,
                    MMFF_optimize: bool = True
                    ) -> Tuple[Chem.Mol, np.ndarray]:
    """
    Get the coordinates of all atoms in a molecule using rdkit.
    If the rdkit.Chem.mol object already has a conformer it just retrieves the coordinates
    without optimizing using MMFF.

    Parameters
    ----------
    mol : rdkit.Chem.Mol object
        RDKit molecule object

    MMFF_optimize : bool
        Whether or not to use MMFF to optimize geometry

    Returns
    -------
    tuple
        rdkit.Chem.Mol
            Mol object. If the input mol did not have a conformer, it uses MMFF to optimize an
            embeded molecule and includes hydrogens.
        np.ndarray: shape = (N,3)
            Positions of each atom's center.
    """
    try:
        mol.GetConformer()
    except ValueError:
        try:
            mol = Chem.AddHs(mol)
            Chem.AllChem.EmbedMolecule(mol, maxAttempts = 200)
            if MMFF_optimize:
                Chem.AllChem.MMFFOptimizeMolecule(mol)
                Chem.SanitizeMol(mol)

            mol.GetConformer() # test whether conformer generation succeeded
        except Exception as e:
            print('Failed to embed molecule:', e)
            return None
    return mol, mol.GetConformer().GetPositions()


def get_atomic_vdw_radii(mol: rdkit.Chem.Mol) -> np.ndarray:
    """
    Get the van der Waals radii of all atoms in a molecule using rdkit.

    Parameters
    ----------
    mol : rdkit.Chem.Mol object

    Returns
    -------
    np.ndarray : shape = (N,)
        vdW radii for each atom.
    """
    radii = np.zeros((mol.GetNumAtoms(),))
    for i, _ in enumerate(radii):
        # get the van der Waals radii of each atom
        radii[i] = PT.GetRvdw(mol.GetAtomWithIdx(i).GetAtomicNum())
    return radii


###################################
# Sampling from molecular surface #
###################################
def sample_molecular_surface_with_radius(centers: np.ndarray,
                                         radii: Union[np.ndarray, List],
                                         probe_radius: float = 1.2,
                                         num_samples_per_atom: int = 20
                                         ) -> np.ndarray:
    """
    Samples points from the surface of vdW radius of atoms and combines it into one molecule (Vectorized).

    Parameters
    ----------
    centers : np.ndarray (N, 3)
        Cartesian coordinates of the atom centers of a molecule.

    radii : np.ndarray (N,)
        van der Waals radii of each atom (in Angstrom) in the same order as the centers parameter.

    probe_radius : float (default = 1.2)
        The radius of a probe atom to act as a "solvent accessible surface".
        Default = 1.2 angstroms which is the radius of a Hydrogen atom.

    num_samples_per_atom : int, optional
        Number of points to sample from the surface of each atom. Default is 20.
        Note that this value is scaled by a given atom's relative vdW radius to a carbon
        and SQUARED. Typically choose a value between 15 and 35. For example, if set to
        20, a carbon atom would have 400 sampled points.

    Returns
    -------
    np.ndarray
        Array of shape (N*num_points_per_atom, 3) containing the coordinates of each
        point sampled from each atom.
    """
    # get surface radius based on vdW radii, cutoff
    # cutoff = 1.4
    if num_samples_per_atom > 50:
        raise ValueError('Do not set num_samples_per_atom to be larger than 50 for performance\
                         issues. The number is squared internally.')
    if not isinstance(radii, np.ndarray):
        radii = np.array(radii)
    # surf_radii = cutoff * (radii / 1.7) + probe_radius # scaled vdW radius + probe radius
    surf_radii = radii + probe_radius # vdW radius + probe radius
    # get number of samples per atom dependent on vdw radii
    num_samples_per_atom = np.ceil((num_samples_per_atom * (radii / 1.7))**2)

    # Sample unit surfaces from normal distribution
    x = np.random.normal(size=(int(num_samples_per_atom.sum()), 3))
    # reformat radii and center arrays to match number of samples per atom
    Rs = np.repeat(surf_radii.reshape((-1,1)), [int(n) for n in num_samples_per_atom], axis=0)
    centers = np.repeat(centers, [int(n) for n in num_samples_per_atom], axis=0)

    # Scale surfaces by radii and translate to centers
    surface = ((x / np.linalg.norm(x, axis=1)[:, np.newaxis]) * Rs) + centers
    return surface


def sample_molecular_surface_with_radius_fibonacci(centers: np.ndarray,
                                                   radii: Union[np.ndarray, List],
                                                   probe_radius: float = 1.2,
                                                   num_samples_per_atom: int = 20
                                                   ) -> np.ndarray:
    """
    Samples points from the surface of vdW radius of atoms and combines it into one molecule (Vectorized).

    Parameters
    ----------
    centers : np.ndarray (N, 3)
        Cartesian coordinates of the atom centers of a molecule.

    radii : np.ndarray (N,)
        van der Waals radii of each atom (in Angstrom) in the same order as the centers parameter.

    probe_radius : float (default = 1.2)
        The radius of a probe atom to act as a "solvent accessible surface".
        Default = 1.2 angstroms which is the radius of a Hydrogen atom.

    num_samples_per_atom : int, optional
        Number of points to sample from the surface of each atom. Default is 20.
        Note that this value is scaled by a given atom's relative vdW radius to a carbon
        and SQUARED. Typically choose a value between 15 and 35. For example, if set to
        20, a carbon atom would have 400 sampled points.

    Returns
    -------
    np.ndarray
        Array of shape (N*num_points_per_atom, 3) containing the coordinates of each
        point sampled from each atom.
    """
    # get surface radius based on vdW radii, cutoff
    # cutoff = 1.4
    if num_samples_per_atom > 50:
        raise ValueError('Do not set num_samples_per_atom to be larger than 50 for performance\
                         issues. The number is squared internally.')
    if not isinstance(radii, np.ndarray):
        radii = np.array(radii)
    # surf_radii = cutoff * (radii / 1.7) + probe_radius # scaled vdW radius + probe radius
    surf_radii = radii + probe_radius # vdW radius + probe radius
    # get number of samples per atom dependent on vdw radii
    num_samples_per_atom = np.ceil((num_samples_per_atom * (radii / 1.7))**2)

    # store points for spheres generated by same sized radii since deterministic
    spheres = {num_samples : _get_points_fibonacci(num_samples) for num_samples in set(num_samples_per_atom)}
    # Apply a random SO(3) rotation to each atom's sphere seperately
    # spheres = np.vstack([np.dot(spheres[num_samples], special_ortho_group.rvs(3).T) for num_samples in num_samples_per_atom])
    # Don't apply random SO(3) rotation since mesh sampling is stochastic and 5-8x faster.
    spheres = np.vstack([spheres[num_samples] for num_samples in num_samples_per_atom])

    # reformat radii and center arrays to match number of samples per atom for elementwise mult and add
    Rs = np.repeat(surf_radii.reshape((-1,1)), [int(n) for n in num_samples_per_atom], axis=0)
    centers = np.repeat(centers, [int(n) for n in num_samples_per_atom], axis=0)

    # Scale surfaces by radii and translate to centers
    surface = (spheres * Rs) + centers
    return surface


def _get_points_fibonacci(num_samples: int):
    """
    Generate points on unit sphere using fibonacci approach.
    Adapted from Morfeus:
    https://github.com/digital-chemistry-laboratory/morfeus/blob/main/morfeus/geometry.py

    Parameters
    ----------
    num_samples : int
        Number of points to sample from the surface of a sphere

    Returns
    -------
    np.ndarray (num_samples,3)
        Coordinates of the sampled points.
    """
    offset = 2.0 / num_samples
    increment = np.pi * (3.0 - np.sqrt(5.0))

    i = np.arange(num_samples)
    y = ((i * offset) - 1) + (offset / 2)
    r = np.sqrt(1 - np.square(y))
    phi = np.mod((i + 1), num_samples) * increment
    x = np.cos(phi) * r
    z = np.sin(phi) * r

    points = np.column_stack((x, y, z))
    return points


def get_point_cloud(points: np.ndarray,
                    color: List[float] = [0.0, 0.0, 0.0]
                    ) -> o3d.geometry.PointCloud:
    """
    Convert np.ndarray of points to a Open3D Point Cloud object.

    Parameters
    ----------
    points : np.ndarray (N, 3)
        Coordinates of points.

    color : list[float] (default=[0,0,0] (black))

    Returns
    -------
    open3d.geometry.PointCloud object
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.paint_uniform_color(color)
    return pcd


def _get_molecular_surface_mesh(centers: np.ndarray,
                                radii: Union[np.ndarray, List],
                                num_samples_per_atom: int = 25,
                                probe_radius: float = 1.2,
                                ball_radii: List[float] = [1.2],
                                color: List[float] = [1.0, 0.0, 0.0]
                                ) -> Tuple[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud]:
    """
    Generate a surface mesh representation of a molecule's surface. The dense point cloud is also
    returned.

    Parameters
    ----------
    centers : np.ndarray (N, 3)
        Cartesian coordinates of the atom centers of a molecule.

    radii : np.ndarray (N,)
        van der Waals radii of each atom (in Angstrom) in the same order as the centers parameter.

    num_samples_per_atom : int (default = 20)
        Number of points to sample from the surface of each atom.
        Note that this value is scaled by a given atom's relative vdW radius to a carbon and
        SQUARED. Typically choose a value between 15 and 35.
            E.g., if set to 20, a carbon atom would have 400 sampled points.

    probe_radius : float (default = 1.2)
        The radius of a probe atom to act as a "solvent accessible surface".
        Default = 1.2 angstroms which is the radius of a Hydrogen atom.

    ball_radii : list[float] (default = [1.2])
        The radius of the ball(s) used in Open3D's ball pivoting algorithm to generate a triangle
        mesh.

    color : list[float] (default = [1., 0., 0.])
        RGB color values for the point cloud (default is red).

    Returns
    -------
    tuple
        o3d.geometry.TriangleMesh : Mesh representing the molecular surface.
        o3d.geometry.PointCloud : Dense point cloud representing the molecular surface.
    """
    points = sample_molecular_surface_with_radius_fibonacci(centers=centers,
                                                            radii=radii,
                                                            probe_radius=probe_radius,
                                                            num_samples_per_atom=num_samples_per_atom
                                                            )
    # distances of every point with respect to the centers of each atom
    dist_matrix = distance.cdist(points, centers)
    # mask out the points within vdw radius of each atom
    mask = np.where(np.all(dist_matrix >= radii + probe_radius - 0.01, axis=1), 1., 0.).astype(bool)

    # generate point cloud
    pcd = get_point_cloud(points[mask], color=color)

    # Generate surface mesh and sample from it evenly
    pcd.estimate_normals()
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(ball_radii))
    return mesh, points


def get_molecular_surface_point_cloud(centers: np.ndarray,
                                      radii: Union[np.ndarray, List],
                                      num_points: Union[int, None] = None,
                                      num_samples_per_atom: int = 25,
                                      probe_radius: float = 1.2,
                                      ball_radii: List[float] = [1.2],
                                      color: List[float] = [1.0, 0.0, 0.0]
                                      ) -> o3d.geometry.PointCloud:
    """
    Gets the point cloud representation of a molecule's van der Waals surface. Takes into account
    the vdW radii of different atoms. Removes overlapping points within vdW radii of neighboring
    atoms.

    Parameters
    ----------
    centers : np.ndarray (N, 3)
        Cartesian coordinates of the atom centers of a molecule.

    radii : np.ndarray (N,)
        van der Waals radii of each atom (in Angstrom) in the same order as the centers parameter.

    num_points : int (default = None)
        The total number of points in the final point cloud. If None, it returns as many as what
        was left after cleaning up the atom-sampled surface point cloud.

    num_samples_per_atom : int, optional
        Number of points to sample from the surface of each atom. Default is 25.
        Note that this value is scaled by a given atom's relative vdW radius to a carbon
        and SQUARED. Typically choose a value between 15 and 35. For example, if set to
        20, a carbon atom would have 400 sampled points.

    probe_radius : float, optional
        The radius of a probe atom to act as a "solvent accessible surface".
        Default is 1.2 angstroms which is the radius of a Hydrogen atom.

    ball_radii : list, optional
        The radius of the ball(s) used in Open3D's ball pivoting algorithm to generate
        a triangle mesh. Default is [1.2].

    color : list, optional
        RGB color values for the point cloud (default is [1., 0., 0.] red).

    Returns
    -------
    o3d.geometry.PointCloud
        Point cloud object representation of the molecular surface.
    """
    mesh, points = _get_molecular_surface_mesh(centers=centers,
                                               radii=radii,
                                               num_samples_per_atom=num_samples_per_atom,
                                               probe_radius=probe_radius,
                                               ball_radii=ball_radii,
                                               color=color)
    if num_points is None:
        num_points = len(points)
    pcd = mesh.sample_points_poisson_disk(num_points)
    return pcd


def get_molecular_surface(centers:np.ndarray,
                          radii:Union[np.ndarray, List],
                          num_points:Union[int, None] = None,
                          num_samples_per_atom:Union[int, None] = None,
                          probe_radius: float = 1.2,
                          ball_radii: List[float] = [1.2],
                          method: str = 'mesh',
                          sdf_s: float = SMOOTH_SDF_S,
                          sdf_iters: int = SMOOTH_SDF_ITERS,
                          sdf_knn: int = SMOOTH_SDF_KNN,
                          sdf_jitter: float = SMOOTH_SDF_JITTER,
                          even: str = SMOOTH_SDF_EVEN,
                          seed: Union[int, None] = None,
                          ) -> np.ndarray:
    """
    Gets the point cloud representation of a molecule's van der Waals surface and outputs a
    numpy array. Takes into account the vdW radii of different atoms. Removes overlapping points
    within vdW radii of neighboring atoms.

    Parameters
    ----------
    centers : np.ndarray (N, 3)
        Cartesian coordinates of the atom centers of a molecule.

    radii : np.ndarray (N,)
        van der Waals radii of each atom (in Angstrom) in the same order as the centers parameter.

    num_points : int (default = None)
        The total number of points in the final point cloud. If None, it returns as many as what
        was left after cleaning up the atom-sampled surface point cloud.

    num_samples_per_atom : int, optional
        Number of points to sample from the surface of each atom. Default is None, which
        lets the chosen method apply its own default (25 for ``method='mesh'``).
        Note that this value is scaled by a given atom's relative vdW radius to a carbon
        and SQUARED. Typically choose a value between 15 and 35. For example, if set to
        20, a carbon atom would have 400 sampled points.

    probe_radius : float, optional
        The radius of a probe atom to act as a "solvent accessible surface".
        Default is 1.2 angstroms which is the radius of a Hydrogen atom.

    ball_radii : list, optional
        The radius of the ball(s) used in Open3D's ball pivoting algorithm to generate
        a triangle mesh. Default is [1.2]. Only used by ``method='mesh'``.

    method : str, optional
        ``'mesh'`` (default) uses Open3D ball pivoting plus Poisson-disk resampling.
        ``'smooth_sdf'`` uses the mesh-free, Open3D-free surfacer
        ``get_molecular_surface_smooth_sdf``, which rounds the concave atom-border seams so
        atom centers are not trivially recoverable from the surface.

    sdf_s, sdf_iters, sdf_knn, sdf_jitter, even, seed
        Forwarded to ``get_molecular_surface_smooth_sdf`` when ``method='smooth_sdf'``;
        ignored otherwise.

    Returns
    -------
    np.ndarray
        Coordinates of points representing the molecular surface.
    """
    method = (method or 'mesh').lower()
    if method == 'mesh':
        # None -> 25, the mesh path's default density
        nspa = 25 if num_samples_per_atom is None else num_samples_per_atom
        pcd = get_molecular_surface_point_cloud(centers=centers, radii=radii,
                                                num_points=num_points,
                                                num_samples_per_atom=nspa,
                                                probe_radius=probe_radius,
                                                ball_radii=ball_radii)
        return np.asarray(pcd.points)
    if method in ('smooth_sdf', 'sdf', 'smooth', 'fast'):
        # None -> SMOOTH_SDF_NSPA (resolved inside)
        return get_molecular_surface_smooth_sdf(centers=centers, radii=radii,
                                                num_points=num_points,
                                                num_samples_per_atom=num_samples_per_atom,
                                                probe_radius=probe_radius,
                                                s=sdf_s, iters=sdf_iters, knn=sdf_knn,
                                                jitter=sdf_jitter, even=even, seed=seed)
    raise ValueError(f"Unknown surface method {method!r}; expected 'mesh' or 'smooth_sdf'.")


# Mesh-free smooth + stochastic surfacer (opt-in; Open3D-free). Samples the envelope of the
# union of (vdW + probe) spheres, projects the points onto a smooth-min implicit surface
# (rounding the concave atom-border seams so atom centers cannot be read off the surface),
# and evenly resamples to an exact count.
def _farthest_point_sample(points: np.ndarray, num_points: Union[int, None], start: int = 0) -> np.ndarray:
    """Even (blue-noise-like) subsample via greedy farthest-point sampling.

    Returns exactly ``min(num_points, len(points))`` points (all of them if num_points is None).
    Deterministic given ``start``. O(num_points * len(points)).
    """
    P = len(points)
    n = P if num_points is None else min(int(num_points), P)
    if n <= 0:
        return points[:0]
    sel = np.empty(n, dtype=np.intp)
    min_d2 = np.full(P, np.inf)
    last = int(start) % P
    for i in range(n):
        sel[i] = last
        d2 = np.sum((points - points[last]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)
        min_d2[last] = -1.0           # never re-pick (handles coincident points)
        last = int(np.argmax(min_d2))
    return points[sel]


def _get_masked_surface_candidates(centers: np.ndarray,
                                   radii: Union[np.ndarray, List],
                                   num_samples_per_atom: int = 25,
                                   probe_radius: float = 1.2,
                                   stochastic: bool = True,
                                   seed: Union[int, None] = None) -> np.ndarray:
    """Sample the outer boundary of the union of (vdW + probe) spheres.

    Samples each atom's sphere and keeps the points outside every other atom's sphere, the
    same masking ``_get_molecular_surface_mesh`` uses. ``stochastic=True`` uses the random
    sampler; ``seed`` makes it reproducible.
    """
    radii = np.asarray(radii)
    if stochastic:
        if seed is not None:
            _state = np.random.get_state()
            np.random.seed(int(seed))
        try:
            points = sample_molecular_surface_with_radius(centers, radii, probe_radius, num_samples_per_atom)
        finally:
            if seed is not None:
                np.random.set_state(_state)
    else:
        points = sample_molecular_surface_with_radius_fibonacci(centers, radii, probe_radius, num_samples_per_atom)
    dist_matrix = distance.cdist(points, centers)
    mask = np.all(dist_matrix >= radii + probe_radius - 0.01, axis=1)
    return points[mask]


def _smoothmin_sdf_project(points: np.ndarray,
                           centers: np.ndarray,
                           a: np.ndarray,
                           s: float = SMOOTH_SDF_S,
                           iters: int = SMOOTH_SDF_ITERS,
                           knn: int = SMOOTH_SDF_KNN) -> np.ndarray:
    """Project points onto the smooth-min (metaball) iso-surface ``g(x)=0`` of the sphere union,

        g(x) = -(1/s) * logsumexp_i( -s * (||x - c_i|| - a_i) )

    via Newton steps ``x <- x - g * grad g / ||grad g||^2`` with the analytic gradient
    ``grad g = sum_i w_i (x-c_i)/||x-c_i||`` (softmin weights ``w_i``). Smaller ``s`` rounds
    the concave atom-border seams more; larger ``s`` approaches the sharp sphere union. Only
    each point's ``knn`` nearest atoms contribute, chosen once up front, so cost is O(M*knn).
    """
    points = np.asarray(points, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    M = len(points)
    if M == 0:
        return points
    k = min(int(knn), len(centers))
    nn = np.argpartition(distance.cdist(points, centers), k - 1, axis=1)[:, :k]  # (M,k) nearest atoms
    C = centers[nn]                          # (M,k,3)
    A = a[nn]                                 # (M,k)
    X = points.copy()
    for _ in range(int(iters)):
        diff = X[:, None, :] - C                              # (M,k,3)
        dist = np.sqrt(np.sum(diff * diff, axis=2)) + 1e-9    # (M,k)
        di = dist - A
        lse = logsumexp(-s * di, axis=1)                      # (M,)
        w = np.exp((-s * di) - lse[:, None])                 # (M,k) softmin weights
        g = -(1.0 / s) * lse                                 # (M,)
        grad = np.einsum('mk,mkd->md', w, diff / dist[..., None])  # (M,3)
        gn2 = np.sum(grad * grad, axis=1) + 1e-12
        X = X - (g / gn2)[:, None] * grad
    return X


def get_molecular_surface_smooth_sdf(centers: np.ndarray,
                                     radii: Union[np.ndarray, List],
                                     num_points: Union[int, None] = None,
                                     num_samples_per_atom: Union[int, None] = None,
                                     probe_radius: float = 1.2,
                                     s: float = SMOOTH_SDF_S,
                                     iters: int = SMOOTH_SDF_ITERS,
                                     knn: int = SMOOTH_SDF_KNN,
                                     jitter: float = SMOOTH_SDF_JITTER,
                                     even: str = SMOOTH_SDF_EVEN,
                                     seed: Union[int, None] = None) -> np.ndarray:
    """Mesh-free, Open3D-free smooth and stochastic molecular surface.

    Samples the envelope of the union of (vdW + probe) spheres, projects the points onto the
    smooth-min implicit iso-surface (rounding the concave atom-border seams so atom centers
    are not trivially recoverable), optionally jitters them off the surface, and resamples
    evenly to exactly ``num_points``. Stochastic per call unless ``seed`` is given. A model
    trained on the mesh surface sees this surface as a distribution shift.

    Parameters
    ----------
    centers, radii, num_points, num_samples_per_atom, probe_radius
        As in ``get_molecular_surface``.
    s, iters, knn
        Smooth-min sharpness, Newton steps, and nearest-atom count (see ``_smoothmin_sdf_project``).
    jitter : float
        Std (Angstrom) of extra outward jitter to mimic the mesh's facet noise. Default 0.
    even : str
        'fps' (even, blue-noise-like; default) or 'random' resample to ``num_points``.
    seed : int or None
        If given, the surface is reproducible; otherwise it varies run to run.

    Returns
    -------
    np.ndarray (num_points, 3) float32
    """
    if num_samples_per_atom is None:
        num_samples_per_atom = SMOOTH_SDF_NSPA
    candidates = _get_masked_surface_candidates(centers, radii,
                                                num_samples_per_atom=num_samples_per_atom,
                                                probe_radius=probe_radius,
                                                stochastic=True, seed=seed)
    if len(candidates) == 0:
        return candidates.astype(np.float32)
    a = np.asarray(radii) + probe_radius
    surf = _smoothmin_sdf_project(candidates, centers, a, s=s, iters=iters, knn=knn)
    if jitter and jitter > 0.0:
        rng = np.random.default_rng(seed)
        nn = np.argmin(distance.cdist(surf, np.asarray(centers, dtype=float)), axis=1)
        nvec = surf - np.asarray(centers, dtype=float)[nn]
        nvec = nvec / (np.linalg.norm(nvec, axis=1, keepdims=True) + 1e-9)
        surf = surf + nvec * rng.normal(0.0, jitter, size=(len(surf), 1))
    if num_points is None:
        out = surf
    elif (even or 'fps').lower() == 'random':
        rng = np.random.default_rng(seed)
        m = min(int(num_points), len(surf))
        out = surf[rng.choice(len(surf), size=m, replace=False)]
    else:
        out = _farthest_point_sample(surf, num_points)
    return np.asarray(out).astype(np.float32)


def get_molecular_surface_point_cloud_const_density(centers: np.ndarray,
                                                    radii: Union[np.ndarray, List],
                                                    density: float = 0.3,
                                                    num_samples_per_atom: int = 25,
                                                    probe_radius: float = 1.2,
                                                    ball_radii: List[float] = [1.2],
                                                    color: List[float] = [1.0, 0.0, 0.0]
                                                    ) -> o3d.geometry.PointCloud:
    """
    Gets the point cloud representation of a molecule's van der Waals surface. Takes into account
    the vdW radii of different atoms. Removes overlapping points within vdW radii of neighboring
    atoms.

    Parameters
    ----------
    centers : np.ndarray (N, 3)
        Cartesian coordinates of the atom centers of a molecule.

    radii : np.ndarray (N,)
        van der Waals radii of each atom (in Angstrom) in the same order as the centers parameter.

    density : float (default = 0.3)
        The density of points on the surface. The number of points is calculated from the solvent
        accessible surface area approximately computed by the surface area of the generated mesh.

    num_samples_per_atom : int, optional
        Number of points to sample from the surface of each atom. Default is 25.
        Note that this value is scaled by a given atom's relative vdW radius to a carbon
        and SQUARED. Typically choose a value between 15 and 35. For example, if set to
        20, a carbon atom would have 400 sampled points.

    probe_radius : float, optional
        The radius of a probe atom to act as a "solvent accessible surface".
        Default is 1.2 angstroms which is the radius of a Hydrogen atom.

    ball_radii : list, optional
        The radius of the ball(s) used in Open3D's ball pivoting algorithm to generate
        a triangle mesh. Default is [1.2].

    color : list, optional
        RGB color values for the point cloud (default is [1., 0., 0.] red).

    Returns
    -------
    o3d.geometry.PointCloud
        Point cloud object representation of the molecular surface.
    """
    mesh, _ = _get_molecular_surface_mesh(centers=centers,
                                          radii=radii,
                                          num_samples_per_atom=num_samples_per_atom,
                                          probe_radius=probe_radius,
                                          ball_radii=ball_radii,
                                          color=color)
    # solv_acc_surf_area = rdFreeSASA.CalcSASA(mol, radii) # solvent accessible surface area
    solv_acc_surf_area = mesh.get_surface_area() # Approximate solvent accessible surface area
    num_points = int(density * solv_acc_surf_area)
    pcd = mesh.sample_points_poisson_disk(num_points)
    return pcd


def get_molecular_surface_const_density(centers:np.ndarray,
                                        radii:Union[np.ndarray, List],
                                        density: float = 0.3,
                                        num_samples_per_atom:int = 25,
                                        probe_radius: float = 1.2,
                                        ball_radii: List[float] = [1.2],
                                        ) -> np.ndarray:
    """
    Gets the point cloud representation of a molecule's van der Waals surface and outputs a
    numpy array. Takes into account the vdW radii of different atoms. Removes overlapping points
    within vdW radii of neighboring atoms.

    Parameters
    ----------
    centers : np.ndarray (N, 3)
        Cartesian coordinates of the atom centers of a molecule.

    radii : np.ndarray (N,)
        van der Waals radii of each atom (in Angstrom) in the same order as the centers parameter.

    density : float (default = 0.3)
        The density of points on the surface. The number of points is calculated from the solvent
        accessible surface area approximately computed by the surface area of the generated mesh.

    num_samples_per_atom : int, optional
        Number of points to sample from the surface of each atom. Default is 25.
        Note that this value is scaled by a given atom's relative vdW radius to a carbon
        and SQUARED. Typically choose a value between 15 and 35. For example, if set to
        20, a carbon atom would have 400 sampled points.

    probe_radius : float, optional
        The radius of a probe atom to act as a "solvent accessible surface".
        Default is 1.2 angstroms which is the radius of a Hydrogen atom.

    ball_radii : list, optional
        The radius of the ball(s) used in Open3D's ball pivoting algorithm to generate
        a triangle mesh. Default is [1.2].

    Returns
    -------
    np.ndarray
        Coordinates of points representing the molecular surface.
    """
    pcd = get_molecular_surface_point_cloud_const_density(centers=centers,
                                                          radii=radii,
                                                          density=density,
                                                          num_samples_per_atom=num_samples_per_atom,
                                                          probe_radius=probe_radius,
                                                          ball_radii=ball_radii)
    return np.asarray(pcd.points)


def get_electrostatics(mol: Chem.Mol, points: np.ndarray) -> np.ndarray:
    """
    Compute the Coulomb potential values at each point for a given molecule.
    Assumes the input "mol" already has an optimized conformer. Gets partial charges from
    MMFF or Gasteiger.

    Parameters
    ----------
    mol : rdkit.Chem.Mol object
        RDKit molecule object with an optimized geometry in conformers.

    points : np.ndarray (N, 3)
        Coordinates of sampled points to compute Coulomb potential at.

    Returns
    -------
    np.ndarray (N)
        Electrostatic potential values corresponding to each point.
    """
    try:
        mol.GetConformer()
    except ValueError as e:
        raise ValueError("Provided rdkit.Chem.Mol object did not have conformer embedded.", e)

    charge_mol = Chem.Mol(mol)
    molec_props = Chem.AllChem.MMFFGetMoleculeProperties(charge_mol)
    if molec_props:
        charges = np.array([molec_props.GetMMFFPartialCharge(i) for i, _ in enumerate(mol.GetAtoms())])
    else:
        print("MMFF charges not available for the input molecule, defaulting to Gasteiger charges.")
        AllChem.ComputeGasteigerCharges(mol)
        charges=np.array([a.GetDoubleProp('_GasteigerCharge') for a in mol.GetAtoms()])

    centers = mol.GetConformer().GetPositions()

    distances = np.linalg.norm(points[:, np.newaxis] - centers, axis=2)
    # Calculate the potentials
    E_pot = np.dot(charges, 1 / distances.T) * COULOMB_SCALING
    # Ensure that invalid distances (where distance is 0) are handled
    E_pot[np.isinf(E_pot)] = 0
    return E_pot


def get_electrostatics_given_point_charges(charges: np.ndarray,
                                           positions: np.ndarray,
                                           points: np.ndarray)-> np.ndarray:
    """
    Compute the Coulomb potential values at each point for a given set of charges at defined positions.

    Parameters
    ----------
    charges : np.ndarray (N,)
        Charges, with units [V]

    positions : np.ndarray (N, 3)
        Coordinates of point charges, with units of A.

    points : np.ndarray (M, 3)
        Coordinates of point cloud at which to compute electrostatic potential, with units of A.

    Returns
    -------
    np.ndarray (M,)
        Electrostatic potential values corresponding to each point.
    """

    distances = np.linalg.norm(points[:, np.newaxis] - positions, axis=2)
    # Calculate the potentials
    E_pot = np.dot(charges, 1. / distances.T) * COULOMB_SCALING # [eV/e] = [V]
    # Ensure that invalid distances (where distance is 0) are handled
    E_pot[np.isinf(E_pot)] = 0
    return E_pot


def color_pcd_with_electrostatics(pcd: o3d.geometry.PointCloud, E_pot: np.ndarray) -> None:
    """
    Color the point cloud based on elecrostatic potential. Colors are only scaled to the molecule
    itself (i.e., the colors are not comparable to different molecules).
    Red is positive, blue is negative, black is neutral.
    """

    colors = np.zeros((len(E_pot), 3))
    colors[:,0] = np.where(E_pot < 0, 0, E_pot/np.max(E_pot)).squeeze()
    colors[:,2] = np.where(E_pot >= 0, 0, -E_pot/np.max(-E_pot)).squeeze()
    pcd.colors = o3d.utility.Vector3dVector(colors)


def get_sample_atom_volume(R: float, num_samples: int) -> np.ndarray:
    """
    Sample points uniformly from a sphere.
    https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume

    Parameters
    ----------
    R : float
        Radius
    num_samples : int
        Number of samples

    Returns
    -------
    np.ndarray (num_samples, 3)
    """
    phi = np.random.uniform(0, 2*np.pi, num_samples)
    cos_theta = np.random.uniform(-1, 1, num_samples)
    u = np.random.uniform(0,1, num_samples)

    theta = np.arccos(cos_theta)
    r = R * u**(1/3)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.column_stack((x, y, z))


def get_molecular_volume(centers: np.ndarray, radii: np.ndarray, num_samples: int = None, num_samples_per_atom: int=15):
    """
    Sample points uniformly from a molecule's volume. Does not include probe radius.
    Ignores possible overlap of vdW radius of atoms when randomly sub-selecting points for the
    volume.

    Parameters
    ----------
    centers : np.ndarray (N, 3)
        Cartesian coordinates of the atom centers of a molecule.

    radii : np.ndarray (N,)
        van der Waals radii of each atom (in Angstrom) in the same order as the centers parameter.

    num_samples_per_atom : int, optional
        Number of points to sample from the surface of each atom. Default is 15.
        Note that this value is scaled by a given atom's relative vdW radius to a carbon
        and CUBED. Typically choose a value between 10 and 20. For example, if set to
        20, a carbon atom would have 8000 sampled points.

    Returns
    -------
    np.ndarray
        Coordinates of points representing the molecular surface.
    """
    if num_samples_per_atom > 20:
        raise ValueError('Do not set num_samples_per_atom to be larger than 20 for performance\
                         issues. The number is cubed internally.')
    # cutoff = 1.4
    # radii = cutoff * (radii / 1.7)
    num_samples_per_atom = np.ceil((num_samples_per_atom * (radii / 1.7))**3)

    points = np.vstack([get_sample_atom_volume(r, int(num_samples_per_atom[i])) for i, r in enumerate(radii)])
    centers = np.repeat(centers, [int(n) for n in num_samples_per_atom], axis=0)
    points += centers # translate atoms to their centers

    # Subselect points by masking
    if num_samples is not None:
        idx = np.arange(len(points))
        np.random.shuffle(idx)
        if num_samples > len(points):
            num_samples = len(points)
        points = points[idx[:num_samples]]
    return points
