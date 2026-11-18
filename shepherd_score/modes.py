"""
Alignment modes for Torch (autograd and analytical) and JAX.
For accelerated modes, look at shepherd_score/accel/_modes.py.

A mode defines which arrays to read from a ``Molecule`` and how to score a posed fit against the
reference. Everything else (seeding, the SE(3) optimizer, result bookkeeping) is shared, so a
new mode is a small class and needs no new ``MoleculePair`` method::

    pair.align("vol")                              # by name, parameters as keywords
    pair.align("surf_esp", lam=0.3, similarity="tversky")
    pair.align(MyMode(width=1.2))                  # or a Mode instance

Writing a mode:
1. Subclass ``Mode`` as a frozen dataclass
2. Implement ``inputs`` and ``overlap`` (a Gaussian-type cross overlap; without similarity reduction
3. list which inputs move with the pose in ``points`` / ``vectors``
4. Override ``score`` instead of ``overlap`` when the score is not an overlap ratio.

Optionally, blend modes with ``Combined``.
"""
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, Optional, Tuple

import numpy as np
import torch

from shepherd_score.alignment.utils.se3 import apply_SE3_transform, apply_SO3_transform
from shepherd_score.score.constants import LAM_SCALING
from shepherd_score.score.gaussian_overlap import VAB_2nd_order, get_linear_hard_sphere_overlap
from shepherd_score.score.electrostatic_scoring import VAB_2nd_order_esp, _esp_comparison
from shepherd_score.score.pharmacophore_scoring import get_overlap_pharm
from shepherd_score.score.atomtype_scoring import get_overlap_atomtype
from shepherd_score.score.similarity import (Similarity, similarity_from_overlaps, pharm_similarity,
                                             pharm_similarity_name)


def _esp_overlap(points_1, points_2, values_1, values_2, alpha, lam):
    """``VAB_2nd_order_esp`` with per-point values given as (N,) or (B, N)."""
    v1 = values_1.unsqueeze(-1)
    v2 = values_2.unsqueeze(-1)
    return VAB_2nd_order_esp(points_1, points_2, v1, v2, alpha, lam)


def _heavy_positions(mol) -> np.ndarray:
    """Heavy-atom coordinates aligned 1:1 with the ``no_H=True`` per-atom getters."""
    return np.ascontiguousarray(mol.mol.GetConformer().GetPositions()[mol._nonH_atoms_idx])


def _all_positions(mol) -> np.ndarray:
    return mol.mol.GetConformer().GetPositions()


def _repeat_to_batch(ref: dict, fit: dict, key: str) -> dict:
    """Repeat unbatched ``ref`` inputs over the pose batch of ``fit`` (for scorers that do not
    broadcast)."""
    if fit[key].dim() == ref[key].dim() + 1:
        n = fit[key].shape[0]
        return {k: v.repeat((n,) + (1,) * v.dim()) for k, v in ref.items()}
    return ref


def _pose_moved(mode, inputs: dict, se3: torch.Tensor, already_posed: dict) -> dict:
    """``Mode.pose`` that reuses earlier results. ``already_posed`` maps
    ``(id(array), transform)`` to that array under ``se3``; arrays found there are not
    transformed again, and new results are added to it."""
    posed = dict(inputs)
    for keys, transform in ((mode.points, apply_SE3_transform), (mode.vectors, apply_SO3_transform)):
        for k in keys:
            x = inputs[k]
            if transform is apply_SE3_transform and x.shape[-2] == 0:
                continue
            key = (id(x), transform)
            if key not in already_posed:
                already_posed[key] = transform(x, se3)
            posed[k] = already_posed[key]
    return posed


class Mode:
    """Base class for alignment modes. See the module docstring."""
    name: ClassVar[str] = ''
    points: ClassVar[Tuple[str, ...]] = ('points',)    # inputs moved by the pose
    vectors: ClassVar[Tuple[str, ...]] = ()            # inputs rotated by the pose
    seed: ClassVar[str] = 'points'                     # input whose COM/PCA seeds the optimizer
    requires: ClassVar[Tuple[str, ...]] = ()           # 'surface' and/or 'pharmacophores'
    similarity: Similarity = 'tanimoto'

    def inputs(self, mol) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def overlap(self, a: dict, b: dict) -> torch.Tensor:
        raise NotImplementedError

    def prepare(self, ref: dict, fit: dict):
        """Pose-invariant quantities, computed once per alignment (the self overlaps)."""
        return self.overlap(ref, ref), self.overlap(fit, fit)

    def score(self, ref: dict, fit: dict, cache=None) -> torch.Tensor:
        """Similarity of the (already posed) ``fit`` to ``ref``."""
        AA, BB = cache if cache is not None else self.prepare(ref, fit)
        return similarity_from_overlaps(self.overlap(ref, fit), AA, BB, self.similarity)

    def pose(self, inputs: dict, se3: torch.Tensor) -> dict:
        return _pose_moved(self, inputs, se3, {})

    def seed_points(self, inputs: dict) -> torch.Tensor:
        return inputs[self.seed]

    def trans_centers(self, mol) -> np.ndarray:
        """Where ``trans_init`` places the fit's center of mass."""
        return mol.atom_pos

    def backends(self) -> Dict[str, Callable]:
        """Specialized optimizers (``'analytical'``, ``'jax'``) that replace the generic
        autograd loop. Each takes ``(ref, fit, **opt)`` and returns ``(aligned, se3, score)``."""
        return {}


# ---------------------------------------------------------------------------------------------
# Shape

def _rocs_analytical(ref, fit, alpha, **opt):
    from shepherd_score.alignment import optimize_ROCS_overlay_analytical
    aligned, se3, score = optimize_ROCS_overlay_analytical(ref['points'], fit['points'], alpha, **opt)
    return {'points': aligned}, se3, score


def _rocs_jax(ref, fit, alpha, **opt):
    from shepherd_score.alignment_jax import optimize_ROCS_overlay_jax
    aligned, se3, score = optimize_ROCS_overlay_jax(ref['points'], fit['points'], alpha, **opt)
    return {'points': aligned}, se3, score


@dataclass(frozen=True)
class _Shape(Mode):
    alpha: float = 0.81
    similarity: Similarity = 'tanimoto'

    def overlap(self, a, b):
        return VAB_2nd_order(a['points'], b['points'], self.alpha)

    def backends(self):
        if self.similarity != 'tanimoto':
            return {}
        return {'analytical': partial(_rocs_analytical, alpha=self.alpha),
                'jax': partial(_rocs_jax, alpha=self.alpha)}


@dataclass(frozen=True)
class Vol(_Shape):
    """Atom-centered Gaussian volume overlap (heavy atoms unless ``no_H=False``)."""
    name: ClassVar[str] = 'vol'
    no_H: bool = True

    def inputs(self, mol):
        return {'points': mol.get_positions(self.no_H)}


@dataclass(frozen=True)
class Surf(_Shape):
    """Gaussian overlap of the molecular surface point clouds."""
    name: ClassVar[str] = 'surf'
    requires: ClassVar[Tuple[str, ...]] = ('surface',)

    def inputs(self, mol):
        return {'points': mol.surf_pos}


# ---------------------------------------------------------------------------------------------
# Excluded volume

def _as_points(points, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(np.asarray(points, dtype=np.float32), dtype=like.dtype, device=like.device)


@dataclass(frozen=True, eq=False)    # eq=False: holds an array
class Avoid(Mode):
    """Linear hard-sphere penalty ``sum relu((min_dist - d) / min_dist)`` between the fit's atoms
    and ``avoid_points``, a fixed (K, 3) cloud in the reference frame that belongs to neither
    molecule. It is positive, so weight it negatively in a ``Combined``."""
    name: ClassVar[str] = 'avoid'
    avoid_points: Any = None
    min_dist: float = 2.0
    no_H: bool = True

    def __post_init__(self):
        if self.avoid_points is None:
            raise ValueError(f'{type(self).__name__} needs avoid_points, a (K, 3) cloud in the '
                             'reference frame.')

    def inputs(self, mol):
        return {'points': mol.get_positions(self.no_H)}

    def prepare(self, ref, fit):
        return _as_points(self.avoid_points, fit['points'])

    def score(self, ref, fit, cache=None):
        avoid = cache if cache is not None else self.prepare(ref, fit)
        return get_linear_hard_sphere_overlap(fit['points'], avoid, self.min_dist)


def _rocs_avoid_analytical(ref, fit, alpha, avoid_points, avoid_min_dist, avoid_weight, **opt):
    from shepherd_score.alignment import optimize_ROCS_overlay_analytical
    aligned, se3, score = optimize_ROCS_overlay_analytical(
        ref['points'], fit['points'], alpha, avoid_points=_as_points(avoid_points, ref['points']),
        avoid_min_dist=avoid_min_dist, avoid_weight=avoid_weight, **opt)
    return {'points': aligned}, se3, score


@dataclass(frozen=True, eq=False)
class VolAvoid(Vol):
    """``vol`` minus ``avoid_weight`` times the ``Avoid`` penalty on the same atoms."""
    name: ClassVar[str] = 'vol_avoid'
    avoid_points: Any = None
    avoid_weight: float = 1.0
    avoid_min_dist: float = 2.0

    def __post_init__(self):
        object.__setattr__(self, '_avoid', Avoid(self.avoid_points, self.avoid_min_dist, self.no_H))

    def prepare(self, ref, fit):
        return super().prepare(ref, fit), self._avoid.prepare(ref, fit)

    def score(self, ref, fit, cache=None):
        shape, avoid = cache if cache is not None else self.prepare(ref, fit)
        return super().score(ref, fit, shape) - self.avoid_weight * self._avoid.score(ref, fit, avoid)

    def backends(self):
        if self.similarity != 'tanimoto':
            return {}
        return {'analytical': partial(_rocs_avoid_analytical, alpha=self.alpha,
                                      avoid_points=self.avoid_points,
                                      avoid_min_dist=self.avoid_min_dist,
                                      avoid_weight=self.avoid_weight)}


# ---------------------------------------------------------------------------------------------
# Electrostatics

def _esp_analytical(ref, fit, alpha, lam, **opt):
    from shepherd_score.alignment import optimize_ROCS_esp_overlay_analytical
    aligned, se3, score = optimize_ROCS_esp_overlay_analytical(
        ref['points'], fit['points'], ref['charges'], fit['charges'], alpha, lam, **opt)
    return {'points': aligned}, se3, score


def _esp_jax(ref, fit, alpha, lam, **opt):
    from shepherd_score.alignment_jax import optimize_ROCS_esp_overlay_jax
    aligned, se3, score = optimize_ROCS_esp_overlay_jax(
        ref['points'], fit['points'], ref['charges'], fit['charges'], alpha, lam, **opt)
    return {'points': aligned}, se3, score


@dataclass(frozen=True)
class _ESP(Mode):
    alpha: float = 0.81
    similarity: Similarity = 'tanimoto'

    def _lam(self):
        return self.lam

    def overlap(self, a, b):
        return _esp_overlap(a['points'], b['points'], a['charges'], b['charges'], self.alpha, self._lam())

    def backends(self):
        if self.similarity != 'tanimoto':
            return {}
        return {'analytical': partial(_esp_analytical, alpha=self.alpha, lam=self._lam()),
                'jax': partial(_esp_jax, alpha=self.alpha, lam=self._lam())}


@dataclass(frozen=True)
class VolESP(_ESP):
    """Atom-centered Gaussian overlap weighted by partial-charge agreement.

    ``lam`` is the width of the charge kernel ``exp(-|q_i - q_j|^2 / lam)``, used raw.
    """
    name: ClassVar[str] = 'vol_esp'
    lam: float = 0.1
    no_H: bool = True

    def inputs(self, mol):
        pos = _heavy_positions(mol) if self.no_H else _all_positions(mol)
        return {'points': pos, 'charges': mol.get_charges(self.no_H)}


@dataclass(frozen=True)
class SurfESP(_ESP):
    """Surface Gaussian overlap weighted by electrostatic potential agreement.

    ``lam`` is given in the conventional units and scaled by ``LAM_SCALING`` internally.
    """
    name: ClassVar[str] = 'surf_esp'
    requires: ClassVar[Tuple[str, ...]] = ('surface',)
    lam: float = 0.3

    def inputs(self, mol):
        return {'points': mol.surf_pos, 'charges': mol.surf_esp}

    def _lam(self):
        return LAM_SCALING * self.lam


def _combo_jax(ref, fit, alpha, lam, probe_radius, esp_weight, **opt):
    from shepherd_score.alignment_jax import optimize_esp_combo_score_overlay_jax
    aligned, se3, score = optimize_esp_combo_score_overlay_jax(
        ref_centers_w_H=ref['centers_w_H'], fit_centers_w_H=fit['centers_w_H'],
        ref_centers=ref['centers'], fit_centers=fit['centers'],
        ref_points=ref['points'], fit_points=fit['points'],
        ref_partial_charges=ref['partial_charges'], fit_partial_charges=fit['partial_charges'],
        ref_surf_esp=ref['surf_esp'], fit_surf_esp=fit['surf_esp'],
        ref_radii=ref['radii'], fit_radii=fit['radii'],
        alpha=alpha, lam=lam, probe_radius=probe_radius, esp_weight=esp_weight, **opt)
    return {'points': aligned}, se3, score


@dataclass(frozen=True)
class VolAndSurfESP(Mode):
    """ShaEP-style score: ``esp_weight * esp_agreement + (1 - esp_weight) * shape``.

    Shape is volumetric when ``alpha == 0.81`` and surface-based otherwise. ``similarity``
    applies to the shape term; the ESP agreement is an average, not an overlap ratio.
    """
    name: ClassVar[str] = 'vol_and_surf_esp'
    points: ClassVar[Tuple[str, ...]] = ('points', 'centers', 'centers_w_H')
    requires: ClassVar[Tuple[str, ...]] = ('surface',)
    alpha: float = 0.81
    lam: float = 0.001
    probe_radius: float = 1.0
    esp_weight: float = 0.5
    similarity: Similarity = 'tanimoto'

    def inputs(self, mol):
        return {'centers_w_H': _all_positions(mol),
                'centers': mol.atom_pos if self.alpha == 0.81 else mol.surf_pos,
                'points': mol.surf_pos,
                'partial_charges': mol.partial_charges,
                'surf_esp': mol.surf_esp,
                'radii': mol.radii}

    def overlap(self, a, b):
        return VAB_2nd_order(a['centers'], b['centers'], self.alpha)

    def score(self, ref, fit, cache=None):
        shape = super().score(ref, fit, cache)
        esp_1 = _esp_comparison(ref['points'], fit['centers_w_H'], fit['partial_charges'],
                                ref['surf_esp'], fit['radii'], self.probe_radius, self.lam)
        esp_2 = _esp_comparison(fit['points'], ref['centers_w_H'], ref['partial_charges'],
                                fit['surf_esp'], ref['radii'], self.probe_radius, self.lam)
        esp = (esp_1 + esp_2) / (ref['points'].shape[-2] + fit['points'].shape[-2])
        return self.esp_weight * esp + (1 - self.esp_weight) * shape

    def backends(self):
        if self.similarity != 'tanimoto':
            return {}
        return {'jax': partial(_combo_jax, alpha=self.alpha, lam=self.lam,
                               probe_radius=self.probe_radius, esp_weight=self.esp_weight)}


# ---------------------------------------------------------------------------------------------
# Pharmacophores

def _pharm_optimizer(fn, ref, fit, **opt):
    anchors, vectors, se3, score = fn(ref['types'], fit['types'], ref['anchors'], fit['anchors'],
                                      ref['vectors'], fit['vectors'], **opt)
    return {'types': fit['types'], 'anchors': anchors, 'vectors': vectors}, se3, score


@dataclass(frozen=True)
class Pharm(Mode):
    """Directional pharmacophore overlap.

    ``similarity`` uses the pharmacophore scorer's own reduction, in which Tversky is clamped at
    1. The analytical and JAX optimizers take only the named similarities, so other ``(a, b)``
    weights run on the generic optimizer.
    """
    name: ClassVar[str] = 'pharm'
    points: ClassVar[Tuple[str, ...]] = ('anchors',)
    vectors: ClassVar[Tuple[str, ...]] = ('vectors',)
    seed: ClassVar[str] = 'anchors'
    requires: ClassVar[Tuple[str, ...]] = ('pharmacophores',)
    similarity: Similarity = 'tanimoto'
    extended_points: bool = False
    only_extended: bool = False
    directionless: bool = False

    def inputs(self, mol):
        return {'types': mol.pharm_types, 'anchors': mol.pharm_ancs, 'vectors': mol.pharm_vecs}

    def prepare(self, ref, fit):
        """Self overlaps (VAA, VBB), as ``optimize_pharm_overlay`` precomputes them. ``None`` for
        directional extended points, which the scorer always recomputes."""
        if self.extended_points and not self.directionless:
            return None
        from shepherd_score.score.analytical_gradients import compute_self_overlaps_pharm
        return compute_self_overlaps_pharm(ref['types'], fit['types'], ref['anchors'], fit['anchors'],
                                           ref['vectors'], fit['vectors'],
                                           directionless=self.directionless)

    def score(self, ref, fit, cache=None):
        ref = _repeat_to_batch(ref, fit, 'anchors')
        return get_overlap_pharm(ref['types'], fit['types'], ref['anchors'], fit['anchors'],
                                 ref['vectors'], fit['vectors'],
                                 similarity=pharm_similarity(self.similarity),
                                 extended_points=self.extended_points,
                                 only_extended=self.only_extended,
                                 directionless=self.directionless,
                                 precomputed_self_overlaps=cache)

    def trans_centers(self, mol):
        return mol.pharm_ancs

    def backends(self):
        if self.directionless or not isinstance(pharm_similarity(self.similarity), str):
            return {}
        from shepherd_score.alignment import optimize_pharm_overlay_analytical
        kw = dict(similarity=pharm_similarity_name(self.similarity),
                  extended_points=self.extended_points, only_extended=self.only_extended)

        def jax(ref, fit, vectorized=True, **opt):
            from shepherd_score.alignment_jax import (optimize_pharm_overlay_jax,
                                                      optimize_pharm_overlay_jax_vectorized)
            fn = optimize_pharm_overlay_jax_vectorized if vectorized else optimize_pharm_overlay_jax
            return _pharm_optimizer(fn, ref, fit, **kw, **opt)

        return {'analytical': partial(_pharm_optimizer, optimize_pharm_overlay_analytical, **kw),
                'jax': jax,
                'jax_unvectorized': partial(jax, vectorized=False)}


# ---------------------------------------------------------------------------------------------
# Per-atom scalar fields and atom types (scored at heavy-atom centers)

_FIELDS = {
    'lipo': lambda mol: mol.get_lipophilicity(no_H=True),
    'mr': lambda mol: mol.get_molar_refractivity(no_H=True),
    'fukui': lambda mol: mol.get_fukui(no_H=True),
}


def _empty(ref, fit) -> bool:
    return ref['points'].shape[-2] == 0 or fit['points'].shape[-2] == 0


def _zeros_like_batch(fit) -> torch.Tensor:
    return torch.zeros(fit['points'].shape[:-2], device=fit['points'].device)


@dataclass(frozen=True)
class Field(Mode):
    """A per-atom scalar field (``'lipo'``, ``'mr'`` or ``'fukui'``) overlaid like a charge
    field: heavy-atom Gaussians weighted by ``exp(-|v_i - v_j|^2 / lam)``. Scores 0 when either
    molecule has no heavy atoms."""
    name: ClassVar[str] = 'field'
    field: str = 'lipo'
    alpha: float = 0.81
    lam: float = 0.1
    similarity: Similarity = 'tanimoto'

    def inputs(self, mol):
        return {'points': _heavy_positions(mol), 'values': _FIELDS[self.field](mol)}

    def overlap(self, a, b):
        return _esp_overlap(a['points'], b['points'], a['values'], b['values'], self.alpha, self.lam)

    def score(self, ref, fit, cache=None):
        if _empty(ref, fit):
            return _zeros_like_batch(fit)
        return super().score(ref, fit, cache).reshape(fit['points'].shape[:-2])


@dataclass(frozen=True)
class AtomType(Mode):
    """Gaussian overlap in which only same-element heavy atoms contribute."""
    name: ClassVar[str] = 'atomtype'
    alpha: float = 0.81
    similarity: Similarity = 'tanimoto'

    def inputs(self, mol):
        return {'points': _heavy_positions(mol), 'labels': mol.get_atomic_numbers(no_H=True)}

    def prepare(self, ref, fit):
        return None

    def score(self, ref, fit, cache=None):
        if _empty(ref, fit):
            return _zeros_like_batch(fit)
        # the scorer takes unbatched labels and (optionally batched) centers
        ref_points = _repeat_to_batch({'points': ref['points']}, fit, 'points')['points']
        fit_labels = fit['labels'][0] if fit['labels'].dim() == 2 else fit['labels']
        return get_overlap_atomtype(ref['labels'], fit_labels, ref_points, fit['points'],
                                    self.alpha, pharm_similarity_name(self.similarity)
                                    ).reshape(fit['points'].shape[:-2])


# ---------------------------------------------------------------------------------------------
# Blends

def _poses_by_default(mode: Mode) -> bool:
    """Whether ``mode`` moves its inputs with the base ``Mode.pose`` (so poses can be shared)."""
    return type(mode).pose is Mode.pose


@dataclass(frozen=True)
class Combined(Mode):
    """Weighted sum of modes, all moved by one pose: ``sum(w_i * mode_i)``.

    A convenience class for combining modes: each term is scored separately under the generic
    autograd optimizer.

    This is a generic operator and may not be optimized for speed. Speedups can be achieved
    by creating a new mode with a fused objective, analytical optimizer, or batched kernel
    in ``accel``.

    Seeding and ``trans_init`` follow the first mode. Inputs and aligned outputs are tuples
    with one entry per mode.

    There may be several terms that move the same set of input coordinates
    For example, in ``vol_lipo``, both the shape and the logP field sit on the heavy atoms.
    ``inputs`` compares the arrays each term moves with the pose (its ``points`` and ``vectors``),
    once per alignment, and gives terms with equal values one shared array object.
    ``pose`` then transforms each shared array once per optimizer step instead of once per term.
    """
    terms: Tuple[Tuple[float, Mode], ...] = ()
    label: str = 'combined'

    @property
    def requires(self):
        return tuple({r for _, m in self.terms for r in m.requires})

    def inputs(self, mol):
        seen = []
        out = []
        for _, m in self.terms:
            x = m.inputs(mol)
            if _poses_by_default(m):
                x = dict(x)
                for k in (*m.points, *m.vectors):
                    same = next((s for s in seen if np.shape(s) == np.shape(x[k])
                                 and np.array_equal(s, x[k])), None)
                    if same is None:
                        seen.append(x[k])
                    else:
                        x[k] = same
            out.append(x)
        return tuple(out)

    def prepare(self, ref, fit):
        return tuple(m.prepare(r, f) for (_, m), r, f in zip(self.terms, ref, fit))

    def score(self, ref, fit, cache=None):
        cache = cache if cache is not None else (None,) * len(self.terms)
        return sum(w * m.score(r, f, c) for (w, m), r, f, c in zip(self.terms, ref, fit, cache))

    def pose(self, inputs, se3):
        already_posed = {}    # shared by all terms, so a shared array is transformed once
        return tuple(_pose_moved(m, x, se3, already_posed) if _poses_by_default(m) else m.pose(x, se3)
                     for (_, m), x in zip(self.terms, inputs))

    def seed_points(self, inputs):
        return self.terms[0][1].seed_points(inputs[0])

    def trans_centers(self, mol):
        return self.terms[0][1].trans_centers(mol)


def blend(shape: Mode, other: Mode, weight: float, label: str) -> Combined:
    """``(1 - weight) * shape + weight * other``."""
    return Combined(terms=((1 - weight, shape), (weight, other)), label=label)


def vol_color(color_weight: float = 0.5, alpha: float = 0.81, similarity: Similarity = 'tanimoto',
              directionless: bool = True, extended_points: bool = False, only_extended: bool = False,
              color_similarity: Optional[Similarity] = None, no_H: bool = True) -> Combined:
    """Shape + pharmacophore color (``directionless=False`` keeps the vector weighting).
    ``color_similarity`` defaults to ``similarity``. ``no_H`` selects the shape atoms, as in
    ``vol`` (color is unaffected)."""
    color = Pharm(similarity=similarity if color_similarity is None else color_similarity,
                  directionless=directionless, extended_points=extended_points,
                  only_extended=only_extended)
    return blend(Vol(alpha=alpha, similarity=similarity, no_H=no_H), color, color_weight,
                 'vol_color')


def vol_color_avoid(avoid_points: Any = None, color_weight: float = 0.5, avoid_weight: float = 1.0,
                    avoid_min_dist: float = 2.0, alpha: float = 0.81,
                    similarity: Similarity = 'tanimoto', directionless: bool = True,
                    extended_points: bool = False, only_extended: bool = False,
                    color_similarity: Optional[Similarity] = None, no_H: bool = True) -> Combined:
    """``vol_color`` minus ``avoid_weight`` times the ``Avoid`` penalty on the fit's shape atoms::

        (1 - color_weight) * shape + color_weight * color
            - avoid_weight * sum relu((avoid_min_dist - d) / avoid_min_dist)

    ``avoid_points`` is a fixed (K, 3) cloud in the reference frame, as in ``vol_avoid``. The
    penalty is unnormalized, so the score is at most the ``vol_color`` score (1 on itself with
    the cloud out of reach) and can be negative. ``color_similarity`` defaults to ``similarity``;
    ``similarity`` does not affect the penalty; ``no_H`` selects the atoms of both the shape
    term and the penalty."""
    base = vol_color(color_weight=color_weight, alpha=alpha, similarity=similarity,
                     directionless=directionless, extended_points=extended_points,
                     only_extended=only_extended, color_similarity=color_similarity, no_H=no_H)
    return Combined(terms=base.terms + ((-avoid_weight,
                                         Avoid(avoid_points, min_dist=avoid_min_dist, no_H=no_H)),),
                    label='vol_color_avoid')


def _vol_field(field: str, weight: float, alpha: float, lam: float, similarity: Similarity) -> Combined:
    return blend(Vol(alpha=alpha, similarity=similarity),
                 Field(field=field, alpha=alpha, lam=lam, similarity=similarity),
                 weight, f'vol_{field}')


def vol_lipo(lipo_weight: float = 0.5, alpha: float = 0.81, lam: float = 0.1,
             similarity: Similarity = 'tanimoto') -> Combined:
    """Shape + Crippen logP field."""
    return _vol_field('lipo', lipo_weight, alpha, lam, similarity)


def vol_mr(mr_weight: float = 0.5, alpha: float = 0.81, lam: float = 0.1,
           similarity: Similarity = 'tanimoto') -> Combined:
    """Shape + Crippen molar-refractivity field."""
    return _vol_field('mr', mr_weight, alpha, lam, similarity)


def vol_fukui(fukui_weight: float = 0.5, alpha: float = 0.81, lam: float = 0.1,
              similarity: Similarity = 'tanimoto') -> Combined:
    """Shape + condensed Fukui dual-descriptor field (``Molecule.fukui``)."""
    return _vol_field('fukui', fukui_weight, alpha, lam, similarity)


def vol_atomtype(atomtype_weight: float = 0.5, alpha: float = 0.81,
                 similarity: Similarity = 'tanimoto', type_similarity: Optional[Similarity] = None
                 ) -> Combined:
    """Shape + element-identity overlap. ``type_similarity`` defaults to ``similarity``."""
    return blend(Vol(alpha=alpha, similarity=similarity),
                 AtomType(alpha=alpha, similarity=similarity if type_similarity is None else type_similarity),
                 atomtype_weight, 'vol_atomtype')


# ---------------------------------------------------------------------------------------------
# Registry of names accepted by ``MoleculePair.align``

#: Registry mapping the mode names accepted by ``MoleculePair.align`` to their :class:`Mode` classes or factories.
MODES: Dict[str, Callable[..., Mode]] = {
    'vol': Vol,
    'vol_avoid': VolAvoid,
    'vol_esp': VolESP,
    'surf': Surf,
    'surf_esp': SurfESP,
    'vol_and_surf_esp': VolAndSurfESP,
    'pharm': Pharm,
    'vol_color': vol_color,
    'vol_color_avoid': vol_color_avoid,
    'vol_pharm': partial(vol_color, directionless=False),
    'vol_lipo': vol_lipo,
    'vol_mr': vol_mr,
    'vol_fukui': vol_fukui,
    'vol_atomtype': vol_atomtype,
}
ALIASES = {'esp': 'surf_esp', 'esp_combo': 'vol_and_surf_esp'}


def get_mode(mode, **params) -> Mode:
    """A ``Mode`` from a registered name and its parameters, or a ``Mode`` passed through."""
    if isinstance(mode, Mode):
        if params:
            raise TypeError(f'Parameters {sorted(params)} cannot be combined with a Mode instance; '
                            'set them on the instance instead.')
        return mode
    name = ALIASES.get(mode, mode)
    if name not in MODES:
        raise ValueError(f'Unknown mode {mode!r}. Registered modes: {", ".join(MODES)}.')
    return MODES[name](**params)
