"""
Reduction of Gaussian overlaps to a similarity score.

``similarity`` is either a name or an explicit ``(tversky_alpha, tversky_beta)`` pair:

- ``'tanimoto'``      AB / (AA + BB - AB), symmetric
- ``'tversky'``       Tversky with (0.95, 0.05): rewards the reference being contained in the fit
- ``'tversky_ref'``   Tversky with (1.0, 0.0): normalized by the reference only
- ``'tversky_fit'``   Tversky with (0.05, 0.95): rewards the fit being contained in the reference
- ``(a, b)``          AB / (AB + a * (AA - AB) + b * (BB - AB))

When ``a + b == 1`` the Tversky form reduces to ``AB / (a * AA + b * BB)``, the
``sigma``-form used by the pharmacophore scorers.
"""
from typing import Literal, Tuple, Union

import torch

#: A similarity choice: ``'tanimoto'``, a named Tversky variant, or explicit Tversky weights ``(a, b)``.
Similarity = Union[Literal['tanimoto', 'tversky', 'tversky_ref', 'tversky_fit'], Tuple[float, float]]

#: ``(a, b)`` weights behind each named Tversky similarity.
TVERSKY_WEIGHTS = {
    'tversky': (0.95, 0.05),
    'tversky_ref': (1.0, 0.0),
    'tversky_fit': (0.05, 0.95),
}


def tversky_weights(similarity: Similarity) -> Union[Tuple[float, float], None]:
    """``(a, b)`` for a Tversky similarity, or ``None`` for Tanimoto."""
    if isinstance(similarity, str):
        if similarity.lower() == 'tanimoto':
            return None
        try:
            return TVERSKY_WEIGHTS[similarity.lower()]
        except KeyError:
            raise ValueError(f'Unknown similarity {similarity!r}. Expected one of '
                             f"'tanimoto', {', '.join(map(repr, TVERSKY_WEIGHTS))} or an (alpha, beta) pair.")
    a, b = similarity
    return float(a), float(b)


def similarity_from_overlaps(AB: torch.Tensor,
                             AA: torch.Tensor,
                             BB: torch.Tensor,
                             similarity: Similarity = 'tanimoto') -> torch.Tensor:
    """Similarity from the cross overlap ``AB`` and self overlaps ``AA`` (reference), ``BB`` (fit)."""
    weights = tversky_weights(similarity)
    if weights is None:
        return AB / (AA + BB - AB)
    a, b = weights
    return AB / (AB + a * (AA - AB) + b * (BB - AB))


def pharm_similarity_name(similarity: Similarity) -> str:
    """The string form accepted by the pharmacophore scorers (which only take names)."""
    if isinstance(similarity, str):
        tversky_weights(similarity)  # validate
        return similarity.lower()
    for name, weights in TVERSKY_WEIGHTS.items():
        if tuple(map(float, similarity)) == weights:
            return name
    raise ValueError(f'Pharmacophore similarity must be a name, not {similarity!r}.')


def pharm_similarity(similarity: Similarity) -> Union[str, Tuple[float, float]]:
    """The form ``get_overlap_pharm`` takes: the name where one matches, else the ``(a, b)``
    weights. Matching a name keeps the named scorers' exact arithmetic."""
    if isinstance(similarity, str):
        return pharm_similarity_name(similarity)
    weights = tversky_weights(similarity)
    named = [name for name, w in TVERSKY_WEIGHTS.items() if w == weights]
    return named[0] if named else weights
