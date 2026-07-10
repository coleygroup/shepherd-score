"""
Interaction-profile dataclasses shared by :mod:`shepherd_score.container._core`.

``Surface`` and ``Pharmacophore`` each hold one facet of a molecule's
extracted interaction profile (surface/ESP, pharmacophore)
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from shepherd_score.pharm_utils.pharmacophore import Pharmacophore


@dataclass(eq=False)
class Surface:
    """
    Container for a molecule's molecular-surface interaction profile.

    Backs the ``surf_pos``/``surf_esp``/``probe_radius`` properties that previously
    lived directly on :class:`~shepherd_score.container._core.Molecule` as loose
    attributes.

    Attributes
    ----------
    positions : np.ndarray or None
        Surface point cloud, shape (M, 3). ``None`` if no surface was generated.
    esp : np.ndarray or None
        Electrostatic potential at each surface point, shape (M,). ``None`` if not
        generated.
    probe_radius : float
        Probe radius used to define the solvent-accessible surface.
    """
    positions: Optional[np.ndarray] = None
    esp: Optional[np.ndarray] = None
    probe_radius: float = 1.2
