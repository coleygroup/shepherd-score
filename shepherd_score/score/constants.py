"""
Constants for 3D representations.
"""

import numpy as np
from scipy.interpolate import interp1d

### Shape
## Shape alphas
# constant num of points and no radius scaling
_npoints = np.array([50, 100, 150, 200, 300, 400])
_const_npoints_alphas = np.array([0.6011, 0.8668, 1.022, 1.118, 1.216, 1.258])
ALPHA = interp1d(_npoints, _const_npoints_alphas, 'quadratic')

### Electrostatics
COULOMB_SCALING = 1e4/(4*55.263*np.pi) # eV*A/e^2
LAM_SCALING = COULOMB_SCALING**2

### Pharmacophores
P_TYPES = ('Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'Halogen', 'Cation', 'Anion', 'ZnBinder')

# Based on Pharao's parameters
# https://github.com/gertthijs/pharao/blob/e7edc526cbfc81b3159b3d5c80e0427514118a64/include/pharmacophore.h#L104
P_ALPHAS = {
    'acceptor': 1.0,
    'donor': 1.0,
    'aromatic': 0.7,
    'hydrophobe': 0.7,
    'cation': 1.0,
    'anion': 1.0,
    'znbinder': 1.0,
    'halogen': 1.0
}