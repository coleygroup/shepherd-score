"""
ShEPhERD Score - 3D scoring functions for molecular evaluation
"""
from importlib.metadata import PackageNotFoundError, version

try:  # noqa: SIM105
    __version__ = version("shepherd_score")
except PackageNotFoundError:
    # package is not installed
    pass
__author__ = "Kento Abeywardane"
__email__ = "kento@mit.edu"
__description__ = "3D scoring functions used for evaluation of ShEPhERD"
