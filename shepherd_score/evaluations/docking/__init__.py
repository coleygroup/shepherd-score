"""
Module contains docking evaluation pipeline and target information.
"""

from shepherd_score.evaluations.docking.targets import docking_target_info
from shepherd_score.evaluations.docking.pipelines import DockingEvalPipeline

__all__ = [
    'docking_target_info',
    'DockingEvalPipeline'
]