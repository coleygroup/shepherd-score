"""The batched coarse-to-fine SE(3) alignment engine.

:mod:`engine` runs seed generation, the Adam fine-tuning loop and per-pair best-pose selection
for every registry mode on the dispatched kernels (:mod:`shepherd_score.accel.kernels`), with
one kernel launch per objective term (:mod:`terms`) and the CUDA-graph fine loop in
:mod:`_graphed`. :mod:`pharm` keeps the eager loop for the pharmacophore modes'
``extended_points=True`` objective, which has no kernel. Shared helpers live in
:mod:`~shepherd_score.accel.drivers._common`.
"""
