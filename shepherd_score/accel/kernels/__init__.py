"""Compute kernels for batch alignment.

Each public kernel exists in two op-for-op forms -- a Triton GPU kernel
(``*_triton.py``) and its ``numba`` CPU mirror (:mod:`~shepherd_score.accel.kernels.cpu`)
-- selected per call by tensor device via
:mod:`~shepherd_score.accel.kernels.dispatch` (CUDA tensors -> Triton, CPU
tensors -> numba, in one process).
"""
