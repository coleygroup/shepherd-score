Evaluation
==========

Classes and pipelines for evaluating generated 3D conformers.

All eval classes and pipelines accept ``timeout_minutes`` to cap per-molecule xTB
wall time. :class:`~shepherd_score.evaluations.evaluate.evals.ConditionalEval`
and :class:`~shepherd_score.evaluations.evaluate.pipelines.ConditionalEvalPipeline`
accept ``priority_pharm_indices`` for subset pharmacophore Tversky scoring.

Evaluation Classes
------------------

.. automodule:: shepherd_score.evaluations.evaluate.evals
   :members:
   :undoc-members:
   :show-inheritance:

Evaluation Pipelines
--------------------

.. automodule:: shepherd_score.evaluations.evaluate.pipelines
   :members:
   :undoc-members:
   :show-inheritance:
